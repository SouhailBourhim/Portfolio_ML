"""
features.py — Gold layer.

Builds the ML-ready feature set from the Silver log-returns and Bronze macro data:
  • Stationarity tests (ADF + KPSS) for every return series
  • Lagged, standardized macro features aligned to the returns index
  • Writes both artifacts to data/gold/

Addresses: P2 (stationarity confirmation, non-Gaussian macro signals),
           P3 (macro series capture regime transitions before they appear in returns).

Usage:
    python src/features.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from statsmodels.tsa.stattools import adfuller, kpss

from schemas import MACRO_FEATURES_SCHEMA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("features")

ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = ROOT / "data" / "silver"
BRONZE_DIR = ROOT / "data" / "bronze"
GOLD_DIR   = ROOT / "data" / "gold"


# ── Stationarity tests ───────────────────────────────────────────────────────

def run_stationarity_tests(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Run ADF and KPSS stationarity tests on every return series.

    Addresses: P2 — confirms log-returns are stationary before passing them
    to HMM and DCC-GARCH, both of which require stationarity.

    ADF  H0: series HAS a unit root (non-stationary).
    KPSS H0: series IS stationary.

    They test opposite hypotheses — use both together:
      ADF p < 0.05  AND  KPSS p > 0.05  → STATIONARY       (both agree)
      ADF p > 0.05  AND  KPSS p < 0.05  → NON-STATIONARY   (both agree)
      otherwise                          → AMBIGUOUS — investigate manually

    Args:
        log_returns: Silver log-returns, wide format, DatetimeIndex.

    Returns:
        DataFrame indexed by asset with test statistics, p-values, and conclusion.
    """
    rows = []
    for col in log_returns.columns:
        series = log_returns[col].dropna()

        adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")

        adf_rejects  = adf_p  < 0.05   # rejects unit-root H0 → evidence of stationarity
        kpss_rejects = kpss_p < 0.05   # rejects stationarity H0 → evidence of non-stationarity

        if adf_rejects and not kpss_rejects:
            conclusion = "STATIONARY"
        elif not adf_rejects and kpss_rejects:
            conclusion = "NON-STATIONARY"
        else:
            conclusion = "AMBIGUOUS"

        rows.append({
            "asset":           col,
            "adf_stat":        round(adf_stat,  4),
            "adf_pvalue":      round(adf_p,     4),
            "adf_rejects_H0":  adf_rejects,
            "kpss_stat":       round(kpss_stat, 4),
            "kpss_pvalue":     round(kpss_p,    4),
            "kpss_rejects_H0": kpss_rejects,
            "conclusion":      conclusion,
        })
        log.debug("%s → %s (ADF p=%.3f, KPSS p=%.3f)", col, conclusion, adf_p, kpss_p)

    results = pd.DataFrame(rows).set_index("asset")

    non_stationary = results[results["conclusion"] != "STATIONARY"]
    if not non_stationary.empty:
        log.warning(
            "Non-stationary or ambiguous assets:\n%s",
            non_stationary[["adf_pvalue", "kpss_pvalue", "conclusion"]].to_string(),
        )
    else:
        log.info("All assets: STATIONARY (ADF + KPSS)")

    return results


# ── Macro features ───────────────────────────────────────────────────────────

def build_macro_features(
    raw_macro: pd.DataFrame,
    returns_index: pd.DatetimeIndex,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Prepare FRED macro indicators as ML-ready features.

    Addresses: P2, P3 — macro features provide the external signal that helps
    HMM detect regime transitions before they appear in return correlations.

    Transformations (in order):
    1. Align to the returns business-day index via forward-fill (FRED has gaps).
    2. First-difference each series — levels typically have unit roots.
    3. Standardize (z-score) so all features are on comparable scales.
    4. Lag by lag_days — features at t use only information available at t-1.
       Removing this lag is lookahead bias.

    Args:
        raw_macro: Bronze macro DataFrame, wide format, DatetimeIndex.
        returns_index: Business-day DatetimeIndex from the Silver log-returns.
        lag_days: Number of days to lag. Must be >= 1.

    Returns:
        Wide DataFrame aligned to returns_index with one column per macro series.
    """
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1 to prevent lookahead bias.")

    macro = raw_macro.reindex(returns_index, method="ffill")

    # First-difference to remove unit roots. Use dropna(how='all') so sparse
    # series (e.g. HY_SPREAD which FRED sometimes publishes with gaps) don't
    # wipe out the entire date range — NaN propagates per-column, not per-row.
    macro_diff = macro.diff().dropna(how="all")

    # Z-score per column; pandas mean()/std() skip NaN by default.
    macro_scaled = (macro_diff - macro_diff.mean()) / macro_diff.std()

    # Lag by lag_days to prevent lookahead bias. Drop rows where ALL features
    # are NaN (the leading window after shifting); keep rows with partial data.
    macro_lagged = macro_scaled.shift(lag_days).dropna(how="all")
    macro_lagged.index.name = "Date"

    n_complete = macro_lagged.notna().all(axis=1).sum()
    log.info(
        "Macro features built: %d rows × %d columns (lag=%d, %d fully-complete rows)",
        *macro_lagged.shape, lag_days, n_complete,
    )
    return macro_lagged


# ── Gold pipeline ────────────────────────────────────────────────────────────

def gold_pipeline(lag_days: int = 1) -> dict[str, pd.DataFrame]:
    """
    Full Silver → Gold transformation.

    Reads:  data/silver/log_returns.parquet
            data/bronze/raw_macro.parquet

    Writes: data/gold/log_returns.parquet       (primary model input)
            data/gold/macro_features.parquet     (lagged macro features)
            data/gold/stationarity_report.parquet

    Returns:
        Dict with keys 'log_returns', 'macro_features', 'stationarity'.
    """
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # Load silver log-returns
    lr_path = SILVER_DIR / "log_returns.parquet"
    if not lr_path.exists():
        raise FileNotFoundError(f"Silver log-returns not found at {lr_path}. Run clean.py first.")
    log_returns = pd.read_parquet(lr_path)
    log_returns.index = pd.to_datetime(log_returns.index)

    # Stationarity tests
    stat_results = run_stationarity_tests(log_returns)
    stat_path = GOLD_DIR / "stationarity_report.parquet"
    stat_results.reset_index().to_parquet(stat_path, index=False)
    log.info("Stationarity report written → %s", stat_path)

    # Copy log-returns to gold (primary model input)
    lr_gold_path = GOLD_DIR / "log_returns.parquet"
    pq.write_table(pa.Table.from_pandas(log_returns), lr_gold_path)
    log.info("Gold log_returns written → %s", lr_gold_path)

    # Macro features — FRED global + BAM Moroccan
    fred_path = BRONZE_DIR / "raw_macro.parquet"
    bam_path  = BRONZE_DIR / "raw_bam_macro.parquet"

    raw_frames = []
    if fred_path.exists():
        fred_raw = pd.read_parquet(fred_path)
        fred_raw.index = pd.to_datetime(fred_raw.index)
        raw_frames.append(fred_raw)
    else:
        log.warning("Bronze FRED macro not found — skipping global macro features.")

    if bam_path.exists():
        bam_raw = pd.read_parquet(bam_path)
        bam_raw.index = pd.to_datetime(bam_raw.index)
        raw_frames.append(bam_raw)
        log.info("BAM macro loaded: %d rows × %d columns", *bam_raw.shape)
    else:
        log.warning("Bronze BAM macro not found — run ingest_bam_macro().")

    if raw_frames:
        raw_macro = pd.concat(raw_frames, axis=1)
        macro_features = build_macro_features(raw_macro, log_returns.index, lag_days=lag_days)
        MACRO_FEATURES_SCHEMA.validate(macro_features)
        mf_path = GOLD_DIR / "macro_features.parquet"
        pq.write_table(pa.Table.from_pandas(macro_features), mf_path)
        log.info("Gold macro_features written → %s", mf_path)
    else:
        log.warning("No macro data available — skipping macro features.")
        macro_features = pd.DataFrame()

    return {
        "log_returns":    log_returns,
        "macro_features": macro_features,
        "stationarity":   stat_results,
    }


if __name__ == "__main__":
    gold_pipeline()
