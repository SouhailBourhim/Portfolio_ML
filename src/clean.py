"""
clean.py — Silver layer.

Transforms Bronze prices into a clean, calendar-aligned log-returns matrix
and validates the result with Pandera.

Addresses: P1 (removes calendar misalignment that inflates correlations),
           P2 (log-returns are stationary; price levels are not).

Usage:
    python src/clean.py
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schemas import validate_log_returns, ALL_ASSETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("clean")

ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"


# ── Calendar alignment ───────────────────────────────────────────────────────

def align_calendars(prices: pd.DataFrame, ffill_limit: int = 5) -> pd.DataFrame:
    """
    Align BVC (Casablanca) and NYSE trading calendars to a common business-day grid.

    Addresses: P1 — misaligned calendars introduce spurious zeros in return
    series that inflate apparent correlation estimates.

    Strategy:
    - Expand index to cover every business day in the date range.
    - Forward-fill only (last known price). Backfill is forbidden: it uses
      future prices to fill past gaps, which is lookahead bias.
    - ffill_limit caps consecutive fills to avoid propagating stale prices
      across long holiday stretches.
    - Drop the initial window where not all series have data yet.

    Args:
        prices: Raw adjusted close prices, wide format, DatetimeIndex.
        ffill_limit: Max consecutive business days to forward-fill.

    Returns:
        Prices aligned to a full business-day grid with no remaining NaNs.
    """
    prices.index = pd.to_datetime(prices.index)
    bday_index = pd.bdate_range(prices.index.min(), prices.index.max())
    reindexed = prices.reindex(bday_index).ffill(limit=ffill_limit)
    reindexed.index.name = "Date"

    # Columns with zero non-NaN values never had data — fail fast rather than
    # producing a 0-row DataFrame after dropna().
    all_nan_cols = reindexed.columns[reindexed.isna().all()].tolist()
    if all_nan_cols:
        raise ValueError(
            f"Tickers have no price data at all (likely rate-limited or delisted): "
            f"{all_nan_cols}. Re-run ingest.py to retry the download."
        )

    aligned = reindexed.dropna()
    log.info("Calendar alignment: %d → %d rows", len(prices), len(aligned))
    return aligned


# ── Log-returns ──────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns: r_t = ln(P_t / P_{t-1}).

    Addresses: P2 — price levels have unit roots (non-stationary). Log-returns
    are stationary for most financial series, satisfying HMM and DCC-GARCH
    requirements. Time-additive: multi-period return = sum of daily log-returns.

    Why not pct_change():
    - pct_change() computes simple returns which are not time-additive.
    - np.log(P/P.shift(1)) is equivalent for small moves but better-behaved
      numerically and produces a more symmetric distribution.

    Args:
        prices: Aligned adjusted close prices, wide format.

    Returns:
        Log-returns matrix, one fewer row than input (first row dropped).
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    log.info("Log-returns computed: %d rows × %d columns", *log_ret.shape)
    return log_ret


# ── Illiquidity flag ─────────────────────────────────────────────────────────

def flag_illiquid_assets(log_returns: pd.DataFrame, max_consecutive_zeros: int = 5) -> None:
    """
    Warn if any asset has a run of consecutive zero-return days.

    Addresses: P1 — BVC stocks can have long stretches of zero trading volume.
    Illiquid assets produce unreliable covariance estimates.

    Why consecutive runs (not total count):
    - 1-2 consecutive zeros are normal: they correspond to market holidays that
      fell on a business day (e.g. July 4th, Eid Al-Fitr). Forward-filling
      produces exactly one zero-return per holiday, which is correct behavior.
    - 5+ consecutive zeros suggest a trading halt or genuine illiquidity.
    """
    def max_run(s: pd.Series) -> int:
        is_zero = (s.abs() < 1e-10).astype(int)
        groups = (is_zero != is_zero.shift()).cumsum()
        return int(is_zero.groupby(groups).sum().max())

    max_runs = log_returns.apply(max_run)
    suspicious = max_runs[max_runs >= max_consecutive_zeros]
    if not suspicious.empty:
        warnings.warn(
            f"Possible illiquidity — assets with {max_consecutive_zeros}+ consecutive zero-return days:\n"
            f"{suspicious.to_string()}",
            UserWarning,
            stacklevel=2,
        )
        log.warning("Illiquid assets flagged: %s", suspicious.index.tolist())


# ── Silver pipeline ──────────────────────────────────────────────────────────

def merge_bvc_prices(etf_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BVC prices into the ETF price matrix if available.

    BVC data starts June 2021 (medias24 free-tier limit). The merged matrix
    will have NaN for BVC columns before that date. calendar alignment and
    dropna() in align_calendars() will naturally restrict the usable window
    to the period where ALL columns have data.

    Addresses: P3 — BVC equities provide Moroccan-market exposure that is
    essential for a portfolio managed by a Moroccan institution.
    """
    bvc_path = BRONZE_DIR / "bvc_prices.parquet"
    if not bvc_path.exists():
        log.info("No BVC prices found at %s — running ETFs only.", bvc_path)
        return etf_prices

    bvc = pd.read_parquet(bvc_path)
    bvc.index = pd.to_datetime(bvc.index)
    bvc.index.name = "Date"

    merged = etf_prices.join(bvc, how="outer")
    log.info(
        "BVC prices merged: %d BVC columns added, matrix now %d cols",
        len(bvc.columns), merged.shape[1],
    )
    return merged


def silver_pipeline(ffill_limit: int = 5) -> pd.DataFrame:
    """
    Full Bronze → Silver transformation.

    Steps: load ETF prices → merge BVC prices → calendar align →
           log-returns → illiquidity check → Pandera validate → write Parquet.

    Returns:
        Validated log-returns DataFrame (wide, DatetimeIndex).
    """
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    prices_path = BRONZE_DIR / "raw_prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Bronze prices not found at {prices_path}. Run ingest.py first.")

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = merge_bvc_prices(prices)

    aligned = align_calendars(prices, ffill_limit=ffill_limit)
    log_returns = compute_log_returns(aligned)
    flag_illiquid_assets(log_returns)

    validated = validate_log_returns(log_returns)

    out_path = SILVER_DIR / "log_returns.parquet"
    pq.write_table(pa.Table.from_pandas(validated), out_path)
    log.info("Silver log_returns written: %d rows × %d columns → %s",
             *validated.shape, out_path)

    _write_validation_report(validated)
    return validated


def _write_validation_report(log_returns: pd.DataFrame) -> None:
    """Write a human-readable JSON summary of the Silver layer to data/silver/."""
    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_trading_days": len(log_returns),
        "n_assets": log_returns.shape[1],
        "date_range": {
            "start": str(log_returns.index.min().date()),
            "end":   str(log_returns.index.max().date()),
        },
        "assets_present": list(log_returns.columns),
        "assets_missing": [a for a in ALL_ASSETS if a not in log_returns.columns],
        "nan_count": int(log_returns.isna().sum().sum()),
        "pandera_validation": "PASSED",
        "return_stats": {
            col: {
                "mean_annualised": round(log_returns[col].mean() * 252, 6),
                "vol_annualised":  round(log_returns[col].std() * (252 ** 0.5), 6),
                "skewness":        round(float(log_returns[col].skew()), 4),
                "excess_kurtosis": round(float(log_returns[col].kurt()), 4),
            }
            for col in log_returns.columns
        },
    }
    report_path = SILVER_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Validation report written → %s", report_path)


if __name__ == "__main__":
    silver_pipeline()
