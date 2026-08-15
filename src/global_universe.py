"""
global_universe.py — The `global_2004` data layer (pre-registration §2, §3, §9).

Addresses: P1, P2, P3 — builds the first opportunity set on which the
covariance-model ladder and the HMM regime layer can be tested while being
BOTH synchronous and empirically allocation-expressive under the project's
own 25% cap.

WHY THIS UNIVERSE EXISTS. Two measured defects in the released universes,
neither of them a modelling failure:

  * `etf_2017` — with 5 assets at a 25% cap, `min_variance_lw` emits ONE
    distinct allocation across 248 rebalances (171 at a 0.30 cap). The
    arithmetic does not force that corner, but empirically the constraint
    dominates the objective, so a better covariance estimate has nowhere to
    show up.
  * `full_2021` — Casablanca and NYSE sessions barely overlap and BVC prices
    are frequently stale. Measured: BVC same-day correlation vs SPY 0.0041
    against lag-1 0.0779, a 19.1x ratio, with 17.1% zero-return days, versus
    0.4902 / -0.0287 and 0.6% for the US-listed block. The daily covariance
    matrix understates cross-market dependence.

This module builds a universe with neither defect. It does NOT touch
`log_returns.parquet` or `log_returns_etf.parquet`, and its configuration
lives in `params_global_2004.yaml` precisely so that no released runner
changes behaviour (see that file's header for the full reasoning).

SCOPE OF THIS MODULE. Data only. It computes no Sharpe ratio, no portfolio
return, and no performance quantity of any kind — the first checkpoint is a
data-readiness artifact, and paying for the evaluation before the universe is
known to be sound is how a null result becomes unattributable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

log = logging.getLogger("global_universe")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "params_global_2004.yaml"


def load_global_config(path: Path | None = None) -> dict[str, Any]:
    """Load `params_global_2004.yaml`.

    Addresses: P4 — the frozen instrument set is configuration, never a
    literal in a module, so the protocol and the code cannot drift apart
    silently. `tests/test_global_universe.py` closes the loop by parsing the
    pre-registration and asserting the two still agree.
    """
    config_path = Path(path) if path is not None else CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle)["global_2004"])


# ── Data contract ────────────────────────────────────────────────────────────
#
# Defined HERE rather than in `src/schemas.py` for the same reason the config
# is not in `params.yaml`: `schemas.py` is a declared DVC dependency of the
# `ingest`, `clean` and `features` stages, so adding a contract there would
# invalidate the entire released Gold layer and force a full pipeline rebuild
# to add a check that applies only to this experiment. Isolation is the
# approved design (pre-registration §10.3, amendment 2).

def build_global_returns_schema(tickers: list[str]):
    """Pandera contract for the `global_2004` Silver/Gold return matrix.

    Addresses: P1, P4 — the same quality floor `LOG_RETURNS_SCHEMA` imposes on
    the released universes: every frozen instrument present, no NaN, bounded
    daily moves, a sorted unique DatetimeIndex, and enough history to estimate
    a covariance matrix at all.

    The ±50% daily bound is the project's existing sanity limit. It is a
    corruption detector, not a market view: a legitimate daily log-return
    outside it would be a once-in-history event, whereas a decimal-shifted
    price feed produces them routinely.
    """
    from pandera.pandas import Check, Column, DataFrameSchema

    return DataFrameSchema(
        columns={
            ticker: Column(
                float,
                checks=[Check.in_range(-0.5, 0.5, include_min=True, include_max=True)],
                nullable=False,
                required=True,
            )
            for ticker in tickers
        },
        index=None,
        strict=True,        # no unexpected columns: the set is FROZEN
        ordered=False,
        coerce=True,
        name="global_2004_log_returns",
    )


def validate_global_returns(log_returns: pd.DataFrame, tickers: list[str],
                            min_rows: int = 5000) -> pd.DataFrame:
    """Validate the return matrix against the contract, raising on violation.

    Addresses: P4 — a contract that is defined but never invoked is
    documentation. This is called by the runner before anything downstream
    reads the matrix.
    """
    if not isinstance(log_returns.index, pd.DatetimeIndex):
        raise ValueError("global_2004 returns must carry a DatetimeIndex.")
    if not log_returns.index.is_monotonic_increasing:
        raise ValueError("global_2004 returns index must be sorted.")
    if log_returns.index.has_duplicates:
        raise ValueError("global_2004 returns index contains duplicate dates.")
    if len(log_returns) < min_rows:
        raise ValueError(
            f"global_2004 has {len(log_returns)} rows, below the {min_rows} "
            "floor. The universe exists to provide ~21 years; a short matrix "
            "means an instrument silently truncated the common history."
        )
    missing = [t for t in tickers if t not in log_returns.columns]
    if missing:
        raise ValueError(f"Frozen instruments absent from the matrix: {missing}")

    build_global_returns_schema(tickers).validate(log_returns)
    return log_returns


# ── Bronze ───────────────────────────────────────────────────────────────────

def ingest_global_prices(
    tickers: list[str],
    start: str,
    out_path: Path,
) -> pd.DataFrame:
    """Download adjusted closes for the frozen instrument set into Bronze.

    Addresses: P1, P2 — the raw price series every later estimate derives from.

    `auto_adjust=True` is mandatory project-wide (§15.3): without it, splits
    and distributions appear as fake returns. Every instrument here is a
    US-listed ETF, so unlike the BVC side there is no separate dividend
    reconstruction to do — the adjusted series IS the total return.

    Args:
        tickers: The frozen list from `params_global_2004.yaml`.
        start: Common-history start date.
        out_path: Bronze parquet destination. Written, never appended to.

    Returns:
        Wide price frame, DatetimeIndex named "Date", one column per ticker.
    """
    import yfinance as yf

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %d global instruments from %s", len(tickers), start)

    frames = []
    for ticker in tickers:
        raw = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if raw.empty or not raw["Close"].notna().any():
            raise ValueError(
                f"No data returned for {ticker!r}. The instrument set is FROZEN "
                "(pre-registration §2.2) — do not substitute a ticker to work "
                "around a download failure; retry, then record an amendment."
            )
        series = raw["Close"].copy()
        series.name = ticker
        series.index = pd.to_datetime(series.index).tz_localize(None)
        frames.append(series)

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices.sort_index()

    pq.write_table(pa.Table.from_pandas(prices), out_path)
    log.info("Bronze written: %d rows x %d columns -> %s", *prices.shape, out_path)
    return prices


# ── Silver ───────────────────────────────────────────────────────────────────

def build_global_silver(
    prices: pd.DataFrame,
    ffill_limit: int,
    out_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align to the observed trading calendar and compute log-returns.

    Addresses: P2 — log-returns are the stationary input the HMM and the
    covariance ladder require.

    ⚠️ THIS UNIVERSE DOES NOT USE `clean.align_calendars`, and the reason is
    the point of the universe. That function expands the index to every
    BUSINESS DAY and forward-fills, because it exists to reconcile Casablanca
    against NYSE — two genuinely different calendars. Every instrument here
    trades on ONE calendar, so a business-day grid does not align anything; it
    INVENTS ~9.7 rows per year (US market holidays: 5,468 observed dates
    became 5,672), forward-fills all ten columns across them, and turns each
    into an exact-zero return.

    Measured on the first run: 2,040 forward-filled cells (3.6%) and a 3.8-4.6%
    zero-return share in EVERY asset — uniform across SPY and GLD alike, which
    is the signature of a calendar artifact rather than of illiquidity, since
    no liquidity problem is uniform across a mega-cap equity ETF and a gold
    trust. Those synthetic zeros would depress measured volatility and
    correlation, contaminating the very covariance input this universe was
    built to measure cleanly.

    So the calendar here is the OBSERVED one: the dates on which the
    instruments actually traded. The project's other rules are unchanged —
    forward-fill only (never backfill, §15.4), capped at `ffill_limit`, and
    drop the leading window before every series exists.

    Returns:
        (log_returns, coverage) — the Silver matrix and the coverage facts the
        readiness artifact reports.
    """
    import sys

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from clean import compute_log_returns

    raw_missing = int(prices.isna().sum().sum())

    # The observed trading calendar: dates where at least one instrument
    # priced. No synthetic business days are introduced.
    aligned = prices.sort_index()
    aligned = aligned.loc[aligned.notna().any(axis=1)]
    # Residual per-asset gaps (a genuine missing print, not a market holiday)
    # are carried forward under the same capped, forward-only rule.
    aligned = aligned.ffill(limit=ffill_limit)
    # Drop the leading window before every series exists, so the matrix starts
    # where the universe is complete rather than where its earliest member is.
    aligned = aligned.loc[aligned.notna().all(axis=1)]

    filled_cells = int(
        aligned.notna().sum().sum() - prices.reindex(aligned.index).notna().sum().sum()
    )

    log_returns = compute_log_returns(aligned)
    log_returns = log_returns.dropna(how="any")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(log_returns), out_path)

    total_cells = int(aligned.shape[0] * aligned.shape[1])
    zero_share = {
        col: float((log_returns[col] == 0.0).mean()) for col in log_returns.columns
    }
    coverage = {
        "bronze_rows": int(len(prices)),
        "aligned_rows": int(len(aligned)),
        "silver_rows": int(len(log_returns)),
        "n_assets": int(log_returns.shape[1]),
        "raw_missing_cells": raw_missing,
        "forward_filled_cells": max(0, filled_cells),
        "forward_filled_share": float(max(0, filled_cells) / total_cells) if total_cells else 0.0,
        "zero_return_share_by_asset": zero_share,
        "max_zero_return_share": float(max(zero_share.values())) if zero_share else 0.0,
        "nan_count": int(log_returns.isna().sum().sum()),
    }
    log.info(
        "Silver written: %d rows x %d assets, %d cells forward-filled -> %s",
        len(log_returns), log_returns.shape[1], coverage["forward_filled_cells"], out_path,
    )
    return log_returns, coverage


# ── Gold ─────────────────────────────────────────────────────────────────────

def load_retained_macro(macro_exclude: list[str], root: Path | None = None) -> pd.DataFrame:
    """Load the Bronze macro levels with the Bank Al-Maghrib block removed.

    Addresses: P2, P3 — retains the global risk/rate/credit/dollar signals that
    bear on a USD multi-asset portfolio and drops the Moroccan ones that do not.

    The exclusion is economic, not convenience: `global_2004` is US-listed and
    USD-denominated, so a Moroccan policy rate has no role in it. It also
    happens to matter enormously for coverage — retaining `TAUX_DIR` would push
    the first fully dense feature row to 2017-01-04 and throw away twelve of
    the twenty-one years the universe exists to provide.
    """
    base = Path(root) if root is not None else ROOT
    frames = []
    for relative in ("data/bronze/raw_macro.parquet", "data/bronze/raw_bam_macro.parquet"):
        path = base / relative
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        log.warning("No Bronze macro found; global features will be return-only.")
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1)
    macro = macro.loc[:, ~macro.columns.duplicated()]
    dropped = [c for c in macro.columns if c in set(macro_exclude)]
    macro = macro.drop(columns=dropped, errors="ignore")
    log.info("Macro retained: %s (excluded: %s)", list(macro.columns), dropped)
    return macro


def build_global_gold(
    log_returns: pd.DataFrame,
    macro: pd.DataFrame,
    ml_config: Mapping[str, Any],
    returns_out: Path,
    features_out: Path,
    manifest_out: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Persist the Gold return matrix and the causal feature set.

    Addresses: P2, P3 — applies the audited Phase 3 feature definitions to the
    new universe unchanged, so a difference in results cannot be attributed to
    a difference in feature construction.

    Returns:
        (features, manifest) — the feature matrix and its manifest entry,
        which records leading-NaN warm-up per §15.13 rather than absorbing it.
    """
    import sys

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from ml_features import ML_CORE_FEATURES, _leading_nan_count, build_ml_feature_set

    returns_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(log_returns), returns_out)

    features = build_ml_feature_set(log_returns, macro, ml_config)
    pq.write_table(pa.Table.from_pandas(features), features_out)

    leading = {col: int(_leading_nan_count(features[col])) for col in features.columns}
    complete = features.dropna(how="any")
    manifest = {
        "universe": "global_2004",
        "rows": int(len(features)),
        "columns": list(features.columns),
        "core_features": list(ML_CORE_FEATURES),
        "global_standardization": False,
        "leading_nan_by_column": leading,
        "max_leading_nan": int(max(leading.values())) if leading else 0,
        "fully_complete_rows": int(len(complete)),
        "first_fully_complete_date": (
            str(complete.index.min().date()) if len(complete) else None
        ),
        "date_range": {
            "start": str(features.index.min().date()),
            "end": str(features.index.max().date()),
        },
        "warmup_policy": (
            "Leading NaN are REPORTED, never backfilled (AGENTS.md §15.4). Any "
            "strategy consuming a macro column on this universe must either "
            "start after first_fully_complete_date or handle the gap "
            "explicitly. The BAM block is excluded, so max_leading_nan is the "
            "DXY warm-up rather than TAUX_DIR's 3,101."
        ),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(
        "Gold written: %d feature rows, max_leading_nan=%d, first dense %s",
        len(features), manifest["max_leading_nan"], manifest["first_fully_complete_date"],
    )
    return features, manifest


# ── Readiness diagnostics ────────────────────────────────────────────────────

def measure_lag_dominance(log_returns: pd.DataFrame, reference: str = "SPY") -> dict[str, Any]:
    """Per-asset lag dominance against a reference: does ρ(0) dominate ρ(±1)?

    Addresses: P1 — detects the stale-price lead/lag signature that makes a
    daily covariance matrix understate cross-market dependence, which is the
    defect measured on `full_2021` (`docs/NONSYNC_COVARIANCE.md`).

    For each non-reference asset `i`:

        D_i = max(|ρ_i(−1)|, |ρ_i(+1)|) − |ρ_i(0)|          gate: max_i D_i ≤ 0

    Contemporaneous dependence must dominate BOTH lead and lag dependence.

    ⚠️ TWO EARLIER VERSIONS OF THIS GATE WERE WRONG, and both failures are
    worth carrying here because they are easy to repeat:

      1. A RATIO, `|ρ(1)/ρ(0)|`, thresholded at 0.25. It divides by a quantity
         that is near zero for any asset genuinely uncorrelated with the
         reference, so it exploded on GLD (ρ(0)=0.0657, ρ(1)=0.0396) and failed
         a clean universe for being diversified.
      2. An ABSOLUTE bound, `|ρ(1)| ≤ 0.20`. Stable, but useless: the
         documented stale-price block in `full_2021` has a largest lag
         correlation of only 0.0897, so it would have PASSED the known-bad
         control. A gate that passes both the clean universe and the defective
         control tests nothing.

    Lag dominance has no denominator and no tuned threshold — the bound is
    zero, fixed by the meaning of the statistic. Verified before adoption:
    `global_2004` worst −0.0248, `etf_2017` worst −0.0264, `full_2021`'s four
    BVC assets +0.0479 to +0.0807 while its four US-listed assets pass. See
    the amendment in `docs/GLOBAL_UNIVERSE_PREREGISTRATION.md` §10.3.

    WHAT IT LICENSES. "No stale-price lead/lag signature detected." NOT
    "synchrony proven" — the statistic can only fail to find the signature it
    is built to detect.
    """
    if reference not in log_returns.columns:
        raise ValueError(f"Reference {reference!r} not in the universe.")

    ref = log_returns[reference]
    per_asset: dict[str, Any] = {}
    for col in log_returns.columns:
        if col == reference:
            continue
        rho0 = abs(float(log_returns[col].corr(ref)))
        # ρ(+1): the reference leads the asset — the stale-price direction,
        # where the later-closing market has not yet reacted.
        rho_lag = abs(float(log_returns[col].corr(ref.shift(1))))
        # ρ(−1): the asset leads the reference. Checked too, so the gate is
        # symmetric and cannot be evaded by ordering the pair the other way.
        rho_lead = abs(float(log_returns[col].corr(ref.shift(-1))))
        dominance = max(rho_lag, rho_lead) - rho0
        per_asset[col] = {
            "abs_corr_same_day": round(rho0, 4),
            "abs_corr_ref_leads": round(rho_lag, 4),
            "abs_corr_asset_leads": round(rho_lead, 4),
            "lag_dominance": round(dominance, 4),
            "passes": bool(dominance <= 0.0),
            "pct_zero_days": round(float((log_returns[col] == 0.0).mean()), 4),
        }

    values = [v["lag_dominance"] for v in per_asset.values()]
    failing = sorted(k for k, v in per_asset.items() if not v["passes"])
    worst = max(per_asset, key=lambda k: per_asset[k]["lag_dominance"]) if per_asset else None
    return {
        "statistic": "D_i = max(|rho_i(-1)|, |rho_i(+1)|) - |rho_i(0)|",
        "rule": "max_i D_i <= 0",
        "reference": reference,
        "per_asset": per_asset,
        "max_lag_dominance": round(float(max(values)), 4) if values else None,
        "worst_asset": worst,
        "failing_assets": failing,
        "mean_pct_zero_days": round(float(np.mean([v["pct_zero_days"] for v in per_asset.values()])), 4),
        "interpretation": (
            "max_lag_dominance <= 0 means no stale-price lead/lag signature was "
            "detected. It does NOT prove synchrony."
        ),
    }


def verify_no_lookahead(
    log_returns: pd.DataFrame,
    features: pd.DataFrame,
    min_train_days: int,
    rebalance_freq: str,
) -> dict[str, Any]:
    """Future-corruption gate, run on this universe's REAL data.

    Addresses: P4 — the single most important correctness criterion in the
    project (§18). Mirrors `tests/test_phase3_integration.py`, but against the
    actual `global_2004` matrices rather than a synthetic fixture, because what
    could differ here is the DATA SHAPE, not the engine: this universe carries
    221 leading NaN in `DXY_DIFF_L1`, and §11 records that a macro column's
    warm-up is precisely the trap a new universe springs on the next model.

    The probe deliberately tilts on a MACRO column for that reason. A probe
    reading only the dense return features would exercise the safe path and
    prove nothing about the risky one.

    Both directions are asserted, and the second is what makes the first
    meaningful: past weights must be UNCHANGED by future corruption (no leak),
    and future weights must CHANGE (proof the probe genuinely consumes the
    feature, so the unchanged-past result is not vacuous).

    No performance quantity is produced. `run_backtest` computes returns
    internally; none is read, recorded, or interpreted here — only weights.
    """
    import sys

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from backtest import run_backtest
    from strategies import Strategy

    probe_column = "DXY_DIFF_L1"
    if probe_column not in features.columns:
        raise ValueError(f"{probe_column!r} absent; the macro block is misconfigured.")

    class _MacroProbe(Strategy):
        """Tilts on a macro column that carries a leading-NaN warm-up."""

        name = "macro_probe"

        def fit(self, train_returns, extras=None):
            assets = train_returns.columns
            n = len(assets)
            if not extras or "features" not in extras:
                return pd.Series(1.0 / n, index=assets)
            latest = extras["features"].iloc[-1]
            raw = latest.get(probe_column, np.nan)
            signal = 0.0 if pd.isna(raw) else float(np.tanh(raw))
            w0 = 0.25 + 0.5 * (0.5 * (signal + 1.0))
            weights = pd.Series((1.0 - w0) / (n - 1), index=assets)
            weights.iloc[0] = w0
            return weights

    kwargs = dict(min_train_days=min_train_days, rebalance_freq=rebalance_freq,
                  universe_name="global_2004")
    clean = run_backtest(log_returns, _MacroProbe(), extras={"features": features}, **kwargs)

    corrupted = features.copy()
    cutoff = corrupted.index[len(corrupted) // 2]
    corrupted.loc[corrupted.index > cutoff] = 99.0
    poisoned = run_backtest(log_returns, _MacroProbe(), extras={"features": corrupted}, **kwargs)

    pre = clean.target_weights.index <= cutoff
    post = clean.target_weights.index > cutoff
    past_identical = clean.target_weights.loc[pre].equals(poisoned.target_weights.loc[pre])
    future_moved = not clean.target_weights.loc[post].equals(poisoned.target_weights.loc[post])

    return {
        "probe_feature": probe_column,
        "probe_feature_leading_nan": int(features[probe_column].isna().sum()),
        "cutoff": str(cutoff.date()),
        "n_rebalances_before_cutoff": int(pre.sum()),
        "n_rebalances_after_cutoff": int(post.sum()),
        "past_weights_unchanged_by_future_corruption": bool(past_identical),
        "future_weights_did_change": bool(future_moved),
        "guarantee_is_non_vacuous": bool(future_moved),
        "note": (
            "Weights only. run_backtest computes returns internally; none is "
            "read or reported at this checkpoint."
        ),
    }


def measure_allocation_freedom(
    log_returns: pd.DataFrame,
    max_weight: float,
    rebalance_freq: str,
    min_train_days: int,
) -> dict[str, Any]:
    """Does the optimizer actually express a view under the SAME 25% cap?

    Addresses: P1, P4 — this is the design-validity check that `etf_2017`
    failed and that the whole experiment rests on. It is computed from WEIGHTS
    ONLY: distinct allocations, how many assets sit at the cap, and the
    effective number of positions. No return is accumulated, no Sharpe is
    formed, no performance quantity is produced.

    That distinction is what keeps this legitimate at the readiness checkpoint.
    Measuring whether a constraint set leaves the optimizer any freedom is a
    property of the feasible region; measuring whether the resulting portfolio
    made money is the experiment itself, and comes later.
    """
    import sys

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from strategies import MaxSharpe, MinVarianceLW

    dates = [
        d for d in pd.date_range(log_returns.index.min(), log_returns.index.max(), freq=rebalance_freq)
        if len(log_returns.loc[:d]) >= min_train_days
    ]
    out: dict[str, Any] = {"n_rebalances": len(dates), "max_weight": max_weight}

    for name, factory in (("min_variance_lw", MinVarianceLW), ("max_sharpe", MaxSharpe)):
        weights = pd.DataFrame(
            [factory(max_weight=max_weight).fit(log_returns.loc[:d]) for d in dates],
            index=dates,
        )[log_returns.columns]
        distinct = int(len(weights.round(6).drop_duplicates()))
        out[name] = {
            "distinct_allocations": distinct,
            "distinct_share": round(distinct / len(dates), 4) if dates else 0.0,
            "mean_assets_at_cap": round(float((weights > max_weight - 1e-4).sum(axis=1).mean()), 4),
            "mean_effective_positions": round(float((1.0 / (weights ** 2).sum(axis=1)).mean()), 4),
            "mean_assets_held": round(float((weights > 1e-6).sum(axis=1).mean()), 4),
        }
    return out
