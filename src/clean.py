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
from ingest import START_DATE
# Imported at module level, NOT lazily inside silver_pipeline. The lazy import
# hid a real failure: Dagster's long-lived gRPC code server (started 2026-07-24)
# had no `dividends` module in its import state when the file landed 2026-07-25,
# so every scheduled run got 30 s into `log_returns` and died with
# ModuleNotFoundError — invisible until someone read the run history (§17.9).
# At module level the same breakage surfaces when the code location LOADS,
# which Dagster reports immediately and `dagster definitions validate` catches.
from dividends import load_bvc_dividends

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

    # dropna() below restricts the window to where ALL columns have data.
    # Log which columns are responsible and how much history is being lost
    # so a late-starting ticker (e.g. BVC equities) doesn't silently crop
    # years of ETF history off the front of the dataset.
    requested_start = reindexed.index.min()
    effective_start = reindexed.dropna().index.min()
    if pd.notna(effective_start) and effective_start > requested_start:
        lost_days = (effective_start - requested_start).days
        late_cols = reindexed.columns[reindexed.loc[requested_start].isna()].tolist()
        log.warning(
            "Calendar alignment is dropping %d days (%s -> %s) because %s "
            "have no data before that point. Effective history window starts "
            "at %s, not the requested %s.",
            lost_days, requested_start.date(), effective_start.date(),
            late_cols, effective_start.date(), requested_start.date(),
        )

    aligned = reindexed.dropna()
    log.info("Calendar alignment: %d → %d rows", len(prices), len(aligned))
    return aligned


# ── Log-returns ──────────────────────────────────────────────────────────────

def compute_log_returns(
    prices: pd.DataFrame, dividends: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Compute daily log-returns: r_t = ln((P_t + D_t) / P_{t-1}).

    Addresses: P2 — price levels have unit roots (non-stationary). Log-returns
    are stationary for most financial series, satisfying HMM and DCC-GARCH
    requirements. Time-additive: multi-period return = sum of daily log-returns.

    Why not pct_change():
    - pct_change() computes simple returns which are not time-additive.
    - np.log(P/P.shift(1)) is equivalent for small moves but better-behaved
      numerically and produces a more symmetric distribution.

    DIVIDENDS (added 2026-07-25, fixes a real bias — see docs/DIVIDEND_BIAS.md).
    The ETF side arrives dividend-adjusted (`yfinance auto_adjust=True`), but
    BVCscrap's `feature="Value"` is price-only, so the two halves of the
    universe meant different things: ETFs were total return, BVC assets were
    missing 3.6-4.3%/yr of dividends. That does not cancel in a portfolio
    comparison — `equal_weight` is forced to hold the understated assets while
    the optimizers can flee them, so it inflated every optimizer's measured
    edge. Passing `dividends` reconstructs the total return a holder actually
    earned, making both halves comparable.

    Applying a dividend on its EX-DATE is not lookahead: the ex-date is
    precisely when the price mechanically drops by the payment, so adding it
    back that day restores the holder's true return. This is the same
    convention `auto_adjust=True` already applies to the ETFs.

    Args:
        prices: Aligned close prices, wide format.
        dividends: Optional tidy frame from `dividends.load_bvc_dividends`
            with columns (ex_date, ticker, amount). Assets absent from it pass
            through unchanged — correct for ETFs, which are already total
            return. `None` reproduces the old price-only behaviour exactly.

    Returns:
        Log-returns matrix, one fewer row than input (first row dropped).
    """
    if dividends is None or dividends.empty:
        log_ret = np.log(prices / prices.shift(1)).dropna()
        log.info("Log-returns computed (price-only): %d rows × %d columns", *log_ret.shape)
        return log_ret

    payments = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    applied = skipped = 0
    for ticker, group in dividends.groupby("ticker"):
        if ticker not in prices.columns:
            continue
        column = payments.columns.get_loc(ticker)
        for ex_date, amount in zip(group["ex_date"], group["amount"]):
            position = prices.index.searchsorted(pd.Timestamp(ex_date))
            if position >= len(prices.index):
                skipped += 1          # ex-date beyond the price window
                continue
            payments.iloc[position, column] += float(amount)
            applied += 1

    log_ret = np.log((prices + payments) / prices.shift(1)).dropna()
    log.info(
        "Log-returns computed (TOTAL RETURN): %d rows × %d columns; "
        "%d dividends applied, %d outside the price window",
        *log_ret.shape, applied, skipped,
    )
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


class DividendDataUnavailable(RuntimeError):
    """
    BVC dividend history could not be loaded and the caller demanded it.

    Addresses: P4 — the price-only fallback understates every BVC asset by
    ~3.0-4.3%/yr and asymmetrically inflates the optimizers' measured edge
    (docs/DIVIDEND_BIAS.md). That is survivable when a human is watching the
    WARNING; it is not survivable on the unattended Dagster schedule, where
    nobody reads logs and the corrupted Silver layer silently feeds every
    downstream phase. `require_dividends=True` converts that WARNING into a
    hard stop for exactly those callers.
    """


def silver_pipeline(
    ffill_limit: int = 5,
    include_bvc: bool = True,
    output_stem: str = "log_returns",
    adjust_dividends: bool = True,
    require_dividends: bool = False,
) -> pd.DataFrame:
    """
    Full Bronze → Silver transformation.

    Steps: load ETF prices → merge BVC prices → calendar align →
           log-returns → illiquidity check → Pandera validate → write Parquet.

    Addresses: P1, P2. With include_bvc=False, also P3/P4 — the ETF-only
    universe keeps the full 2017+ history (the BVC merge is what truncates
    to 2021+), restoring the COVID-2020 crisis window for backtesting.

    Args:
        ffill_limit: Max consecutive business days to forward-fill.
        include_bvc: Merge BVC prices into the matrix. False produces the
            ETF-only backtest universe (Phase 2 dual-universe design).
        output_stem: Base filename for outputs. The default writes the
            canonical log_returns.parquet / validation_report.json; other
            stems write alongside without touching the canonical files.
        adjust_dividends: Reconstruct BVC total returns from the scraped
            dividend history (see `compute_log_returns` and
            docs/DIVIDEND_BIAS.md). Only meaningful when `include_bvc` is
            True — the ETFs are already dividend-adjusted at ingest. Set
            False only to reproduce the pre-2026-07-25 price-only numbers.
        require_dividends: Treat an unavailable/empty dividend history as a
            FATAL error instead of degrading to price-only returns. Defaults
            False so ad-hoc and offline use still works; every UNATTENDED
            caller (Dagster's `log_returns` asset, `pipeline.py`) passes True,
            because a WARNING nobody reads is indistinguishable from success.

    Returns:
        Validated log-returns DataFrame (wide, DatetimeIndex).

    Raises:
        DividendDataUnavailable: if `require_dividends` and the BVC dividend
            history could not be loaded or came back empty.
    """
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    prices_path = BRONZE_DIR / "raw_prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Bronze prices not found at {prices_path}. Run ingest.py first.")

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    if include_bvc:
        prices = merge_bvc_prices(prices)
    else:
        log.info("ETF-only universe requested — skipping BVC merge.")

    dividends = None
    if include_bvc and adjust_dividends:
        failure: str | None = None
        try:
            dividends = load_bvc_dividends()
        except Exception as exc:  # noqa: BLE001 — network/scrape, must not be silent
            dividends, failure = None, str(exc)

        # An EMPTY frame is the same failure wearing a success costume: the
        # scrape "worked" but yielded nothing, and compute_log_returns would
        # quietly produce price-only returns. Treated identically.
        if failure is None and (dividends is None or dividends.empty):
            failure = "the scrape returned no dividend rows"

        if failure is not None:
            dividends = None
            message = (
                f"BVC dividend history unavailable ({failure}) — returns would "
                f"fall back to PRICE-ONLY, understating BVC assets by "
                f"~3.0-4.3%/yr relative to the dividend-adjusted ETFs and "
                f"inflating every optimizer's measured edge. "
                f"See docs/DIVIDEND_BIAS.md."
            )
            # Degrading to price-only would silently reintroduce the exact bias
            # this step exists to remove (§15.13). Attended callers get a
            # WARNING they can act on; unattended ones must not proceed at all.
            if require_dividends:
                raise DividendDataUnavailable(message)
            log.warning("%s Continuing with price-only returns.", message)

    aligned = align_calendars(prices, ffill_limit=ffill_limit)
    log_returns = compute_log_returns(aligned, dividends=dividends)
    flag_illiquid_assets(log_returns)

    validated = validate_log_returns(log_returns, expect_bvc=include_bvc)

    out_path = SILVER_DIR / f"{output_stem}.parquet"
    pq.write_table(pa.Table.from_pandas(validated), out_path)
    log.info("Silver %s written: %d rows × %d columns → %s",
             output_stem, *validated.shape, out_path)

    _write_validation_report(validated, output_stem=output_stem)
    return validated


def _write_validation_report(log_returns: pd.DataFrame, output_stem: str = "log_returns") -> None:
    """Write a human-readable JSON summary of the Silver layer to data/silver/."""
    requested_start = pd.Timestamp(START_DATE)
    effective_start = log_returns.index.min()
    truncated_days = max(0, (effective_start - requested_start).days)

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_trading_days": len(log_returns),
        "n_assets": log_returns.shape[1],
        "date_range": {
            "start": str(log_returns.index.min().date()),
            "end":   str(log_returns.index.max().date()),
        },
        "requested_start_date": str(requested_start.date()),
        "history_truncated_by_days": truncated_days,
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
    report_name = (
        "validation_report.json" if output_stem == "log_returns"
        else f"validation_report_{output_stem}.json"
    )
    report_path = SILVER_DIR / report_name
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Validation report written → %s", report_path)


if __name__ == "__main__":
    silver_pipeline()                                              # 9-asset universe (2021+)
    silver_pipeline(include_bvc=False, output_stem="log_returns_etf")  # ETF universe (2017+)
