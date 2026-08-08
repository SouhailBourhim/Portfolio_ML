"""
assets.py — Dagster asset definitions for the Phase 1 data pipeline.

Wraps the existing Bronze/Silver/Gold functions (ingest.py, clean.py,
features.py) as Dagster software-defined assets, one per medallion layer
output, so the pipeline can be scheduled and monitored from the Dagster UI
instead of run manually via `python src/pipeline.py`.

Addresses: P4 — a scheduled, versioned pipeline run is a precondition for
reproducible walk-forward backtesting later; ad-hoc manual runs make it hard
to know exactly which data snapshot produced which downstream result.

The wrapped functions still do their own file I/O (writing Parquet to
data/bronze|silver|gold/); these assets exist for scheduling, lineage, and
run history, not as a replacement data layer.
"""

import json
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from dagster import AssetExecutionContext, MetadataValue, asset

from ingest import ingest_bam_macro, ingest_bvc, ingest_macro, ingest_prices
from clean import silver_pipeline
from dividends import load_bvc_dividends
from features import gold_pipeline
from ml_features import run_phase3


@asset(group_name="bronze", description="ETF adjusted-close prices via yfinance.")
def raw_etf_prices(context: AssetExecutionContext) -> None:
    df = ingest_prices()
    context.add_output_metadata({
        "rows": df.shape[0],
        "columns": df.shape[1],
        "date_range": f"{df.index.min().date()} -> {df.index.max().date()}",
    })


@asset(group_name="bronze", description="Global macro indicators via FRED.")
def raw_fred_macro(context: AssetExecutionContext) -> None:
    df = ingest_macro()
    context.add_output_metadata({"rows": df.shape[0], "columns": df.shape[1]})


@asset(group_name="bronze", description="BVC (Bourse de Casablanca) equity prices.")
def raw_bvc_prices(context: AssetExecutionContext) -> None:
    df = ingest_bvc()
    context.add_output_metadata({"rows": df.shape[0], "columns": df.shape[1]})


@asset(group_name="bronze", description="Bank Al-Maghrib indicators (FX + policy rate).")
def raw_bam_macro(context: AssetExecutionContext) -> None:
    df = ingest_bam_macro()
    context.add_output_metadata({"rows": df.shape[0], "columns": df.shape[1]})


@asset(
    group_name="bronze",
    description=(
        "BVC per-share dividend history (amounts + ex-dates) scraped from "
        "casablanca-bourse.com. A Bronze asset in its own right because the "
        "9-asset universe's returns are WRONG without it: the ETFs arrive "
        "dividend-adjusted and the BVC names do not, so omitting this "
        "understates Moroccan assets by ~3.0-4.3%/yr (docs/DIVIDEND_BIAS.md). "
        "Wired here so the scrape refreshes on the same schedule as the prices "
        "it corrects, per the 2026-07-20 lesson (CLAUDE.md §17.7)."
    ),
)
def bvc_dividends(context: AssetExecutionContext) -> None:
    df = load_bvc_dividends()
    if df.empty:
        raise ValueError(
            "BVC dividend scrape returned no rows — refusing to report success. "
            "log_returns would silently fall back to price-only returns."
        )
    context.add_output_metadata({
        "n_dividends": len(df),
        "tickers": MetadataValue.md(", ".join(sorted(df["ticker"].unique()))),
        "date_range": f"{df['ex_date'].min()} -> {df['ex_date'].max()}",
    })


@asset(
    group_name="bronze",
    description=(
        "Official Bank Al-Maghrib USD/MAD reference rates (Cours de référence). "
        "A Bronze asset in its own right for the same reason as bvc_dividends: "
        "the 9-asset universe's returns are WRONG without it, because it is what "
        "expresses the USD-denominated ETF sleeve in the MAD numéraire. The Yahoo "
        "USDMAD=X quote in raw_bam_macro is NOT a substitute — its daily changes "
        "correlate with the official rate's at 0.028 and it overstates FX "
        "volatility 6x. Cache-first: a complete window is a no-op that only "
        "refreshes the quality report, which matters because the BAM gateway "
        "allows 5 requests/minute and a full refetch costs ~4.4 hours."
    ),
)
def bam_fx_reference(context: AssetExecutionContext) -> None:
    import subprocess

    script = ROOT / "scripts" / "backfill_bam_fx.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, cwd=str(ROOT)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"BAM FX fetch failed (exit {result.returncode}). Without it the 9-asset "
            f"universe cannot be expressed in MAD, and there is deliberately no "
            f"fallback to the Yahoo quote.\n{result.stderr[-2000:]}"
        )

    path = ROOT / "data" / "bronze" / "bam_fx_reference.parquet"
    series = pd.read_parquet(path)["USDMAD"]
    quality = json.loads(
        (ROOT / "data" / "bronze" / "bam_fx_reference_quality.json").read_text()
    )
    context.add_output_metadata({
        "n_rates": len(series),
        "date_range": f"{series.index.min().date()} -> {series.index.max().date()}",
        "density": quality["gap_structure"]["density"],
        "longest_gap_business_days":
            quality["gap_structure"]["longest_consecutive_missing_business_days"],
        "quality_gate": "PASSES" if quality["quality"]["passed"] else "FAILS",
    })


@asset(
    group_name="silver",
    # bam_fx_reference carries the OFFICIAL USD/MAD rate, which converts the
    # USD-denominated ETF sleeve into the MAD numéraire before returns are
    # computed. It is an input to the NUMBER, not just a macro feature, so it
    # belongs on this edge -- the §17.7 lesson (a Gold/Silver input invisible to
    # the asset graph goes stale silently on every scheduled run).
    # NOT raw_bam_macro: that is the Yahoo quote, retained for macro features only.
    deps=[raw_etf_prices, raw_bvc_prices, bvc_dividends, bam_fx_reference],
    description=(
        "Calendar-aligned, Pandera-validated log-returns (9-asset universe), "
        "MAD-denominated, on a TOTAL-RETURN basis for the BVC names."
    ),
)
def log_returns(context: AssetExecutionContext) -> None:
    # require_dividends: on an unattended schedule nobody reads WARNINGs, so a
    # failed dividend scrape must fail the asset instead of quietly writing a
    # price-only Silver layer that every downstream phase would then trust.
    df = silver_pipeline(require_dividends=True)
    context.add_output_metadata({
        "rows": df.shape[0],
        "n_assets": df.shape[1],
        "date_range": f"{df.index.min().date()} -> {df.index.max().date()}",
    })


@asset(
    group_name="silver",
    # Deliberately NO raw_bam_macro dependency. This universe is five
    # USD-denominated ETFs and nothing else: one numéraire, no mixed-currency
    # defect, nothing to convert (currency.resolve_currency_policy). It also
    # runs from 2004-11, and no USD/MAD series obtainable for this project
    # reaches that far back — so an FX edge here would permanently block a
    # universe that was never broken.
    deps=[raw_etf_prices],
    description=(
        "ETF-only log-returns (2004-11+, includes the GFC, COVID and the 2022 "
        "rate shock), USD-denominated — "
        "the Phase 2 dual-universe design's second backtest universe. Wired as "
        "its own asset so it refreshes every run instead of drifting stale "
        "relative to the 9-asset log_returns asset above."
    ),
)
def log_returns_etf(context: AssetExecutionContext) -> None:
    df = silver_pipeline(include_bvc=False, output_stem="log_returns_etf")
    context.add_output_metadata({
        "rows": df.shape[0],
        "n_assets": df.shape[1],
        "date_range": f"{df.index.min().date()} -> {df.index.max().date()}",
    })


@asset(
    group_name="gold",
    deps=[log_returns, log_returns_etf, raw_fred_macro, raw_bam_macro],
    description="Stationarity report + lagged macro features — Phase 2 input.",
)
def gold_layer(context: AssetExecutionContext) -> None:
    result = gold_pipeline()
    stat = result["stationarity"]
    n_stationary = int((stat["conclusion"] == "STATIONARY").sum())
    context.add_output_metadata({
        "n_stationary_assets": n_stationary,
        "pct_stationary": MetadataValue.float(round(n_stationary / len(stat), 3)),
        "n_macro_feature_rows": len(result["macro_features"]),
    })


@asset(
    group_name="gold",
    deps=[gold_layer],
    description=(
        "Phase 3 causal ML features (both universes) + reproducibility manifest — "
        "Phase 4 input, delivered to strategies via the Phase 2 engine's extras seam."
    ),
)
def ml_features_layer(context: AssetExecutionContext) -> None:
    results = run_phase3()
    # run_phase3() -> build_ml_feature_set() already raises ValueError before
    # returning an empty universe (no rows survive the warm-up filter), so this
    # can't fire today — guarded anyway so a future change to that upstream
    # invariant fails here with a clear message, not an opaque .min() crash.
    context.add_output_metadata({
        universe: MetadataValue.md(
            f"EMPTY — 0 rows (unexpected; check build_ml_feature_set warm-up filtering)"
            if df.empty else
            f"{df.shape[0]} rows x {df.shape[1]} cols, "
            f"{df.index.min().date()} -> {df.index.max().date()}"
        )
        for universe, df in results.items()
    })
