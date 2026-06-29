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

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from dagster import AssetExecutionContext, MetadataValue, asset

from ingest import ingest_bam_macro, ingest_bvc, ingest_macro, ingest_prices
from clean import silver_pipeline
from features import gold_pipeline


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
    group_name="silver",
    deps=[raw_etf_prices, raw_bvc_prices],
    description="Calendar-aligned, Pandera-validated log-returns.",
)
def log_returns(context: AssetExecutionContext) -> None:
    df = silver_pipeline()
    context.add_output_metadata({
        "rows": df.shape[0],
        "n_assets": df.shape[1],
        "date_range": f"{df.index.min().date()} -> {df.index.max().date()}",
    })


@asset(
    group_name="gold",
    deps=[log_returns, raw_fred_macro, raw_bam_macro],
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
