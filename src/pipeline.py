"""
pipeline.py — Phase 1 entry point.

Executes the full Bronze → Silver → Gold data pipeline with MLflow tracking.
Run this once to produce a validated, ML-ready Gold layer.

Usage:
    python src/pipeline.py
"""

import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

import mlflow

# Load .env so FRED_API_KEY is available
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from ingest import ingest_prices, ingest_macro, ingest_bvc, ingest_bam_macro
from clean import silver_pipeline
from features import gold_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("pipeline")


def run_phase1() -> None:
    """
    Execute the full Phase 1 data pipeline.

    Stages:
        1. Bronze: download prices (yfinance) + macro (FRED)
        2. Silver: calendar alignment, log-returns, Pandera validation
        3. Gold: stationarity tests, lagged macro features

    MLflow logs parameters and metrics for every run so results are
    reproducible and comparable across experiments.
    """
    mlflow.set_experiment("phase1_data_pipeline")

    with mlflow.start_run(run_name="phase1_full"):

        # ── Bronze ────────────────────────────────────────────────────────────
        log.info("=== BRONZE: ingesting raw data ===")
        raw_prices  = ingest_prices()
        raw_macro   = ingest_macro()
        bvc_prices  = ingest_bvc()
        bam_macro   = ingest_bam_macro()

        mlflow.log_params({
            "n_etf_tickers":    raw_prices.shape[1],
            "n_bvc_tickers":    bvc_prices.shape[1],
            "bvc_start":        str(bvc_prices.index.min().date()),
            "etf_start":        str(raw_prices.index.min().date()),
            "end_date":         str(raw_prices.index.max().date()),
            "n_fred_series":    raw_macro.shape[1],
            "n_bam_series":     bam_macro.shape[1],
        })

        # ── Silver ────────────────────────────────────────────────────────────
        log.info("=== SILVER: cleaning and validating ===")
        log_returns = silver_pipeline()

        mlflow.log_metrics({
            "n_trading_days": len(log_returns),
            "n_assets":       log_returns.shape[1],
        })

        # ── Gold ──────────────────────────────────────────────────────────────
        log.info("=== GOLD: feature engineering ===")
        gold = gold_pipeline()

        stat = gold["stationarity"]
        n_stationary = (stat["conclusion"] == "STATIONARY").sum()
        mlflow.log_metrics({
            "n_stationary_assets": int(n_stationary),
            "pct_stationary":      round(n_stationary / len(stat), 3),
            "n_macro_feature_rows": len(gold["macro_features"]),
        })

        mlflow.log_artifacts(str(Path(__file__).resolve().parents[1] / "data" / "gold"),
                             artifact_path="gold_layer")

        log.info("=== Phase 1 complete. Gold layer ready for Phase 2. ===")
        log.info("Stationarity summary:\n%s",
                 stat[["adf_pvalue", "kpss_pvalue", "conclusion"]].to_string())


if __name__ == "__main__":
    try:
        run_phase1()
    except Exception as exc:
        log.error("Phase 1 failed: %s", exc, exc_info=True)
        sys.exit(1)
