"""
definitions.py — Dagster entry point for the data pipeline (Phases 1-3).

Defines the asset job and its daily schedule. Launch the local UI with:
    dagster dev -w workspace.yaml

The webserver started by `dagster dev` runs an embedded daemon, so the
schedule below fires automatically as long as that process is running. For
unattended scheduling (e.g. via macOS launchd) run `dagster-daemon run`
and `dagster-webserver` as separate long-lived processes instead, pointed
at a persistent DAGSTER_HOME.

Addresses: P4 — replaces manual `python src/pipeline.py` runs with a
versioned, scheduled job so every Gold-layer snapshot is reproducible and
timestamped. Both backtest universes (log_returns, log_returns_etf) and the
Phase 3 ML features are all registered here — a snapshot missing any one of
them is exactly the silent drift this project treats as a bug (see the
2026-07-20 incident: the ETF-only universe and ml_features were never wired
into this job, so it drifted stale relative to the 9-asset universe every
time the schedule fired).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dagster import Definitions, ScheduleDefinition, define_asset_job

from assets import (
    bvc_dividends,
    gold_layer,
    log_returns,
    log_returns_etf,
    ml_features_layer,
    raw_bam_macro,
    raw_bvc_prices,
    raw_etf_prices,
    raw_fred_macro,
)

# Name kept stable (not renamed to reflect Phases 2-3) so this doesn't fragment
# run history in an already-scheduled local Dagster daemon (see scripts/setup_launchd.sh).
pipeline_job = define_asset_job(name="phase1_pipeline_job")

# 22:00 UTC, weekdays — after BVC Casablanca closes (~14:30 UTC) and after
# NYSE closes (20:00 UTC EDT / 21:00 UTC EST), with a safety margin for both.
daily_schedule = ScheduleDefinition(
    name="phase1_daily_refresh",
    job=pipeline_job,
    cron_schedule="0 22 * * 1-5",
    execution_timezone="UTC",
)

defs = Definitions(
    assets=[
        raw_etf_prices, raw_fred_macro, raw_bvc_prices, raw_bam_macro,
        bvc_dividends,
        log_returns, log_returns_etf, gold_layer, ml_features_layer,
    ],
    jobs=[pipeline_job],
    schedules=[daily_schedule],
)
