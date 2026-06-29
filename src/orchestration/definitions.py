"""
definitions.py — Dagster entry point for the Phase 1 pipeline.

Defines the asset job and its daily schedule. Launch the local UI with:
    dagster dev -w workspace.yaml

The webserver started by `dagster dev` runs an embedded daemon, so the
schedule below fires automatically as long as that process is running. For
unattended scheduling (e.g. via macOS launchd) run `dagster-daemon run`
and `dagster-webserver` as separate long-lived processes instead, pointed
at a persistent DAGSTER_HOME.

Addresses: P4 — replaces manual `python src/pipeline.py` runs with a
versioned, scheduled job so every Gold-layer snapshot is reproducible and
timestamped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dagster import Definitions, ScheduleDefinition, define_asset_job

from assets import (
    gold_layer,
    log_returns,
    raw_bam_macro,
    raw_bvc_prices,
    raw_etf_prices,
    raw_fred_macro,
)

phase1_job = define_asset_job(name="phase1_pipeline_job")

# 22:00 UTC, weekdays — after BVC Casablanca closes (~14:30 UTC) and after
# NYSE closes (20:00 UTC EDT / 21:00 UTC EST), with a safety margin for both.
phase1_schedule = ScheduleDefinition(
    name="phase1_daily_refresh",
    job=phase1_job,
    cron_schedule="0 22 * * 1-5",
    execution_timezone="UTC",
)

defs = Definitions(
    assets=[raw_etf_prices, raw_fred_macro, raw_bvc_prices, raw_bam_macro, log_returns, gold_layer],
    jobs=[phase1_job],
    schedules=[phase1_schedule],
)
