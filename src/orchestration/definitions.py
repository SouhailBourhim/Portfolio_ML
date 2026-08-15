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
    bam_fx_reference,
    bvc_dividends,
    global_2004_data,
    gold_layer,
    log_returns,
    log_returns_etf,
    ml_features_layer,
    raw_bam_macro,
    raw_bvc_prices,
    raw_etf_prices,
    raw_fred_macro,
    raw_global_prices,
)

# The RELEASED pipeline — the assets that should refresh nightly.
#
# The selection is explicit rather than default-everything. `define_asset_job`
# with no selection takes EVERY registered asset, so registering the frozen
# global_2004 experiment for lineage would silently have put it on the daily
# schedule: its prices would be re-downloaded and its Gold layer rebuilt every
# weekday, moving the data underneath a protocol that has been committed and
# timestamped. Locked in by
# `tests/test_orchestration.py::test_the_frozen_experiment_is_not_on_the_daily_schedule`.
RELEASED_ASSETS = [
    raw_etf_prices, raw_fred_macro, raw_bvc_prices, raw_bam_macro,
    bvc_dividends, bam_fx_reference,
    log_returns, log_returns_etf, gold_layer, ml_features_layer,
]

# Name kept stable (not renamed to reflect Phases 2-3) so this doesn't fragment
# run history in an already-scheduled local Dagster daemon (see scripts/setup_launchd.sh).
pipeline_job = define_asset_job(name="phase1_pipeline_job", selection=RELEASED_ASSETS)

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
        bvc_dividends, bam_fx_reference,
        log_returns, log_returns_etf, gold_layer, ml_features_layer,
        # global_2004 experiment. Registered so the graph shows its lineage
        # (§17.7), but NOT in pipeline_job / daily_schedule: refreshing a
        # frozen experiment's data nightly would move the ground under a
        # committed protocol. Materialize on demand.
        raw_global_prices, global_2004_data,
    ],
    jobs=[pipeline_job],
    schedules=[daily_schedule],
)
