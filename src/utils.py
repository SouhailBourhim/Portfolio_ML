"""
utils.py — Shared helpers for DuckDB queries and logging setup.
"""

import logging
import os
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data" / "gold"


def load_params() -> dict:
    """
    Load params.yaml — the single source of truth for pipeline and backtest
    configuration (rebalance frequency, costs, weight caps, universe paths).

    Addresses: P4 — configuration read from one audited file means every
    backtest run's parameters are reproducible from git history, and the
    same values feed both `dvc repro` and the Python entry points.
    """
    params_path = ROOT / "params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")
    with open(params_path) as f:
        return yaml.safe_load(f)


def query_gold(sql: str) -> pd.DataFrame:
    """
    Run a SQL query directly against Gold-layer Parquet files via DuckDB.

    DuckDB reads Parquet natively and is 10-100x faster than pandas for
    time-series aggregations. Use this instead of pandas groupby for any
    analytical query on the Gold layer.

    Example:
        returns_2020 = query_gold(
            \"\"\"
            SELECT * FROM 'data/gold/log_returns.parquet'
            WHERE Date >= '2020-01-01' AND Date <= '2020-12-31'
            \"\"\"
        )
    """
    return duckdb.query(sql).df()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def configure_mlflow() -> str:
    """Point MLflow at a backend that actually stores metrics, and return it.

    Addresses: P4 — experiment tracking is part of the audit trail: it is how a
    reported Sharpe can be traced back to the run and parameters that produced
    it. That guarantee was silently void. Under MLflow 3.x the bare-directory
    file store is in maintenance mode, so `mlflow.log_metrics` wrote NOTHING but
    the artifacts — every run directory under `mlruns/` contains an `artifacts/`
    folder and no `meta.yaml`, and the UI renders an empty list. The runners
    believed they were tracking; nothing was persisted.

    A SQLite backend is what MLflow's own error message recommends, and
    `mlflow.db` was already in `.gitignore`, so a database backend was the
    original intent. Setting it here — once, in a shared helper the runners
    call — rather than asking each operator to export an environment variable
    keeps the guarantee in code instead of in discipline.

    An explicit `MLFLOW_TRACKING_URI` always wins, so a team server or a test
    sandbox can override it without editing anything.
    """
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    return uri
