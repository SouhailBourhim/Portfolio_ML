"""
utils.py — Shared helpers for DuckDB queries and logging setup.
"""

import logging
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
