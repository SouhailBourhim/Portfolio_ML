"""
utils.py — Shared helpers for DuckDB queries and logging setup.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "data" / "gold"


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
