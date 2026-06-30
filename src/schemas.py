"""
schemas.py — Pandera data contracts for the portfolio ML pipeline.

Each schema is the single source of truth for the shape and quality
requirements of data at that layer. Validation is called explicitly
in the pipeline; schemas are never imported just for type hints.

Addresses: P1 — ensures covariance inputs are clean and correctly typed.
"""

import logging
import warnings

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column, Check

log = logging.getLogger(__name__)

# Full intended asset universe — BVC equities + international ETFs
BVC_ASSETS = ["IAM.CS", "ATW.CS", "CIH.CS", "BCP.CS"]
ETF_ASSETS = ["SPY", "QQQ", "EEM", "GLD", "TLT"]
ALL_ASSETS = BVC_ASSETS + ETF_ASSETS


def validate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate log-returns DataFrame.

    Addresses: P1 — ensures covariance inputs are clean and correctly typed.

    Builds the schema dynamically from whichever columns are present.
    Warns (does not fail) if BVC tickers are missing — those tickers may need
    to be sourced manually from the Bourse de Casablanca and loaded from CSV.

    Args:
        df: Wide log-returns DataFrame, DatetimeIndex × asset columns.

    Returns:
        Validated DataFrame.

    Raises:
        pandera.errors.SchemaError: if any present column violates quality rules.
        ValueError: if none of the expected ETF tickers are present.
    """
    present = [a for a in ALL_ASSETS if a in df.columns]
    missing = [a for a in ALL_ASSETS if a not in df.columns]
    bvc_missing = [a for a in BVC_ASSETS if a not in df.columns]

    if not present:
        raise ValueError("No expected asset columns found in log-returns DataFrame.")

    if bvc_missing:
        warnings.warn(
            f"BVC tickers missing from log-returns (Yahoo Finance may not cover them). "
            f"Missing: {bvc_missing}. "
            f"To add them manually, place a CSV with columns Date,IAM.CS,ATW.CS,CIH.CS,BCP.CS "
            f"in data/bronze/bvc_prices.csv and re-run the pipeline.",
            UserWarning,
            stacklevel=2,
        )

    if missing:
        log.warning("Missing expected tickers: %s", missing)

    schema = DataFrameSchema(
        columns={
            asset: Column(
                dtype=float,
                nullable=False,
                checks=Check.in_range(-0.5, 0.5,
                    error=f"{asset}: daily return outside ±50% — likely data error or unadjusted split"),
            )
            for asset in present
        },
        index=pa.Index(pa.DateTime, name="Date", coerce=True),
        checks=[
            Check(lambda df: df.shape[0] >= 500,
                  error="Fewer than 500 rows — insufficient history for reliable covariance estimation."),
            Check(lambda df: df.index.is_monotonic_increasing,
                  error="DatetimeIndex is not sorted ascending — downstream models will be wrong."),
            Check(lambda df: df.isna().sum().sum() == 0,
                  error="NaN values in log-returns — alignment or cleaning step failed."),
        ],
        coerce=True,
        strict=False,   # allow extra columns (e.g. if BVC data is added later)
    )
    return schema.validate(df)


# Keep a static schema export for tests that use synthetic data with all 9 columns
LOG_RETURNS_SCHEMA = DataFrameSchema(
    columns={
        asset: Column(
            dtype=float,
            nullable=False,
            checks=Check.in_range(-0.5, 0.5,
                error=f"{asset}: daily return outside ±50% — likely data error or unadjusted split"),
        )
        for asset in ALL_ASSETS
    },
    index=pa.Index(pa.DateTime, name="Date", coerce=True),
    checks=[
        Check(lambda df: df.shape[0] >= 500,
              error="Fewer than 500 rows — insufficient history for reliable covariance estimation."),
        Check(lambda df: df.index.is_monotonic_increasing,
              error="DatetimeIndex is not sorted ascending — downstream models will be wrong."),
        Check(lambda df: df.isna().sum().sum() == 0,
              error="NaN values in log-returns — alignment or cleaning step failed."),
    ],
    coerce=True,
)

# ── Gold: macro features ─────────────────────────────────────────────────────
# strict=False throughout — BAM series (USDMAD, EURMAD, TAUX_DIR) are optional
# additions; the schema validates the ones that are present.

MACRO_FEATURES_SCHEMA = DataFrameSchema(
    columns={
        # required=False: FRED series can be absent if the API call failed or
        # a series was unavailable — schema validates whichever columns exist,
        # rather than failing the whole Gold layer because of one missing series.
        series: Column(dtype=float, nullable=True, required=False)
        for series in ["VIX", "US10Y", "DXY", "HY_SPREAD"]
    },
    index=pa.Index(pa.DateTime, name="Date", coerce=True),
    checks=[
        Check(lambda df: df.index.is_monotonic_increasing,
              error="Macro features DatetimeIndex is not sorted ascending."),
    ],
    coerce=True,
    strict=False,
)
