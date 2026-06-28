"""
test_schemas.py — Tests for Pandera validation schemas.
"""

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from schemas import LOG_RETURNS_SCHEMA, MACRO_FEATURES_SCHEMA, ALL_ASSETS


class TestLogReturnsSchema:
    def test_valid_data_passes(self, synthetic_log_returns):
        validated = LOG_RETURNS_SCHEMA.validate(synthetic_log_returns)
        assert validated is not None

    def test_rejects_nans(self, synthetic_log_returns):
        bad = synthetic_log_returns.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(pa.errors.SchemaError):
            LOG_RETURNS_SCHEMA.validate(bad)

    def test_rejects_extreme_returns(self, synthetic_log_returns):
        bad = synthetic_log_returns.copy()
        bad.iloc[5, bad.columns.get_loc("SPY")] = 0.99   # 99% daily return is clearly wrong
        with pytest.raises(pa.errors.SchemaError):
            LOG_RETURNS_SCHEMA.validate(bad)

    def test_rejects_too_few_rows(self, synthetic_log_returns):
        bad = synthetic_log_returns.iloc[:10]  # only 10 rows
        with pytest.raises(pa.errors.SchemaError):
            LOG_RETURNS_SCHEMA.validate(bad)

    def test_rejects_unsorted_index(self, synthetic_log_returns):
        bad = synthetic_log_returns.iloc[::-1]  # reverse order
        with pytest.raises(pa.errors.SchemaError):
            LOG_RETURNS_SCHEMA.validate(bad)

    def test_all_expected_columns_required(self, synthetic_log_returns):
        bad = synthetic_log_returns.drop(columns=["SPY"])
        with pytest.raises(pa.errors.SchemaError):
            LOG_RETURNS_SCHEMA.validate(bad)


class TestMacroFeaturesSchema:
    def test_valid_macro_passes(self, synthetic_macro):
        validated = MACRO_FEATURES_SCHEMA.validate(synthetic_macro)
        assert validated is not None

    def test_allows_nans_in_macro(self, synthetic_macro):
        with_nans = synthetic_macro.copy()
        with_nans.iloc[0, 0] = np.nan
        MACRO_FEATURES_SCHEMA.validate(with_nans)   # should not raise
