"""
test_features.py — Tests for Gold layer feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from features import run_stationarity_tests, build_macro_features


class TestStationarity:
    def test_returns_dataframe_with_correct_index(self, synthetic_log_returns):
        result = run_stationarity_tests(synthetic_log_returns)
        assert set(result.index) == set(synthetic_log_returns.columns)

    def test_conclusion_column_exists(self, synthetic_log_returns):
        result = run_stationarity_tests(synthetic_log_returns)
        assert "conclusion" in result.columns

    def test_valid_conclusions(self, synthetic_log_returns):
        result = run_stationarity_tests(synthetic_log_returns)
        valid = {"STATIONARY", "NON-STATIONARY", "AMBIGUOUS"}
        assert set(result["conclusion"]).issubset(valid)

    def test_gbm_returns_are_stationary(self, synthetic_log_returns):
        result = run_stationarity_tests(synthetic_log_returns)
        # GBM-simulated returns should be stationary
        assert (result["conclusion"] == "STATIONARY").all()


class TestBuildMacroFeatures:
    def test_output_shape(self, synthetic_macro, synthetic_log_returns):
        result = build_macro_features(synthetic_macro, synthetic_log_returns.index)
        assert result.shape[1] == synthetic_macro.shape[1]

    def test_lag_prevents_same_day_data(self, synthetic_macro, synthetic_log_returns):
        # With lag=1, row 0 of features should be NaN (dropped)
        # and features should be 1 day behind macro
        result = build_macro_features(synthetic_macro, synthetic_log_returns.index, lag_days=1)
        # First available feature date must be strictly after the first macro date
        assert result.index[0] > synthetic_log_returns.index[0]

    def test_rejects_zero_lag(self, synthetic_macro, synthetic_log_returns):
        with pytest.raises(ValueError, match="lookahead"):
            build_macro_features(synthetic_macro, synthetic_log_returns.index, lag_days=0)

    def test_features_are_standardized(self, synthetic_macro, synthetic_log_returns):
        result = build_macro_features(synthetic_macro, synthetic_log_returns.index)
        # After standardization, values should be roughly in [-5, 5]
        assert result.abs().max().max() < 10
