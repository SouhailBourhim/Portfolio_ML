"""
test_clean.py — Tests for the Silver layer (calendar alignment + log-returns).
"""

import numpy as np
import pandas as pd
import pytest

from clean import align_calendars, compute_log_returns, flag_illiquid_assets


class TestAlignCalendars:
    def test_output_is_business_days_only(self, synthetic_prices):
        aligned = align_calendars(synthetic_prices)
        assert aligned.index.dayofweek.max() <= 4  # Mon=0 … Fri=4

    def test_no_nans_after_alignment(self, synthetic_prices):
        aligned = align_calendars(synthetic_prices)
        assert aligned.isna().sum().sum() == 0

    def test_index_is_sorted(self, synthetic_prices):
        aligned = align_calendars(synthetic_prices)
        assert aligned.index.is_monotonic_increasing

    def test_forward_fill_not_backfill(self, synthetic_prices):
        # Introduce a gap on a Monday by setting it to NaN
        prices = synthetic_prices.copy()
        monday = prices.index[7]   # index 7 is a business day
        prices.loc[monday] = np.nan
        aligned = align_calendars(prices)
        # The Monday value should equal Friday's value (forward-filled)
        friday = prices.index[6]
        assert aligned.loc[monday, "SPY"] == pytest.approx(prices.loc[friday, "SPY"])

    def test_columns_preserved(self, synthetic_prices):
        aligned = align_calendars(synthetic_prices)
        assert list(aligned.columns) == list(synthetic_prices.columns)


class TestComputeLogReturns:
    def test_shape_one_fewer_row(self, synthetic_prices):
        aligned = align_calendars(synthetic_prices)
        log_ret = compute_log_returns(aligned)
        assert len(log_ret) == len(aligned) - 1

    def test_no_nans(self, synthetic_prices):
        aligned = align_calendars(synthetic_prices)
        log_ret = compute_log_returns(aligned)
        assert log_ret.isna().sum().sum() == 0

    def test_returns_are_small_numbers(self, synthetic_prices):
        aligned = align_calendars(synthetic_prices)
        log_ret = compute_log_returns(aligned)
        assert log_ret.abs().max().max() < 1.0   # daily returns < 100%

    def test_log_returns_sum_to_total_return(self, synthetic_prices):
        """Multi-period log-return = sum of daily log-returns (time-additivity)."""
        aligned = align_calendars(synthetic_prices)
        log_ret = compute_log_returns(aligned)
        spx = aligned["SPY"]
        expected = np.log(spx.iloc[-1] / spx.iloc[0])
        actual = log_ret["SPY"].sum()
        assert actual == pytest.approx(expected, rel=1e-6)


class TestFlagIlliquidAssets:
    def test_warns_on_consecutive_zeros(self, synthetic_log_returns):
        df = synthetic_log_returns.copy()
        # Inject a 10-day trading halt into IAM.CS
        df.iloc[50:60, df.columns.get_loc("IAM.CS")] = 0.0
        with pytest.warns(UserWarning, match="illiquidity"):
            flag_illiquid_assets(df, max_consecutive_zeros=5)

    def test_no_warn_for_single_holiday_zeros(self, synthetic_log_returns):
        df = synthetic_log_returns.copy()
        # Scattered single zero-return days (holidays) — should not trigger
        df.iloc[10, df.columns.get_loc("SPY")] = 0.0
        df.iloc[50, df.columns.get_loc("SPY")] = 0.0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            flag_illiquid_assets(df, max_consecutive_zeros=5)

    def test_no_warn_for_normal_data(self, synthetic_log_returns):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            flag_illiquid_assets(synthetic_log_returns)
