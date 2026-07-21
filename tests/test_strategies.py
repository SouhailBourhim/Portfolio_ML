"""
test_strategies.py — Tests for the strategy interface and Markowitz baselines.
"""

import numpy as np
import pandas as pd
import pytest

from strategies import (
    DCCGarchStrategy,
    EqualWeight,
    MaxSharpe,
    MinVariance,
    MinVarianceEWMA,
    MinVarianceLW,
    RandomForestSignalStrategy,
    XGBoostSignalStrategy,
)

# LSTMSignalStrategy is deliberately excluded here: retraining a torch LSTM
# 5 times per invariant test (parametrize x extras-comparison) on the full
# 599-row/9-asset synthetic fixture made this suite impractically slow on
# this machine. LSTMSignalStrategy has its own fast, isolated coverage in
# tests/test_lstm_signal.py (10 tests, ~6s, tiny fixtures) and is excluded
# from src/run_phase4b.py's live comparison for the same reason — deferred,
# not abandoned. See CLAUDE.md Phase 4B notes.
ALL_STRATEGIES = [
    EqualWeight(),
    MinVariance(),
    MinVarianceLW(),
    MinVarianceEWMA(),
    DCCGarchStrategy(),
    MaxSharpe(),
    RandomForestSignalStrategy(min_train_rows=50),
    XGBoostSignalStrategy(min_train_rows=50),
]


@pytest.fixture()
def high_low_vol_returns() -> pd.DataFrame:
    """Two assets: LOW has 30x less volatility than HIGH; HIGH has the better Sharpe."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2022-01-03", periods=400)
    return pd.DataFrame(
        {
            "LOW":  rng.normal(0.0000, 0.001, 400),
            "HIGH": rng.normal(0.0020, 0.030, 400),
        },
        index=dates,
    )


class TestWeightInvariants:
    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
    def test_all_strategies_return_weights_summing_to_one(self, strategy, synthetic_log_returns):
        w = strategy.fit(synthetic_log_returns)
        assert w.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
    def test_long_only_no_negative_weights(self, strategy, synthetic_log_returns):
        w = strategy.fit(synthetic_log_returns)
        assert (w >= 0).all()

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
    def test_weights_indexed_by_asset_columns(self, strategy, synthetic_log_returns):
        w = strategy.fit(synthetic_log_returns)
        assert list(w.index) == list(synthetic_log_returns.columns)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
    def test_baselines_accept_and_ignore_extras(self, strategy, synthetic_log_returns):
        extras = {"macro": pd.DataFrame({"VIX": [1.0]}, index=[synthetic_log_returns.index[0]])}
        w_with = strategy.fit(synthetic_log_returns, extras=extras)
        w_without = strategy.fit(synthetic_log_returns)
        pd.testing.assert_series_equal(w_with, w_without)


class TestEqualWeight:
    def test_equal_weight_is_one_over_n(self, synthetic_log_returns):
        w = EqualWeight().fit(synthetic_log_returns)
        n = synthetic_log_returns.shape[1]
        assert w.tolist() == pytest.approx([1.0 / n] * n)


class TestConstraints:
    def test_max_weight_cap_is_respected(self, high_low_vol_returns):
        # Unconstrained min-var would go ~100% into LOW; the cap must bind.
        w = MinVariance(max_weight=0.60).fit(high_low_vol_returns)
        assert w.max() <= 0.60 + 1e-9
        assert w["LOW"] == pytest.approx(0.60, abs=1e-6)

    def test_infeasible_cap_raises(self, synthetic_log_returns):
        # 9 assets x 0.10 cap = 0.9 < 1 -> impossible
        with pytest.raises(ValueError, match="Infeasible"):
            MinVariance(max_weight=0.10).fit(synthetic_log_returns)


class TestOptimizerEconomics:
    def test_min_variance_prefers_low_volatility_asset(self, high_low_vol_returns):
        w = MinVariance(max_weight=1.0).fit(high_low_vol_returns)
        assert w["LOW"] > 0.9

    def test_max_sharpe_prefers_high_sharpe_asset(self, high_low_vol_returns):
        # HIGH: mean 0.002/std 0.03 -> daily SR ~0.067; LOW: SR ~0
        w = MaxSharpe(max_weight=1.0).fit(high_low_vol_returns)
        assert w["HIGH"] > w["LOW"]

    def test_lw_min_variance_prefers_low_volatility_asset(self, high_low_vol_returns):
        # Shrinkage regularizes the covariance; it must not invert the economics
        w = MinVarianceLW(max_weight=1.0).fit(high_low_vol_returns)
        assert w["LOW"] > 0.9

    def test_lw_weights_differ_from_sample_min_variance(self, synthetic_log_returns):
        # On a 9-asset noisy panel the shrunk and sample covariances differ,
        # so the optima should too - if they were identical, LW would be dead code
        w_sample = MinVariance(max_weight=1.0).fit(synthetic_log_returns)
        w_lw = MinVarianceLW(max_weight=1.0).fit(synthetic_log_returns)
        assert (w_sample - w_lw).abs().max() > 1e-4

    def test_ewma_min_variance_prefers_low_volatility_asset(self, high_low_vol_returns):
        # Sanity floor: EWMA must not invert basic min-variance economics either
        w = MinVarianceEWMA(max_weight=1.0).fit(high_low_vol_returns)
        assert w["LOW"] > 0.9

    def test_ewma_reacts_faster_than_lw_to_a_volatility_shift(self):
        # A is calm for the first half of the window, then turns volatile; B stays
        # calm throughout. A flat-window estimator (LW) still credits A's calm
        # past, diluting the shift; EWMA's recency weighting has already priced
        # the recent spike in and should shed more of A than LW does. This is the
        # load-bearing proof that EWMA is genuinely "dynamic" (P2), not a rename
        # of the same static estimate MinVarianceLW already provides.
        rng = np.random.default_rng(11)
        n = 300
        dates = pd.bdate_range("2022-01-03", periods=n)
        a_vol = np.concatenate([np.full(n // 2, 0.001), np.full(n - n // 2, 0.03)])
        b_vol = np.full(n, 0.001)
        returns = pd.DataFrame(
            {"A": rng.normal(0.0, a_vol), "B": rng.normal(0.0, b_vol)}, index=dates
        )

        w_lw = MinVarianceLW(max_weight=1.0).fit(returns)
        w_ewma = MinVarianceEWMA(max_weight=1.0, halflife_days=20).fit(returns)

        assert w_ewma["A"] < w_lw["A"]
