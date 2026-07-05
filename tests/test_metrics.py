"""
test_metrics.py — Tests for out-of-sample performance metrics.

Hand-computed expectations wherever possible: a metrics module verified only
against itself proves nothing.
"""

import numpy as np
import pandas as pd
import pytest

from metrics import (
    annualized_return,
    annualized_sharpe,
    calmar_ratio,
    deflated_sharpe_ratio,
    information_ratio,
    max_drawdown,
    summarize,
)


@pytest.fixture()
def iid_returns() -> pd.Series:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2022-01-03", periods=500)
    return pd.Series(rng.normal(0.0004, 0.01, 500), index=dates)


class TestSharpe:
    def test_sharpe_matches_hand_computation(self):
        r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.01])
        expected = r.mean() / r.std() * np.sqrt(252)
        assert annualized_sharpe(r) == pytest.approx(expected)

    def test_zero_variance_returns_nan(self):
        r = pd.Series([0.01] * 10)
        assert np.isnan(annualized_sharpe(r))

    def test_risk_free_reduces_sharpe_of_positive_returns(self, iid_returns):
        assert annualized_sharpe(iid_returns, risk_free_annual=0.05) < annualized_sharpe(
            iid_returns, risk_free_annual=0.0
        )


class TestDrawdownAndCalmar:
    def test_max_drawdown_hand_computed(self):
        # Wealth: 1.10, 0.99, 1.0395, 0.8316 → peak 1.10, trough 0.8316
        r = pd.Series([0.10, -0.10, 0.05, -0.20])
        expected = 0.8316 / 1.10 - 1
        assert max_drawdown(r) == pytest.approx(expected, rel=1e-9)

    def test_drawdown_is_zero_for_monotonic_gains(self):
        r = pd.Series([0.01, 0.02, 0.005])
        assert max_drawdown(r) == pytest.approx(0.0)

    def test_calmar_is_annual_return_over_abs_drawdown(self):
        r = pd.Series([0.10, -0.10, 0.05, -0.20])
        assert calmar_ratio(r) == pytest.approx(annualized_return(r) / abs(max_drawdown(r)))


class TestAnnualizedReturn:
    def test_geometric_compounding_hand_computed(self):
        # Two days at +1%: total growth 1.01^2, annualized over 2 periods
        r = pd.Series([0.01, 0.01])
        expected = (1.01**2) ** (252 / 2) - 1
        assert annualized_return(r) == pytest.approx(expected)


class TestInformationRatio:
    def test_ir_vs_self_is_nan(self, iid_returns):
        assert np.isnan(information_ratio(iid_returns, iid_returns))

    def test_ir_positive_when_consistently_beating_benchmark(self, iid_returns):
        # A tiny bit of noise so tracking error is nonzero but active mean dominates
        rng = np.random.default_rng(3)
        noise = pd.Series(rng.normal(0, 1e-5, len(iid_returns)), index=iid_returns.index)
        better = iid_returns + 0.0005 + noise
        assert information_ratio(better, iid_returns) > 0


class TestDeflatedSharpe:
    def test_dsr_decreases_as_trials_increase(self, iid_returns):
        sr_daily = float(iid_returns.mean() / iid_returns.std())
        few = deflated_sharpe_ratio(iid_returns, [sr_daily, sr_daily * 0.5])
        # More trials with dispersion → higher expected max SR → lower DSR
        many = deflated_sharpe_ratio(
            iid_returns, [sr_daily * f for f in (1.0, 0.8, 0.5, 0.3, 0.1, -0.1, -0.3, 0.6, 0.2, 0.9)]
        )
        assert many < few

    def test_dsr_single_trial_degenerates_to_psr_with_warning(self, iid_returns):
        sr_daily = float(iid_returns.mean() / iid_returns.std())
        with pytest.warns(UserWarning, match="single trial"):
            dsr = deflated_sharpe_ratio(iid_returns, [sr_daily])
        assert 0.0 <= dsr <= 1.0

    def test_dsr_penalizes_negative_skew(self):
        # Two series engineered to share mean/std but differ in skew:
        # frequent small gains + rare crash vs frequent small losses + rare spike.
        dates = pd.bdate_range("2022-01-03", periods=300)
        neg_skew = pd.Series([0.002] * 299 + [-0.15], index=dates)
        pos_skew = pd.Series([-0.002] * 299 + [0.15], index=dates)
        # Rescale to identical mean and std so ONLY higher moments differ
        for s in (neg_skew, pos_skew):
            s -= s.mean()
        neg_skew = neg_skew / neg_skew.std() * 0.01 + 0.0004
        pos_skew = pos_skew / pos_skew.std() * 0.01 + 0.0004
        sr = float(neg_skew.mean() / neg_skew.std())
        trials = [sr, sr * 0.5]
        assert deflated_sharpe_ratio(neg_skew, trials) < deflated_sharpe_ratio(pos_skew, trials)

    def test_dsr_empty_trials_raises(self, iid_returns):
        with pytest.raises(ValueError, match="at least"):
            deflated_sharpe_ratio(iid_returns, [])


class TestSummarize:
    def test_summarize_reports_gross_and_net(self, iid_returns):
        costs = pd.Series(0.0002, index=iid_returns.index[::21])
        net = iid_returns.copy()
        net.iloc[::21] -= 0.0002
        out = summarize(
            net_returns=net,
            gross_returns=iid_returns,
            turnover=pd.Series([1.0, 0.3, 0.2]),
            trial_sharpes=[0.02, 0.01],
        )
        assert out["sharpe_gross"] > out["sharpe_net"]
        assert out["total_cost_drag"] > 0
        assert out["n_trials"] == 2
        assert "dsr_net" in out
