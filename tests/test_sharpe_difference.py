"""test_sharpe_difference.py — the studentized Sharpe test must be calibrated.

The reason this test exists at all is calibration, not point estimation: the
existing percentile bootstrap already reports a difference, and the claim being
made is that the studentized version is better SIZED. So the suite checks size
and coverage under a known null, not merely that numbers come out.

Positive control first, as everywhere else here: a procedure that never rejects
would agree with every conclusion this project has reached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inference import sharpe_difference_mde, sharpe_difference_test
from metrics import annualized_sharpe


def _pair(n=750, edge=0.0, seed=0, vol=0.01, ar=0.0):
    """Two return series; `edge` is added to the first. `ar` adds AR(1) memory."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)

    def draw():
        eps = rng.normal(0.0004, vol, n)
        if ar:
            out = np.empty(n)
            out[0] = eps[0]
            for t in range(1, n):
                out[t] = ar * out[t - 1] + eps[t]
            return out
        return eps

    return pd.Series(draw() + edge, index=dates), pd.Series(draw(), index=dates)


class TestPositiveControl:
    def test_a_large_genuine_gap_is_detected(self):
        a, b = _pair(edge=0.0015, seed=1)
        res = sharpe_difference_test(a, b, n_boot=500, seed=0)
        assert res["difference"] > 0
        assert res["p_value_studentized_bootstrap"] < 0.05, (
            f"a persistent 15bps/day edge over 750 days was not detected "
            f"(p = {res['p_value_studentized_bootstrap']})"
        )

    def test_the_interval_excludes_zero_when_the_gap_is_large(self):
        a, b = _pair(edge=0.0015, seed=2)
        lo, hi = sharpe_difference_test(a, b, n_boot=500, seed=0)["confidence_interval"]
        assert lo > 0 or hi < 0


class TestNullBehaviour:
    def test_identical_distributions_are_not_flagged(self):
        a, b = _pair(edge=0.0, seed=3)
        res = sharpe_difference_test(a, b, n_boot=500, seed=0)
        assert res["p_value_studentized_bootstrap"] > 0.05

    def test_size_is_near_nominal_under_the_null(self):
        """The property that justifies the method: rejection rate ~= alpha.

        A percentile bootstrap of a non-pivotal statistic over-rejects here;
        the studentized version should sit near its nominal 10%.
        """
        rejections = 0
        trials = 40
        for seed in range(trials):
            a, b = _pair(n=500, edge=0.0, seed=1000 + seed)
            res = sharpe_difference_test(a, b, n_boot=300, alpha=0.10, seed=0)
            if res["p_value_studentized_bootstrap"] < 0.10:
                rejections += 1
        rate = rejections / trials
        assert rate <= 0.30, (
            f"rejected {rejections}/{trials} = {rate:.2f} of true nulls at a "
            f"nominal 10% level; the test is badly over-sized"
        )


class TestAgreementWithTheProjectsConventions:
    def test_the_reported_sharpes_track_annualized_sharpe_to_the_ddof_convention(self):
        """Same EXCESS convention; the ddof convention differs, by design.

        The delta method is derived on sigma^2 = gamma - mu^2, a ddof=0
        variance, while `annualized_sharpe` uses pandas' ddof=1. Both cannot
        hold exactly. This function is internally consistent on the
        Ledoit-Wolf basis, so the residual gap against `annualized_sharpe` is
        O(1/n) and nothing more — pinned loosely here so a genuine divergence
        in the excess-return handling would still be caught.
        """
        a, b = _pair(edge=0.0005, seed=4)
        res = sharpe_difference_test(a, b, n_boot=200, risk_free_annual=0.03, seed=0)
        assert res["sharpe_candidate"] == pytest.approx(
            annualized_sharpe(a, risk_free_annual=0.03), rel=5e-3
        )
        assert res["sharpe_benchmark"] == pytest.approx(
            annualized_sharpe(b, risk_free_annual=0.03), rel=5e-3
        )

    def test_the_ddof_gap_is_the_expected_order_and_no_larger(self):
        """O(1/n): the ratio of the two conventions is sqrt((n-1)/n)."""
        a, b = _pair(n=750, edge=0.0005, seed=4)
        res = sharpe_difference_test(a, b, n_boot=200, seed=0)
        expected_ratio = np.sqrt(750 / 749)  # ddof=0 std is smaller -> Sharpe larger
        assert res["sharpe_candidate"] / annualized_sharpe(a) == pytest.approx(
            expected_ratio, rel=1e-9
        )

    def test_the_difference_is_the_difference_of_those_sharpes(self):
        a, b = _pair(edge=0.0005, seed=5)
        res = sharpe_difference_test(a, b, n_boot=200, seed=0)
        assert res["difference"] == pytest.approx(
            res["sharpe_candidate"] - res["sharpe_benchmark"], rel=1e-6
        )


class TestSerialDependenceIsHandled:
    def test_autocorrelation_widens_the_standard_error(self):
        """HAC exists for this; an iid SE would understate it."""
        a_iid, b_iid = _pair(n=600, seed=6, ar=0.0)
        a_ar, b_ar = _pair(n=600, seed=6, ar=0.5)
        se_iid = sharpe_difference_test(a_iid, b_iid, n_boot=200, seed=0)["standard_error"]
        se_ar = sharpe_difference_test(a_ar, b_ar, n_boot=200, seed=0)["standard_error"]
        assert se_ar > se_iid


class TestDeterminism:
    def test_the_same_seed_gives_the_same_result(self):
        a, b = _pair(seed=7)
        assert sharpe_difference_test(a, b, n_boot=300, seed=0) == sharpe_difference_test(
            a, b, n_boot=300, seed=0
        )


class TestInputDiscipline:
    def test_misaligned_indexes_are_refused(self):
        a, b = _pair(seed=8)
        b.index = pd.bdate_range("2022-02-01", periods=len(b))
        with pytest.raises(ValueError, match="identical date indexes"):
            sharpe_difference_test(a, b)

    def test_nan_is_refused(self):
        a, b = _pair(seed=9)
        a.iloc[3] = np.nan
        with pytest.raises(ValueError, match="NaN-free"):
            sharpe_difference_test(a, b)

    def test_a_zero_variance_series_is_refused_not_silently_nan(self):
        dates = pd.bdate_range("2022-01-03", periods=100)
        flat = pd.Series(np.full(100, 0.001), index=dates)
        _, b = _pair(n=100, seed=10)
        b.index = dates
        with pytest.raises(ValueError, match="degenerate sample"):
            sharpe_difference_test(flat, b, n_boot=50)


class TestMinimumDetectableEffect:
    def test_mde_matches_the_closed_form(self):
        from scipy.stats import norm

        se = 0.25
        expected = (norm.ppf(0.95) + norm.ppf(0.80)) * se
        assert sharpe_difference_mde(se, alpha=0.10, power=0.80) == pytest.approx(expected)

    def test_more_power_demands_a_larger_effect(self):
        assert sharpe_difference_mde(0.2, power=0.90) > sharpe_difference_mde(0.2, power=0.80)

    def test_a_longer_sample_lowers_the_detectable_effect(self):
        """Root-n scaling: quadrupling the sample halves the MDE."""
        base = sharpe_difference_mde(0.4, n_observed=455, n_target=455)
        longer = sharpe_difference_mde(0.4, n_observed=455, n_target=4 * 455)
        assert longer == pytest.approx(base / 2, rel=1e-9)

    def test_the_test_reports_its_own_mde(self):
        a, b = _pair(seed=11)
        res = sharpe_difference_test(a, b, n_boot=200, alpha=0.10, seed=0)
        assert res["mde_at_power_80"] == pytest.approx(
            sharpe_difference_mde(res["standard_error"], alpha=0.10, power=0.80)
        )

    def test_half_a_rescaling_pair_is_refused(self):
        with pytest.raises(ValueError, match="both n_observed and n_target"):
            sharpe_difference_mde(0.2, n_observed=455)

    def test_an_impossible_power_is_refused(self):
        with pytest.raises(ValueError, match="power must be in"):
            sharpe_difference_mde(0.2, power=1.0)
