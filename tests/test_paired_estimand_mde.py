"""
Power analysis for estimands other than the Sharpe difference.

The motivating measurement: on the frozen-length `full_2021` window the Sharpe
difference between `regime_conditional` and `max_sharpe` sits at 0.01 of its
detection threshold, while the log volatility ratio between the SAME two series
sits above 2. Variance is estimated far more precisely than mean return, so a
risk claim is reachable on a sample where a return claim is not. These tests
pin that behaviour, and pin the silent-failure mode that made the first attempt
at the turnover version wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import paired_estimand_mde


MEAN = lambda x: float(x.mean())
LOG_VOL = lambda x: float(np.log(x.std(ddof=0)))


def _pair(n=600, seed=0, scale_b=1.0, shift_b=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    a = rng.normal(0.0004, 0.01, n)
    b = rng.normal(0.0004 + shift_b, 0.01 * scale_b, n)
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


class TestTheGuardAgainstACollapsedStandardError:
    """The bug this function exists to make impossible.

    A circular block bootstrap with block_len >= n can only produce rotations
    of the input. Any mean is rotation-invariant, so every draw is identical,
    the SE is exactly zero, and `ratio = |observed| / 0` reports a detectable
    effect with total confidence. A monthly rebalance series (21 points) with
    the default 21-day block hits this exactly.
    """

    def test_block_length_at_or_above_n_is_refused(self):
        a, b = _pair(n=21)
        with pytest.raises(ValueError, match="block_len must be in"):
            paired_estimand_mde(a, b, MEAN, block_len=21, n_boot=50)

    def test_the_refusal_explains_the_silent_failure_it_prevents(self):
        a, b = _pair(n=21)
        with pytest.raises(ValueError) as exc:
            paired_estimand_mde(a, b, MEAN, block_len=30, n_boot=50)
        assert "collapses to zero" in str(exc.value)

    def test_a_legal_block_on_the_same_short_series_gives_a_positive_se(self):
        """Non-vacuity: the guard rejects a bad call, it does not reject
        short series as such."""
        a, b = _pair(n=21)
        out = paired_estimand_mde(a, b, MEAN, block_len=3, n_boot=200)
        assert out["standard_error"] > 0
        assert out["blocks"] == 7


class TestItRecoversTheAsymmetryThatMotivatesIt:
    def test_a_pure_volatility_difference_is_detected_where_a_mean_is_not(self):
        # b has 40% higher volatility and an identical drift.
        a, b = _pair(n=600, scale_b=1.4)
        vol = paired_estimand_mde(a, b, LOG_VOL, block_len=21, n_boot=400)
        mean = paired_estimand_mde(a, b, MEAN, block_len=21, n_boot=400)
        assert vol["detectable"], vol
        assert not mean["detectable"], mean
        assert vol["ratio"] > mean["ratio"]

    def test_identical_series_have_no_effect_to_detect(self):
        a, _ = _pair(n=400)
        out = paired_estimand_mde(a, a.copy(), LOG_VOL, block_len=21, n_boot=200)
        assert out["observed"] == pytest.approx(0.0, abs=1e-12)
        assert not out["detectable"]


class TestContract:
    def test_it_is_deterministic_under_a_fixed_seed(self):
        a, b = _pair()
        kw = dict(block_len=21, n_boot=200, seed=7)
        assert (paired_estimand_mde(a, b, MEAN, **kw)
                == paired_estimand_mde(a, b, MEAN, **kw))

    def test_misaligned_indices_are_intersected_not_padded(self):
        a, b = _pair(n=300)
        out = paired_estimand_mde(a, b.iloc[50:], MEAN, block_len=21, n_boot=100)
        assert out["n_observations"] == 250

    def test_too_few_aligned_observations_is_refused(self):
        a, b = _pair(n=300)
        with pytest.raises(ValueError, match="at least 2 aligned"):
            paired_estimand_mde(a.iloc[:1], b.iloc[:1], MEAN, block_len=1)

    def test_mde_grows_with_demanded_power(self):
        a, b = _pair()
        lo = paired_estimand_mde(a, b, MEAN, block_len=21, n_boot=200, power=0.50)
        hi = paired_estimand_mde(a, b, MEAN, block_len=21, n_boot=200, power=0.95)
        assert hi["mde"] > lo["mde"]
