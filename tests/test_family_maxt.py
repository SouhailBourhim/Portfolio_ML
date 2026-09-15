"""
Correcting across a FAMILY of estimands, not just across configurations.

Choosing which estimand to report is a search. `reality_check` corrects across
configurations; this corrects across estimands and strategy pairs. The tests
pin the two properties that make it worth having: the critical value must
exceed the uncorrected 1.96, and it must respond to how correlated the family
actually is, because that is what a Bonferroni bound gets wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import family_maxt_correction

MEAN = lambda x: float(x.mean())
STD = lambda x: float(x.std(ddof=0))


def _series(n=500, seed=0, loc=0.0, scale=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(loc, scale, n), index=idx)


class TestTheCorrectionActuallyCorrects:
    def test_critical_value_exceeds_the_uncorrected_threshold(self):
        base = _series(seed=1)
        fam = {f"h{i}": (base, _series(seed=100 + i), MEAN) for i in range(8)}
        out = family_maxt_correction(fam, n_boot=800, alpha=0.05)
        assert out["critical_value"] > 1.96
        assert out["n_hypotheses"] == 8

    def test_a_larger_independent_family_needs_a_higher_bar(self):
        base = _series(seed=1)
        small = {f"h{i}": (base, _series(seed=200 + i), MEAN) for i in range(3)}
        large = {f"h{i}": (base, _series(seed=200 + i), MEAN) for i in range(20)}
        c_small = family_maxt_correction(small, n_boot=800, alpha=0.05)["critical_value"]
        c_large = family_maxt_correction(large, n_boot=800, alpha=0.05)["critical_value"]
        assert c_large > c_small

    def test_a_perfectly_redundant_family_costs_almost_nothing(self):
        """Shared draws are the point. Ten copies of one hypothesis are one
        hypothesis; Bonferroni would charge for ten."""
        a, b = _series(seed=1), _series(seed=2)
        one = family_maxt_correction({"h": (a, b, MEAN)}, n_boot=1200, alpha=0.05)
        ten = family_maxt_correction({f"h{i}": (a, b, MEAN) for i in range(10)},
                                     n_boot=1200, alpha=0.05)
        assert ten["critical_value"] == pytest.approx(one["critical_value"], abs=1e-9)


class TestItSeparatesRealEffectsFromNoise:
    def test_a_large_true_effect_survives_a_family_of_nulls(self):
        base = _series(seed=1, scale=0.01)
        fam = {"real": (base, _series(seed=2, scale=0.03), STD)}
        fam.update({f"null{i}": (base, _series(seed=300 + i), MEAN) for i in range(9)})
        out = family_maxt_correction(fam, n_boot=1500, alpha=0.05)
        assert out["results"]["real"]["survives"]
        assert not any(out["results"][f"null{i}"]["survives"] for i in range(9))


class TestContract:
    def test_it_is_deterministic_under_a_fixed_seed(self):
        a, b = _series(seed=1), _series(seed=2)
        fam = {"h": (a, b, MEAN)}
        kw = dict(n_boot=400, alpha=0.05, seed=11)
        assert (family_maxt_correction(fam, **kw)["critical_value"]
                == family_maxt_correction(fam, **kw)["critical_value"])

    def test_an_empty_family_is_refused(self):
        with pytest.raises(ValueError, match="must not be empty"):
            family_maxt_correction({}, n_boot=100)

    def test_ragged_hypotheses_are_refused_because_draws_are_shared(self):
        a, b = _series(n=500, seed=1), _series(n=500, seed=2)
        fam = {"long": (a, b, MEAN), "short": (a.iloc[:100], b.iloc[:100], MEAN)}
        with pytest.raises(ValueError, match="same number of aligned"):
            family_maxt_correction(fam, n_boot=100)

    def test_block_length_at_or_above_n_is_refused(self):
        a, b = _series(n=50, seed=1), _series(n=50, seed=2)
        with pytest.raises(ValueError, match="block_len must be in"):
            family_maxt_correction({"h": (a, b, MEAN)}, block_len=50, n_boot=100)
