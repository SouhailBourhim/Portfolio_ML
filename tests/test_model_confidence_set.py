"""test_model_confidence_set.py — the set must eliminate when it should.

Built on the same principle as test_reality_check.py: a procedure that never
eliminates would agree with every conclusion this project has already reached
and would never be questioned. So the suite leads with a POSITIVE CONTROL — a
set containing strategies that are genuinely worse MUST shed them — and only
then asks whether it behaves under the null.

The complementary failure matters just as much here and does not arise for
RC/SPA: an MCS that eliminates under exchangeability would manufacture a
ranking out of noise, which is exactly the over-reading the procedure exists
to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics import model_confidence_set


DATES = pd.bdate_range("2022-01-03", periods=750)


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=DATES)


def _exchangeable(n_strategies: int, seed: int = 0) -> dict[str, pd.Series]:
    """Strategies drawn from one distribution: none is genuinely better."""
    rng = np.random.default_rng(seed)
    return {
        f"strat_{k:02d}": _series(rng.normal(0.0004, 0.010, len(DATES)))
        for k in range(n_strategies)
    }


class TestPositiveControl:
    """If this fails, every retention result below is meaningless."""

    def test_clearly_worse_strategies_are_eliminated(self):
        candidates = _exchangeable(6, seed=1)
        # Three strategies losing 30 bps/day for three years. Nothing subtle.
        for k in range(3):
            candidates[f"awful_{k}"] = _series(
                np.random.default_rng(100 + k).normal(-0.003, 0.010, len(DATES))
            )
        result = model_confidence_set(candidates, n_boot=500, seed=0)
        for k in range(3):
            assert f"awful_{k}" in result["excluded"], (
                f"awful_{k} lost 30bps/day for 750 days and survived elimination — "
                f"the MCS is too conservative to say anything"
            )
        assert result["n_included"] < result["n_candidates"]

    def test_the_best_strategy_is_never_eliminated(self):
        candidates = _exchangeable(6, seed=2)
        candidates["winner"] = _series(
            np.random.default_rng(7).normal(0.004, 0.010, len(DATES))
        )
        result = model_confidence_set(candidates, n_boot=500, seed=0)
        assert "winner" in result["included"]


class TestNullBehaviour:
    def test_exchangeable_strategies_are_not_eliminated(self):
        """Eliminating here would manufacture a ranking out of noise."""
        result = model_confidence_set(_exchangeable(8, seed=3), n_boot=500, seed=0)
        assert result["n_included"] >= 7, (
            f"only {result['n_included']} of 8 exchangeable strategies survived; "
            f"the procedure is eliminating on noise"
        )


class TestSizeBehaviour:
    def test_a_larger_size_cannot_grow_the_set(self):
        """The set has confidence 1 - size, so raising size may only shrink it."""
        candidates = _exchangeable(5, seed=4)
        for k in range(3):
            candidates[f"awful_{k}"] = _series(
                np.random.default_rng(200 + k).normal(-0.002, 0.010, len(DATES))
            )
        sizes = [0.05, 0.10, 0.25]
        counts = [
            model_confidence_set(candidates, size=s, n_boot=500, seed=0)["n_included"]
            for s in sizes
        ]
        assert counts == sorted(counts, reverse=True), (
            f"|MCS| must not grow with size: {list(zip(sizes, counts))}"
        )


class TestReportingDiscipline:
    def test_the_interpretation_avoids_the_banned_equivalence_wording(self):
        """AGENTS.md section 5.2: nothing here is 'statistically indistinguishable'."""
        result = model_confidence_set(_exchangeable(4, seed=5), n_boot=300, seed=0)
        text = result["interpretation"].lower()
        for banned in ("indistinguishable", "statistically significant", "equivalent"):
            assert banned not in text, f"interpretation uses banned wording: {banned!r}"
        assert "retained by the mcs procedure" in text

    def test_the_statistic_is_declared_as_mean_return(self):
        """It is not a Sharpe criterion, and the artifact must not imply it is."""
        result = model_confidence_set(_exchangeable(3, seed=6), n_boot=300, seed=0)
        assert result["statistic"] == "mean_return"

    def test_included_and_excluded_partition_the_candidates(self):
        candidates = _exchangeable(5, seed=7)
        result = model_confidence_set(candidates, n_boot=300, seed=0)
        assert set(result["included"]) | set(result["excluded"]) == set(candidates)
        assert not set(result["included"]) & set(result["excluded"])


class TestDeterminism:
    def test_the_same_seed_gives_the_same_set(self):
        candidates = _exchangeable(6, seed=8)
        a = model_confidence_set(candidates, n_boot=400, seed=0)
        b = model_confidence_set(candidates, n_boot=400, seed=0)
        assert a == b


class TestInputDiscipline:
    def test_a_single_candidate_is_refused(self):
        """A confidence set over one model is vacuous, not a degenerate answer."""
        with pytest.raises(ValueError, match="at least two candidates"):
            model_confidence_set({"only": _series(np.zeros(len(DATES)))})

    def test_a_misaligned_candidate_is_refused_not_aligned(self):
        candidates = _exchangeable(2, seed=9)
        shifted = candidates["strat_01"].copy()
        shifted.index = pd.bdate_range("2022-02-01", periods=len(DATES))
        candidates["strat_01"] = shifted
        with pytest.raises(ValueError, match="identical date indexes"):
            model_confidence_set(candidates)

    def test_nan_is_refused(self):
        candidates = _exchangeable(2, seed=10)
        candidates["strat_00"].iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN-free"):
            model_confidence_set(candidates)

    def test_too_little_data_for_one_block_is_refused(self):
        short = pd.bdate_range("2022-01-03", periods=5)
        candidates = {
            "a": pd.Series(np.zeros(5), index=short),
            "b": pd.Series(np.ones(5) * 0.001, index=short),
        }
        with pytest.raises(ValueError, match="Not enough observations"):
            model_confidence_set(candidates, block_len=21)
