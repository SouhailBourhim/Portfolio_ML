"""test_pbo.py — the overfitting probability must separate skill from selection.

The paper's own headline property is the trap: PBO tends to 1 as the search
widens EVEN WHEN nothing has skill. So a test suite that only checked "PBO is
high on noise" would pass on an implementation that always returned 1.

The discriminating checks are therefore two-sided. Pure noise must give PBO
near the 0.5 a coin would produce at a fixed trial count, and rise with the
breadth of the search; a genuinely persistent winner must give PBO near 0.
An implementation that cannot tell those apart is useless here, because this
project's entire question is whether a 240-configuration maximum is real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inference import probability_of_backtest_overfitting


DATES = pd.bdate_range("2020-01-01", periods=1024)


def _noise_trials(n_trials: int, seed: int = 0) -> dict[str, pd.Series]:
    """No configuration has any edge; any winner is selection luck."""
    rng = np.random.default_rng(seed)
    return {
        f"cfg_{k:03d}": pd.Series(rng.normal(0.0004, 0.01, len(DATES)), index=DATES)
        for k in range(n_trials)
    }


def _trials_with_a_persistent_winner(n_trials: int, seed: int = 0) -> dict[str, pd.Series]:
    """One configuration is genuinely better in every sub-period."""
    trials = _noise_trials(n_trials - 1, seed=seed)
    rng = np.random.default_rng(seed + 500)
    trials["winner"] = pd.Series(rng.normal(0.006, 0.01, len(DATES)), index=DATES)
    return trials


class TestPositiveControl:
    """A real, persistent edge must NOT look like overfitting."""

    def test_a_persistent_winner_gives_a_low_pbo(self):
        res = probability_of_backtest_overfitting(
            _trials_with_a_persistent_winner(20, seed=1), n_splits=10
        )
        assert res["pbo"] < 0.10, (
            f"a configuration better in every sub-period was judged overfit "
            f"(PBO = {res['pbo']:.3f}) — the procedure cannot detect real skill"
        )

    def test_a_persistent_winner_is_not_flagged_as_loss_making(self):
        res = probability_of_backtest_overfitting(
            _trials_with_a_persistent_winner(20, seed=2), n_splits=10
        )
        assert res["probability_of_loss"] < 0.10


class TestNullBehaviour:
    def test_independent_noise_gives_a_coin_flip(self):
        """With independent candidates the winner's OOS rank is uniform.

        Averaged over seeds, because the per-seed spread is wide: single draws
        range from roughly 0.30 to 0.90 at these sizes, so a one-seed assertion
        would be testing the seed rather than the estimator.
        """
        values = [
            probability_of_backtest_overfitting(_noise_trials(20, seed=s), n_splits=10)["pbo"]
            for s in range(5)
        ]
        mean_pbo = sum(values) / len(values)
        assert 0.35 <= mean_pbo <= 0.65, (
            f"mean PBO {mean_pbo:.3f} over independent noise; the in-sample "
            f"winner's out-of-sample rank should be close to uniform"
        )

    def test_breadth_alone_does_not_inflate_pbo(self):
        """The property it is tempting to assume, and that does NOT hold.

        "PBO tends to 1 as the search widens" is about dependence between the
        candidates and the selection, not about the count. With INDEPENDENT
        series, widening the search from 4 to 60 leaves PBO near a coin flip,
        and an implementation that drifted upward with N would be reporting
        breadth rather than overfitting.
        """
        def mean_pbo(n_trials: int) -> float:
            vals = [
                probability_of_backtest_overfitting(
                    _noise_trials(n_trials, seed=100 + s), n_splits=10
                )["pbo"]
                for s in range(5)
            ]
            return sum(vals) / len(vals)

        narrow, wide = mean_pbo(4), mean_pbo(60)
        assert abs(wide - narrow) < 0.25, (
            f"PBO moved from {narrow:.3f} to {wide:.3f} purely by widening an "
            f"independent search; it is tracking trial count, not overfitting"
        )


class TestReportedDiagnostics:
    def test_the_partition_count_is_the_binomial_coefficient(self):
        from math import comb

        res = probability_of_backtest_overfitting(_noise_trials(5, seed=5), n_splits=10)
        assert res["n_partitions"] == comb(10, 5)

    def test_leftover_observations_are_dropped_and_reported(self):
        trials = {
            k: v.iloc[:1003] for k, v in _noise_trials(4, seed=6).items()
        }  # 1003 = 8*125 + 3
        res = probability_of_backtest_overfitting(trials, n_splits=8)
        assert res["n_observations_used"] == 1000
        assert res["n_observations_dropped"] == 3

    def test_the_interpretation_states_the_selection_caveat(self):
        res = probability_of_backtest_overfitting(_noise_trials(6, seed=7), n_splits=8)
        assert "regardless of genuine skill" in res["interpretation"]

    def test_pbo_is_a_probability(self):
        res = probability_of_backtest_overfitting(_noise_trials(8, seed=8), n_splits=8)
        assert 0.0 <= res["pbo"] <= 1.0


class TestDeterminism:
    def test_cscv_is_deterministic(self):
        """CSCV enumerates every partition; there is nothing to seed."""
        trials = _noise_trials(8, seed=9)
        assert probability_of_backtest_overfitting(
            trials, n_splits=8
        ) == probability_of_backtest_overfitting(trials, n_splits=8)


class TestInputDiscipline:
    def test_a_single_trial_is_refused(self):
        with pytest.raises(ValueError, match="at least two trials"):
            probability_of_backtest_overfitting(
                {"only": pd.Series(np.zeros(len(DATES)), index=DATES)}
            )

    def test_an_odd_split_count_is_refused(self):
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(_noise_trials(4, seed=10), n_splits=7)

    def test_too_few_splits_is_refused(self):
        with pytest.raises(ValueError, match="even and >= 4"):
            probability_of_backtest_overfitting(_noise_trials(4, seed=11), n_splits=2)

    def test_misaligned_indexes_are_refused(self):
        trials = _noise_trials(3, seed=12)
        shifted = trials["cfg_001"].copy()
        shifted.index = pd.bdate_range("2021-01-01", periods=len(DATES))
        trials["cfg_001"] = shifted
        with pytest.raises(ValueError, match="identical date indexes"):
            probability_of_backtest_overfitting(trials, n_splits=8)

    def test_nan_is_refused(self):
        trials = _noise_trials(3, seed=13)
        trials["cfg_000"].iloc[7] = np.nan
        with pytest.raises(ValueError, match="NaN-free"):
            probability_of_backtest_overfitting(trials, n_splits=8)

    def test_too_few_observations_per_block_is_refused(self):
        short = pd.bdate_range("2022-01-03", periods=6)
        trials = {
            "a": pd.Series(np.arange(6, dtype=float), index=short),
            "b": pd.Series(np.arange(6, dtype=float)[::-1], index=short),
        }
        with pytest.raises(ValueError, match="Too few observations"):
            probability_of_backtest_overfitting(trials, n_splits=8)
