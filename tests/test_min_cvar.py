"""
The mean-CVaR rung: does the LP optimise what it claims to optimise?

`MinCVaR` is the first optimizer in `strategies.py` that targets downside
rather than variance or the Sharpe ratio, so the tests that matter are the ones
that would catch it silently optimising something else -- a sign error in the
tail constraint, or the objective quietly collapsing to minimum-variance.

The central test is a MUTUAL positive control, in the sense
`docs/REACHABLE_CLAIMS.md` §6 argues for: in-sample, each optimizer must win on
its OWN objective. A CVaR minimiser that does not beat a variance minimiser on
CVaR is broken, whatever its out-of-sample numbers look like.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategies import EqualWeight, MaxSharpe, MinVarianceLW
from strategies_research import MinCVaR


def _panel(n=500, k=5, seed=0):
    """Assets with deliberately different tails: asset 0 is quiet, asset 1 is
    Gaussian, asset 2 carries rare large losses at the same volatility."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = {}
    cols["quiet"] = rng.normal(0.0003, 0.004, n)
    cols["normal"] = rng.normal(0.0004, 0.010, n)
    jumpy = rng.normal(0.0006, 0.007, n)
    jumpy[rng.choice(n, size=max(1, n // 40), replace=False)] -= 0.05
    cols["jumpy"] = jumpy
    for j in range(k - 3):
        cols[f"filler{j}"] = rng.normal(0.0004, 0.009, n)
    return pd.DataFrame(cols, index=idx)


def _cvar(x: np.ndarray, beta: float = 0.95) -> float:
    losses = -np.asarray(x, dtype=float)
    var = np.quantile(losses, beta)
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(var)


def _portfolio(train: pd.DataFrame, w: pd.Series) -> np.ndarray:
    return np.expm1(train.to_numpy(dtype=float)) @ w.to_numpy(dtype=float)


class TestTheContractEveryStrategyMustMeet:
    def test_weights_are_long_only_capped_and_sum_to_one(self):
        train = _panel()
        w = MinCVaR(max_weight=0.30).fit(train)
        assert list(w.index) == list(train.columns)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= -1e-12).all()
        assert w.max() <= 0.30 + 1e-9

    def test_infeasible_cap_is_rejected_rather_than_silently_renormalised(self):
        train = _panel(k=5)
        with pytest.raises(ValueError, match="Infeasible"):
            MinCVaR(max_weight=0.10).fit(train)   # 5 x 0.10 < 1

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
    def test_confidence_outside_the_unit_interval_is_rejected(self, bad):
        with pytest.raises(ValueError, match="confidence"):
            MinCVaR(confidence=bad)

    def test_negative_return_weight_is_rejected(self):
        with pytest.raises(ValueError, match="return_weight"):
            MinCVaR(return_weight=-1.0)


class TestTheMutualPositiveControl:
    """Each optimizer must win on its own objective, in sample. These are true
    by construction; if one fails, the objective is not what it says it is."""

    def test_cvar_minimiser_beats_every_other_rung_on_cvar(self):
        train = _panel()
        others = [MinVarianceLW(max_weight=0.30), MaxSharpe(max_weight=0.30), EqualWeight()]
        mine = _cvar(_portfolio(train, MinCVaR(max_weight=0.30).fit(train)))
        for strat in others:
            theirs = _cvar(_portfolio(train, strat.fit(train)))
            assert mine <= theirs + 1e-9, f"{strat.name} attains lower CVaR than min_cvar"

    def test_variance_minimiser_still_beats_the_cvar_rung_on_variance(self):
        """The converse, which stops the LP from having quietly become a
        minimum-variance solver with extra steps."""
        train = _panel()
        v_cvar = _portfolio(train, MinCVaR(max_weight=0.30).fit(train)).std(ddof=0)
        v_mv = _portfolio(train, MinVarianceLW(max_weight=0.30).fit(train)).std(ddof=0)
        assert v_mv <= v_cvar + 1e-12

    def test_the_two_rungs_do_not_return_the_same_portfolio(self):
        train = _panel()
        w_c = MinCVaR(max_weight=0.30).fit(train)
        w_v = MinVarianceLW(max_weight=0.30).fit(train)
        assert np.abs(w_c.to_numpy() - w_v.to_numpy()).max() > 1e-3

    def test_it_avoids_the_asset_whose_only_defect_is_its_tail(self):
        """`jumpy` has rare large losses at a volatility the variance rungs are
        relatively relaxed about. The tail-aware objective should hold less of
        it than the variance-aware one."""
        train = _panel()
        w_c = MinCVaR(max_weight=0.30).fit(train)
        w_v = MinVarianceLW(max_weight=0.30).fit(train)
        assert w_c["jumpy"] < w_v["jumpy"] + 1e-9


class TestTheParameters:
    def test_a_higher_confidence_level_looks_further_into_the_tail(self):
        train = _panel()
        w90 = MinCVaR(max_weight=0.30, confidence=0.90).fit(train)
        w99 = MinCVaR(max_weight=0.30, confidence=0.99).fit(train)
        assert np.abs(w90.to_numpy() - w99.to_numpy()).max() > 1e-4

    def test_return_weight_tilts_toward_the_higher_mean_asset(self):
        train = _panel()
        pure = MinCVaR(max_weight=0.30, return_weight=0.0).fit(train)
        tilted = MinCVaR(max_weight=0.30, return_weight=5.0).fit(train)
        best = train.mean().idxmax()
        assert tilted[best] >= pure[best] - 1e-9
        assert np.abs(pure.to_numpy() - tilted.to_numpy()).max() > 1e-4


class TestDeterminism:
    def test_the_same_window_returns_the_same_weights(self):
        """An LP has a global optimum, so unlike the SLSQP rungs this needs no
        seed argument -- but the repo's rule is that reproducibility is
        demonstrated on data, not inferred."""
        train = _panel()
        a = MinCVaR(max_weight=0.30).fit(train)
        b = MinCVaR(max_weight=0.30).fit(train)
        pd.testing.assert_series_equal(a, b)
