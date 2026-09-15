"""
Filtered regime posteriors: is the forward recursion actually causal?

`regime.filtered_posterior_series` exists to remove the in-window
lookahead `docs/EVALUATION_LIMITS.md` Limit #3 documents. The tests that matter
are therefore the two that would catch it failing to do so, and they are
independent of each other:

  CAUSALITY -- row t must not change when observations after t are removed.
  This is the definition of a filtered posterior and the property the smoothed
  one lacks. A test comparing against hmmlearn cannot establish it, since
  hmmlearn has no filtered output to compare against.

  CORRECTNESS -- at t = T, filtered and smoothed coincide by construction,
  because the backward pass has no future to integrate. This anchors the
  recursion, the log-space scaling and the label mapping against an
  independent implementation, at the one index where they must agree.

Together they pin the function from both sides: causality says it ignores the
future, the final-row identity says it has not simply mangled the arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from regime import REGIME_FEATURES, fit_hmm, predict_regime_posterior_series
from regime import filtered_matches_smoothed_at_final_row, filtered_posterior_series


@pytest.fixture(scope="module")
def panel():
    """Emissions drawn from an actual 2-state HMM with OVERLAPPING states.

    The first version of this fixture built a vol-regime series by hand, with a
    calm stretch and a turbulent one. It was useless: the states were so
    separable that 100% of posterior rows saturated at 0 or 1, the backward pass
    had nothing left to contribute, and the smoothed posterior came out
    effectively causal -- so the test asserting otherwise failed on a panel that
    simply had no ambiguity in it.

    The regime literature is explicit that the instability appears under LOW
    signal-to-noise, so the fixture has to supply that. Separation 1.2 against
    unit emission noise leaves ~71% of rows saturated and moves the smoothed
    posterior by ~0.24 when the future is truncated, which is the same regime of
    behaviour the three real panels show.
    """
    rng = np.random.default_rng(0)
    n, p_stay, separation = 400, 0.97, 1.2
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < p_stay else 1 - states[t - 1]
    means = np.array([[0.0, 0.0, 0.0],
                      [separation, separation * 0.8, separation * 0.6]])
    return pd.DataFrame(rng.normal(means[states], 1.0), index=idx,
                        columns=REGIME_FEATURES)


@pytest.fixture(scope="module")
def fit(panel):
    f = fit_hmm(panel, n_states=2, n_restarts=5, random_state_base=0)
    if not f.converged:
        pytest.skip("HMM did not converge on the synthetic panel")
    return f


class TestCausality:
    """The property the whole module exists for."""

    @pytest.mark.parametrize("cut", [0.5, 0.7, 0.9])
    def test_truncating_the_future_does_not_move_an_earlier_row(self, fit, panel, cut):
        full = filtered_posterior_series(fit, panel)
        k = int(len(panel) * cut)
        prefix = filtered_posterior_series(fit, panel.iloc[:k])
        shared = prefix.index
        pd.testing.assert_frame_equal(
            full.loc[shared], prefix, check_exact=False, atol=1e-12,
            obj=f"filtered posterior changed when {len(panel) - k} future rows were removed",
        )

    def test_the_smoothed_posterior_fails_that_same_check(self, fit, panel):
        """Not a test of our code — it pins the defect being fixed, so this file
        documents why the filtered version is needed rather than asserting it."""
        full = predict_regime_posterior_series(fit, panel)
        k = int(len(panel) * 0.7)
        prefix = predict_regime_posterior_series(fit, panel.iloc[:k])
        moved = (full.loc[prefix.index, "bull"] - prefix["bull"]).abs().max()
        assert moved > 1e-6, (
            "the smoothed posterior did not move when the future was truncated; "
            "if this ever holds, Limit #3 has changed character and the note needs revisiting"
        )


class TestCorrectnessAgainstHmmlearn:
    def test_filtered_equals_smoothed_at_the_final_row(self, fit, panel):
        ok, dev = filtered_matches_smoothed_at_final_row(fit, panel)
        assert ok, f"filtered != smoothed at t=T, deviation {dev:.3e}"

    def test_that_identity_holds_at_every_prefix_endpoint(self, fit, panel):
        """Stronger: T is not special, so the identity must hold for any window
        end. This catches an off-by-one that happened to work at one index."""
        for k in (60, 120, 200, 300):
            if k > len(panel):
                continue
            filt_k = filtered_posterior_series(fit, panel.iloc[:k]).iloc[-1]
            smooth_k = predict_regime_posterior_series(fit, panel.iloc[:k]).iloc[-1]
            assert np.abs(filt_k["bull"] - smooth_k["bull"]) < 1e-8, f"mismatch at prefix {k}"


class TestTheContract:
    def test_rows_are_probabilities_summing_to_one(self, fit, panel):
        out = filtered_posterior_series(fit, panel)
        assert (out.to_numpy() >= -1e-12).all() and (out.to_numpy() <= 1 + 1e-12).all()
        np.testing.assert_allclose(out.sum(axis=1).to_numpy(), 1.0, atol=1e-10)

    def test_index_and_columns_match_the_smoothed_function(self, fit, panel):
        a = filtered_posterior_series(fit, panel)
        b = predict_regime_posterior_series(fit, panel)
        assert list(a.index) == list(b.index)
        assert set(a.columns) == set(b.columns)

    def test_a_non_converged_fit_returns_the_same_empty_frame(self, fit, panel):
        import dataclasses
        broken = dataclasses.replace(fit, converged=False)
        out = filtered_posterior_series(broken, panel)
        assert out.empty and list(out.columns) == ["bull", "bear"]

    def test_an_all_nan_window_returns_the_same_empty_frame(self, fit, panel):
        blank = panel.copy()
        blank[REGIME_FEATURES] = np.nan
        out = filtered_posterior_series(fit, blank)
        assert out.empty and list(out.columns) == ["bull", "bear"]


class TestTheMeasuredGap:
    def test_the_gap_is_concentrated_around_regime_switches(self, fit, panel):
        """The experiment reports ~18x more drift near a switch than away from
        one, on three real panels. The mechanism should show up here too: the
        backward pass is informative exactly where the state is about to change."""
        filt = filtered_posterior_series(fit, panel)["bull"]
        smooth = predict_regime_posterior_series(fit, panel)["bull"]
        gap = (filt - smooth).abs()
        state = (smooth > 0.5).astype(int)
        near = (state.diff().abs().fillna(0) > 0).rolling(11, center=True,
                                                          min_periods=1).max().astype(bool)
        if near.sum() == 0 or (~near).sum() == 0:
            pytest.skip("synthetic panel produced no usable switch")
        assert gap[near].mean() > gap[~near].mean()
