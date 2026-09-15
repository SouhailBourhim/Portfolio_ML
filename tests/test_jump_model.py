"""
Statistical jump model: is the dynamic programme actually optimal, and is the
online mode actually causal?

Two properties carry the whole module and neither is visible by inspecting
output. The DP claims a GLOBAL optimum given the centroids, which is checkable
against brute force on small problems. The online mode claims CAUSALITY, which
the offline fit does not have -- item 2.1's caveat is that a full-window jump
model carries the same lookahead as a smoothed HMM, so a module that got this
wrong would silently reproduce the defect it was built to avoid.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jump_model import (
    _assign_states_dp, fit_jump_model, label_states_by_volatility, online_jump_states,
)
from regime import REGIME_FEATURES


def _panel(n=400, separation=1.2, p_stay=0.97, seed=0):
    """Emissions from a 2-state process with overlapping states — the low
    signal-to-noise condition the jump-model literature is addressed to."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < p_stay else 1 - states[t - 1]
    means = np.array([[0.0, 0.0, 0.0],
                      [separation, separation * 0.8, separation * 0.6]])
    frame = pd.DataFrame(rng.normal(means[states], 1.0), index=idx, columns=REGIME_FEATURES)
    return frame, states


class TestTheDynamicProgrammeIsOptimal:
    """The DP is Viterbi with a constant transition cost. On problems small
    enough to enumerate, it must match the true minimum exactly."""

    def test_it_matches_brute_force_on_small_problems(self):
        rng = np.random.default_rng(0)
        for _ in range(60):
            n_obs, n_states = int(rng.integers(3, 8)), int(rng.integers(2, 4))
            dist = rng.random((n_obs, n_states)) * 3
            lam = float(rng.random() * 4)

            got = _assign_states_dp(dist, lam)
            got_cost = (dist[np.arange(n_obs), got].sum()
                        + lam * int((np.diff(got) != 0).sum()))

            def cost(seq):
                seq = list(seq)
                return (dist[np.arange(n_obs), seq].sum()
                        + lam * sum(a != b for a, b in zip(seq, seq[1:])))

            best = min(itertools.product(range(n_states), repeat=n_obs), key=cost)
            assert np.isclose(got_cost, cost(best)), "DP did not find the global optimum"

    def test_zero_penalty_reduces_to_nearest_centroid(self):
        rng = np.random.default_rng(1)
        dist = rng.random((200, 3))
        np.testing.assert_array_equal(_assign_states_dp(dist, 0.0), dist.argmin(axis=1))

    def test_switch_count_never_rises_with_the_penalty(self):
        rng = np.random.default_rng(2)
        dist = rng.random((300, 2)) * 2
        counts = [int((np.diff(_assign_states_dp(dist, lam)) != 0).sum())
                  for lam in (0.0, 0.5, 1.0, 2.0, 5.0, 25.0)]
        assert counts == sorted(counts, reverse=True), f"non-monotone in lambda: {counts}"

    def test_an_overwhelming_penalty_forbids_switching_entirely(self):
        rng = np.random.default_rng(3)
        dist = rng.random((150, 2))
        assert int((np.diff(_assign_states_dp(dist, 1e6)) != 0).sum()) == 0


class TestCausalityOfTheOnlineMode:
    def test_a_state_does_not_change_when_later_data_is_removed(self):
        """The property the offline fit lacks. Each online state is read from a
        prefix ending at its own date, so removing everything after that date
        must leave it untouched."""
        frame, _ = _panel()
        dates = pd.DatetimeIndex([frame.index[k] for k in (300, 340, 380)])
        full = online_jump_states(frame, dates, jump_penalty=2.5,
                                  min_train_rows=252, n_init=2)
        truncated = online_jump_states(frame.iloc[:341], dates[:2], jump_penalty=2.5,
                                       min_train_rows=252, n_init=2)
        shared = full.index.intersection(truncated.index)
        assert len(shared) >= 1
        pd.testing.assert_series_equal(full[shared], truncated[shared])

    def test_the_offline_fit_does_not_have_that_property(self):
        """Pins the reason `online_jump_states` exists: the backtrack pass
        propagates information from the end of the sample backwards, so an
        offline state sequence is not causal."""
        frame, _ = _panel()
        long_fit = fit_jump_model(frame, jump_penalty=2.5, n_init=2)
        short_fit = fit_jump_model(frame.iloc[:300], jump_penalty=2.5, n_init=2)
        # Compare on shared rows, up to a relabelling of the two states.
        a, b = long_fit.states[:300], short_fit.states
        same = float((a == b).mean())
        flipped = float((a == (1 - b)).mean())
        assert max(same, flipped) < 1.0, (
            "offline states were identical on the shared prefix; if this ever holds, "
            "the lookahead caveat in item 2.1 needs revisiting on this data"
        )


class TestTheFit:
    def test_it_recovers_a_well_separated_two_state_process(self):
        frame, truth = _panel(separation=3.0, seed=5)
        fit = fit_jump_model(frame, jump_penalty=2.5, n_init=3)
        acc = max(float((fit.states == truth).mean()), float((fit.states == 1 - truth).mean()))
        assert acc > 0.85, f"recovered only {acc:.2f} of a well-separated process"

    def test_the_objective_is_the_reported_one(self):
        frame, _ = _panel()
        fit = fit_jump_model(frame, jump_penalty=2.5, n_init=2)
        scaled = fit.scaler.transform(frame.dropna().to_numpy())
        dist = ((scaled[:, None, :] - fit.centroids[None, :, :]) ** 2).sum(axis=2)
        recomputed = (dist[np.arange(len(fit.states)), fit.states].sum()
                      + fit.jump_penalty * fit.n_switches)
        assert np.isclose(recomputed, fit.objective, rtol=1e-6)

    def test_restarts_never_return_a_worse_objective_than_one_init(self):
        frame, _ = _panel(seed=7)
        one = fit_jump_model(frame, jump_penalty=2.5, n_init=1)
        many = fit_jump_model(frame, jump_penalty=2.5, n_init=5)
        assert many.objective <= one.objective + 1e-9

    def test_an_empty_frame_returns_an_unconverged_fit_rather_than_raising(self):
        empty = pd.DataFrame(columns=REGIME_FEATURES, dtype=float)
        fit = fit_jump_model(empty)
        assert not fit.converged and fit.states.size == 0

    def test_states_are_labelled_by_volatility_not_by_index(self):
        frame, _ = _panel(separation=3.0, seed=5)
        fit = fit_jump_model(frame, jump_penalty=2.5, n_init=3)
        labels = label_states_by_volatility(fit, frame)
        assert set(labels.values()) == {"bull", "bear"}
        vols = frame["MARKET_VOL_SHORT"].to_numpy()
        bear = [k for k, v in labels.items() if v == "bear"][0]
        bull = [k for k, v in labels.items() if v == "bull"][0]
        assert vols[fit.states == bear].mean() > vols[fit.states == bull].mean()


class TestDeterminism:
    def test_the_same_inputs_give_the_same_fit(self):
        frame, _ = _panel()
        a = fit_jump_model(frame, jump_penalty=2.5, n_init=3, random_state=0)
        b = fit_jump_model(frame, jump_penalty=2.5, n_init=3, random_state=0)
        np.testing.assert_array_equal(a.states, b.states)
        assert np.isclose(a.objective, b.objective)
