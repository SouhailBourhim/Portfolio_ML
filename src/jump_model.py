"""
jump_model.py — statistical jump models for regime identification.

Addresses: P2, P3 — the same mapping as `regime.py`, which this is an
alternative estimator for. P2: regime shifts are the non-stationarity being
detected, and the jump penalty is a direct handle on how readily the estimator
declares one. P3: the detected regime dispatches `RegimeConditionalStrategy` to
a defensive sub-strategy, so state persistence governs whether that defence is
in place when correlations spike.

WHY THIS MODULE EXISTS. `docs/LITERATURE_IMPROVEMENTS.md` item 2.1 (Nystrup,
Kolm & Lindström 2020/2021; Shu, Yu & Mulvey 2024) reports that HMMs under
*high regime persistence, low signal-to-noise and limited data* -- a description
of a 2-state HMM on a five-year panel -- produce state sequences that lack
persistence and stability. The project shows the symptom: six warm-up fallbacks
in `regime_conditional` (6.2% of its out-of-sample days) and posteriors that
move ~0.7 on a refit with states merely relabelled.

A jump model replaces the estimated transition matrix with an explicit penalty.
For features x_1..x_T, K states, centroids theta and a state sequence s, it
minimises

    SUM_t || x_t - theta_{s_t} ||^2  +  lambda * SUM_t 1[ s_t != s_{t-1} ]

Persistence stops being an emergent property of a fitted transition matrix and
becomes a tunable regularisation parameter: lambda is charged at every switch,
so the sequence changes state only when the data pays for it.

The optimisation is coordinate descent, and each half is exact:

  - centroids fixed -> the optimal state SEQUENCE by dynamic programming. This
    is Viterbi with a constant transition cost, so it is a global optimum given
    the centroids, not a greedy pass.
  - sequence fixed -> the optimal centroids are the per-state means.

Each step is non-increasing in the objective and the objective is bounded below,
so the loop converges. It converges to a LOCAL optimum -- the assignment problem
is combinatorial -- which is why `n_init` restarts from different k-means seeds
are taken and the best objective kept, exactly as `regime.fit_hmm` restarts EM.

ON CAUSALITY, which is the reason this module is shaped the way it is. Item
2.1's caveat is explicit: a jump model fitted over a full window has the SAME
in-window lookahead as a smoothed HMM, because the dynamic programme's backward
pass sees every observation. Fitting it once and reading historical states would
reproduce the defect `docs/EVALUATION_LIMITS.md` Limit #3 documents, and any
comparison against a filtered HMM would then be confounded by exactly that
defect. `online_jump_states` therefore refits over expanding prefixes and keeps
only the LAST state of each, which is the only state that saw no future.

WHY NOT IN `src/regime.py`. Twelve DVC stages declare that file as a dependency.
`grep -c jump_model dvc.yaml` must stay 0 until a stage genuinely uses this.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("jump_model")


@dataclass(frozen=True)
class JumpFit:
    """A fitted jump model. Mirrors `regime.HMMFit`'s role."""

    centroids: np.ndarray          # (n_states, n_features), in scaled space
    states: np.ndarray             # (n_obs,) integer state sequence
    scaler: StandardScaler
    objective: float
    n_switches: int
    jump_penalty: float
    converged: bool


def _assign_states_dp(distances: np.ndarray, jump_penalty: float) -> np.ndarray:
    """
    Optimal state sequence for fixed centroids, by dynamic programming.

    `distances[t, k]` is the squared distance from observation t to centroid k.
    Cost to be in state k at t is that distance plus the cheapest way of having
    arrived, where arriving from a different state costs `jump_penalty`.

    Exact given the centroids: every path is considered, none greedily pruned.
    """
    n_obs, n_states = distances.shape
    cost = np.empty((n_obs, n_states), dtype=float)
    back = np.zeros((n_obs, n_states), dtype=np.int64)
    cost[0] = distances[0]
    for t in range(1, n_obs):
        stay = cost[t - 1]
        best_other = np.empty(n_states, dtype=float)
        best_idx = np.empty(n_states, dtype=np.int64)
        for k in range(n_states):
            switched = stay + jump_penalty
            switched[k] = stay[k]          # staying in k is not a switch
            best_idx[k] = int(np.argmin(switched))
            best_other[k] = switched[best_idx[k]]
        cost[t] = distances[t] + best_other
        back[t] = best_idx
    states = np.empty(n_obs, dtype=np.int64)
    states[-1] = int(np.argmin(cost[-1]))
    for t in range(n_obs - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def fit_jump_model(
    features: pd.DataFrame,
    n_states: int = 2,
    jump_penalty: float = 50.0,
    n_init: int = 5,
    max_iter: int = 50,
    random_state: int = 0,
    scaler: StandardScaler | None = None,
) -> JumpFit:
    """
    Fit a statistical jump model by coordinate descent.

    `jump_penalty` is in units of squared distance in STANDARDISED feature
    space, so it is comparable across panels: with p features scaled to unit
    variance, a penalty of p is roughly "switch only if it improves fit by one
    standard deviation on every feature at that observation".

    `scaler` may be supplied to reuse a fit from a training window rather than
    re-standardising -- necessary for online use, where re-scaling on each
    expanding prefix would leak the prefix's own moments into earlier rows.
    """
    clean = features.dropna()
    if clean.empty or len(clean) < n_states:
        return JumpFit(np.empty((0, 0)), np.empty(0, dtype=np.int64),
                       scaler or StandardScaler(), float("inf"), 0, jump_penalty, False)

    if scaler is None:
        scaler = StandardScaler().fit(clean.to_numpy())
    X = scaler.transform(clean.to_numpy())

    best: JumpFit | None = None
    for init in range(n_init):
        km = KMeans(n_clusters=n_states, n_init=10, random_state=random_state + init)
        centroids = km.fit(X).cluster_centers_
        prev_obj, states = float("inf"), None
        converged = False
        for _ in range(max_iter):
            distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            states = _assign_states_dp(distances, jump_penalty)
            for k in range(n_states):
                mask = states == k
                if mask.any():
                    centroids[k] = X[mask].mean(axis=0)
            fit_cost = float(distances[np.arange(len(states)), states].sum())
            n_switch = int((np.diff(states) != 0).sum())
            obj = fit_cost + jump_penalty * n_switch
            if np.isclose(obj, prev_obj, rtol=1e-10, atol=1e-12):
                converged = True
                break
            prev_obj = obj
        candidate = JumpFit(centroids.copy(), states, scaler, prev_obj,
                            int((np.diff(states) != 0).sum()), jump_penalty, converged)
        if best is None or candidate.objective < best.objective:
            best = candidate
    return best


def label_states_by_volatility(fit: JumpFit, features: pd.DataFrame,
                               vol_column: str = "MARKET_VOL_SHORT") -> dict[int, str]:
    """
    Map integer states to `"bull"`/`"bear"` by mean volatility, highest = bear.

    Jump model states are arbitrary integers, as HMM states are, and
    `regime.label_regimes` solves the same problem for the HMM. Labelling by a
    named feature rather than by index is what makes a state sequence
    comparable across refits at all -- without it, a relabelled state reads as
    a regime change.
    """
    clean = features.dropna()
    if fit.states.size == 0 or vol_column not in clean.columns:
        return {i: f"state_{i}" for i in range(max(fit.centroids.shape[0], 1))}
    vols = pd.Series(clean[vol_column].to_numpy(), index=np.arange(len(clean)))
    means = {k: float(vols[fit.states == k].mean()) if (fit.states == k).any() else -np.inf
             for k in range(fit.centroids.shape[0])}
    bear = max(means, key=means.get)
    return {k: ("bear" if k == bear else "bull") for k in means}


def online_jump_states(
    features: pd.DataFrame,
    evaluation_index: pd.DatetimeIndex,
    n_states: int = 2,
    jump_penalty: float = 50.0,
    min_train_rows: int = 252,
    **fit_kwargs,
) -> pd.Series:
    """
    Causal state sequence: refit on each expanding prefix, keep only its LAST state.

    This is the shape item 2.1's caveat demands. A jump model fitted once over
    the whole window is as non-causal as a smoothed HMM -- the dynamic
    programme's backtrack pass propagates information from the end of the sample
    to every earlier row -- so the only state that saw no future is the one at
    the prefix boundary.

    `evaluation_index` names the dates to produce a state for; passing the
    backtest's rebalance dates rather than every day is both faithful to how the
    estimator would be used and what keeps this affordable, since the cost is
    one full fit per evaluation date.

    The scaler is fitted on each prefix, not on the whole panel, for the same
    reason: standardising with the full sample's mean and variance would leak
    them into every earlier row.
    """
    clean = features.dropna()
    out: dict[pd.Timestamp, str] = {}
    for date in evaluation_index:
        prefix = clean.loc[clean.index <= date]
        if len(prefix) < min_train_rows:
            continue
        fit = fit_jump_model(prefix, n_states=n_states, jump_penalty=jump_penalty,
                             **fit_kwargs)
        if not fit.states.size:
            continue
        labels = label_states_by_volatility(fit, prefix)
        out[date] = labels.get(int(fit.states[-1]), "unknown")
    return pd.Series(out, name="regime", dtype=object)
