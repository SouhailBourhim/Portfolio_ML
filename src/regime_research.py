"""
regime_research.py — filtered (causal) regime posteriors.

Addresses: P4, P2 — P4 primarily: this removes an in-window lookahead. The
smoothed posterior gives every historical training row a value computed from
observations after that row's own date, which is the "lookahead bias ... inflate
apparent performance" P4 names, here in its train/serve form rather than its
evaluation form. P2 secondarily, since the object being made causal is a regime
probability, and regimes are P2's mechanism.

WHY THIS MODULE EXISTS. `docs/EVALUATION_LIMITS.md` Limit #3 documents a
train/serve mismatch: `regime.predict_regime_posterior_series` calls hmmlearn's
`predict_proba`, which returns SMOOTHED posteriors

    gamma_t = P(state_t | x_1 ... x_T)

computed by the forward-BACKWARD algorithm. Every historical row of
`REGIME_BULL_PROB` at date t was therefore built using observations after t,
while at inference the same column is filtered. For
`RegimeConditionalStrategy` this is harmless -- only the last row is read, and
at t = T the smoothed posterior equals the filtered one -- but
`ml_signals.attach_regime_feature` trains on every row.

`docs/LITERATURE_IMPROVEMENTS.md` item 2.1 makes the ordering explicit and this
module exists to respect it: a statistical jump model fitted over a full window
carries the SAME in-window lookahead unless it is run online, so comparing it
against a smoothed HMM would confound the estimator change with the causality
defect the project has already documented. Fix causality first, then compare
estimators.

WHY THE FORWARD PASS IS WRITTEN OUT HERE. hmmlearn exposes no public filtered
posterior. The recursion is short and the alternative -- reaching into
`_do_forward_pass`, whose name and signature are private and have moved between
releases -- would be a silent breakage risk on upgrade. Only
`_compute_log_likelihood` is borrowed, for the emission matrix, and
`filtered_matches_smoothed_at_final_row` below checks the whole implementation
against hmmlearn's own smoothed output at t = T, where the two must agree
exactly. That is a positive control in the sense `docs/REACHABLE_CLAIMS.md` §6
argues for: a by-construction identity, checked before the output is trusted.

WHY NOT IN `src/regime.py`. Twelve DVC stages declare that file as a
dependency, so DVC hashes it whole and adding a function no stage calls would
invalidate all twelve -- three to five hours of recompute for code the pipeline
never reaches. Commit b9b45a3 set this rule for `src/metrics.py`. Promoting
filtered posteriors into `ml_signals.attach_regime_feature` is a deliberate
decision that DOES change published F7 numbers and SHOULD pay that rebuild; it
is not made here.

`grep -c regime_research dvc.yaml` must stay 0.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from regime import REGIME_FEATURES, HMMFit

log = logging.getLogger("regime_research")


def filtered_posterior_series(
    hmm_fit: HMMFit,
    feature_window: pd.DataFrame,
    features: list[str] = REGIME_FEATURES,
) -> pd.DataFrame:
    """
    Per-row FILTERED regime posterior: P(state_t | x_1 ... x_t).

    The causal counterpart of `regime.predict_regime_posterior_series`, with an
    identical signature, index and column convention (`"bull"`/`"bear"` via the
    fit's `label_map`) so it is a drop-in substitution. Returns an empty frame
    on a non-converged fit or an all-NaN window, exactly as that function does,
    so callers keep their existing `.empty` fallback.

    The recursion, in log space to avoid underflow over long windows:

        log a_1(i) = log pi_i + log b_i(x_1)
        log a_t(j) = log b_j(x_t) + logsumexp_i [ log a_{t-1}(i) + log A_ij ]
        filtered_t = softmax_i log a_t(i)

    Normalising at each step is what makes this filtered rather than a joint
    likelihood: row t conditions on observations up to t and no further.
    """
    if not hmm_fit.converged or hmm_fit.model is None:
        return pd.DataFrame(columns=["bull", "bear"])

    clean = feature_window[features].dropna()
    if clean.empty:
        return pd.DataFrame(columns=["bull", "bear"])

    model = hmm_fit.model
    X = hmm_fit.scaler.transform(clean.to_numpy())
    log_emission = model._compute_log_likelihood(X)
    n_obs, n_states = log_emission.shape

    with np.errstate(divide="ignore"):
        log_start = np.log(np.asarray(model.startprob_, dtype=float))
        log_trans = np.log(np.asarray(model.transmat_, dtype=float))

    log_alpha = np.empty((n_obs, n_states), dtype=float)
    log_alpha[0] = log_start + log_emission[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for t in range(1, n_obs):
        prior = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
        row = prior + log_emission[t]
        log_alpha[t] = row - logsumexp(row)

    proba = np.exp(log_alpha)
    columns = [hmm_fit.label_map[i] for i in range(n_states)]
    return pd.DataFrame(proba, index=clean.index, columns=columns)


def filtered_matches_smoothed_at_final_row(
    hmm_fit: HMMFit,
    feature_window: pd.DataFrame,
    features: list[str] = REGIME_FEATURES,
    tolerance: float = 1e-8,
) -> tuple[bool, float]:
    """
    Positive control for `filtered_posterior_series`.

    At t = T the forward-backward smoothed posterior IS the filtered one: there
    are no future observations for the backward pass to contribute. So the last
    row of this module's output must equal the last row of hmmlearn's
    `predict_proba` to numerical precision, on any fit and any window. A
    discrepancy means the recursion, the scaling or the label mapping is wrong,
    and it is detectable without any ground truth.

    Returns (passed, max_absolute_deviation).
    """
    from regime import predict_regime_posterior_series

    filtered = filtered_posterior_series(hmm_fit, feature_window, features)
    smoothed = predict_regime_posterior_series(hmm_fit, feature_window, features)
    if filtered.empty or smoothed.empty:
        return True, 0.0
    shared = [c for c in filtered.columns if c in smoothed.columns]
    deviation = float(
        np.max(np.abs(filtered.iloc[-1][shared].to_numpy()
                      - smoothed.iloc[-1][shared].to_numpy()))
    )
    return deviation <= tolerance, deviation
