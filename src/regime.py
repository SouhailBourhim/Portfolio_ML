"""
regime.py — HMM regime detection (bull/bear) on causal Phase 3 features.

Addresses: P2, P3 — detects latent market regimes from return/correlation
dynamics without lookahead; the detected regime conditions portfolio weights
in `strategies.RegimeConditionalStrategy` (P2: model parameters estimated in
one regime are not valid in another; P3: a "bear" regime is exactly where
diversification breaks down and defensive weighting matters most).

Trains ONLY on `REGIME_FEATURES` — three causal Phase 3 outputs
(`MARKET_RETURN`, `MARKET_VOL_SHORT`, `AVG_PAIRWISE_CORR`) already proven
leak-free end-to-end by `tests/test_phase3_integration.py`. Deliberately a
2-state model (bull/bear), not 3 (bull/bear/crisis): the `full_2021`
universe has only ~1,255 rows, and a 3rd state risks thin, seed-sensitive
"crisis" samples on that universe — see CLAUDE.md §12 for the trade-off.

Fits its own `StandardScaler` on ONLY the window it's given — never
globally — per the Phase 3 manifest's `global_standardization: false`
contract (CLAUDE.md §15.14). hmmlearn's EM fit is randomly initialized and
therefore seed-sensitive; `fit_hmm()` tries several deterministic restarts
and keeps the highest-log-likelihood converged fit, for reproducibility and
defensibility ("why this fit, not another" must have an answer).

Failure policy, same convention as the rest of this codebase: below
`min_regime_train_days`, or if every restart fails to converge, log a
WARNING and return a neutral fit whose posterior is 50/50 — never crash the
walk-forward loop over one thin or difficult window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("regime")

REGIME_FEATURES = ["MARKET_RETURN", "MARKET_VOL_SHORT", "AVG_PAIRWISE_CORR"]


@dataclass(frozen=True)
class HMMFit:
    """A converged (or neutral-fallback) HMM fit, kept for auditability."""

    model: object | None
    scaler: object | None
    converged: bool
    log_likelihood: float
    seed_used: int | None
    label_map: dict[int, str] = field(default_factory=dict)


def label_regimes(model, feature_names: list[str] = REGIME_FEATURES) -> dict[int, str]:
    """
    Map hmmlearn's unordered internal state indices to human labels by
    ranking each state's fitted MARKET_RETURN mean descending — the
    highest-mean-return state is "bull", the other is "bear". hmmlearn does
    not guarantee state ordering is stable across fits; this must be
    recomputed every time a model is fit, never assumed from a prior call.

    Addresses: P2 — regime labels must reflect what each state actually
    represents in THIS fit, not a hardcoded index.

    Raises:
        ValueError: if `model` was not fit with exactly 2 states — this
            module only supports the bull/bear MVP (CLAUDE.md §12).
    """
    if model.n_components != 2:
        raise ValueError(
            "label_regimes only supports 2-state (bull/bear) models; "
            f"got n_components={model.n_components}."
        )
    idx = feature_names.index("MARKET_RETURN")
    means = model.means_[:, idx]
    bull_state = int(np.argmax(means))
    bear_state = 1 - bull_state
    return {bull_state: "bull", bear_state: "bear"}


def fit_hmm(
    feature_window: pd.DataFrame,
    n_states: int = 2,
    n_restarts: int = 5,
    random_state_base: int = 0,
    covariance_type: str = "diag",
    min_regime_train_days: int = 252,
    features: list[str] = REGIME_FEATURES,
) -> HMMFit:
    """
    Fit a GaussianHMM on `feature_window[features]`.

    Addresses: P2, P3 — see module docstring.

    Multi-restarts with deterministic seeds (`random_state_base + i`), keeps
    the converged fit with the highest log-likelihood. Never raises: below
    `min_regime_train_days`, or if every restart fails to converge, returns
    a neutral `HMMFit` (`converged=False`, `model=None`) that
    `predict_regime_posterior` turns into a flat 50/50 posterior.

    `features` defaults to the module's `REGIME_FEATURES` but is
    configurable (wired from `params.yaml: regime.features` via
    `strategies.RegimeConditionalStrategy`) — `label_regimes` needs
    `"MARKET_RETURN"` to be present in whatever list is passed.
    """
    if n_states != 2:
        # label_regimes() would raise this anyway, but only AFTER n_restarts
        # GaussianHMM fits — fail immediately instead of wasting the EM runs.
        raise ValueError(
            f"fit_hmm only supports 2-state (bull/bear) models; got n_states={n_states}."
        )

    # Do not memoize this estimator.  Fixed restart seeds make the selected
    # state/label normally stable, but hmmlearn's EM likelihood is not
    # bit-for-bit deterministic on this runtime.  Reusing a prior fit would
    # therefore change the calculation instead of merely avoiding duplicate
    # work, which is unacceptable for a research result.
    return _fit_hmm_uncached(
        feature_window, n_states, n_restarts, random_state_base,
        covariance_type, min_regime_train_days, features,
    )


def _fit_hmm_uncached(
    feature_window: pd.DataFrame,
    n_states: int,
    n_restarts: int,
    random_state_base: int,
    covariance_type: str,
    min_regime_train_days: int,
    features: list[str],
) -> HMMFit:
    """The EM fit itself, unchanged — see `fit_hmm` for the contract.

    Addresses: P2, P3 — split out from `fit_hmm` so the multi-restart EM can
    be memoized on its inputs without the cache lookup sitting in the middle
    of the estimation logic. The returned `HMMFit` is treated as immutable by
    every caller (`predict_regime_posterior*` only reads it), which is what
    makes sharing one instance between strategies safe.
    """
    from hmmlearn.hmm import GaussianHMM

    clean = feature_window[features].dropna()
    if len(clean) < min_regime_train_days:
        log.warning(
            "regime: only %d usable rows (< min_regime_train_days=%d) — "
            "returning neutral 50/50 posterior.",
            len(clean), min_regime_train_days,
        )
        return HMMFit(None, None, False, float("nan"), None, {})

    from sklearn.preprocessing import StandardScaler

    X_raw = clean.to_numpy()
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    best: tuple[GaussianHMM, float, int] | None = None
    for i in range(n_restarts):
        seed = random_state_base + i
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=100,
            random_state=seed,
        )
        try:
            model.fit(X)
            if not model.monitor_.converged:
                continue
            # A fit can report "converged" (EM likelihood plateaued) yet still
            # be degenerate — e.g. one state is never visited, leaving a
            # transmat_ row summing to 0 instead of 1 — which model.score()
            # only discovers when it validates the fitted model. Treat that
            # the same as any other failed restart, not an uncaught crash.
            ll = model.score(X)
        except ValueError as exc:  # hmmlearn can raise on degenerate windows
            log.debug("regime: restart seed=%d raised %s", seed, exc)
            continue
        if best is None or ll > best[1]:
            best = (model, ll, seed)

    if best is None:
        log.warning(
            "regime: no restart converged out of %d attempts — returning neutral 50/50 posterior.",
            n_restarts,
        )
        return HMMFit(None, None, False, float("nan"), None, {})

    model, ll, seed = best
    label_map = label_regimes(model, features)
    return HMMFit(model, scaler, True, float(ll), seed, label_map)


def predict_regime_posterior_series(
    hmm_fit: HMMFit, feature_window: pd.DataFrame, features: list[str] = REGIME_FEATURES
) -> pd.DataFrame:
    """
    Return the FULL per-row regime posterior — one row per date in
    `feature_window` (after dropping NaN rows), columns `"bull"`/`"bear"`.

    Addresses: P2, P3 — the building block both `predict_regime_posterior`
    (single latest-row posterior, for `RegimeConditionalStrategy`'s
    bull/bear dispatch) and `ml_signals.attach_regime_feature` (every
    historical date, to train a supervised model) need — factored out here
    so the transform/`predict_proba` logic isn't duplicated across modules.

    Returns an EMPTY DataFrame (columns `"bull"`/`"bear"`, no rows) if
    `hmm_fit` did not converge or `feature_window` has no usable rows —
    callers needing a scalar fallback should check `.empty` and use the
    neutral 0.5, exactly like `predict_regime_posterior` does below.
    """
    if not hmm_fit.converged or hmm_fit.model is None:
        return pd.DataFrame(columns=["bull", "bear"])

    clean = feature_window[features].dropna()
    if clean.empty:
        return pd.DataFrame(columns=["bull", "bear"])

    X = hmm_fit.scaler.transform(clean.to_numpy())
    proba = hmm_fit.model.predict_proba(X)
    columns = [hmm_fit.label_map[i] for i in range(proba.shape[1])]
    return pd.DataFrame(proba, index=clean.index, columns=columns)


def predict_regime_posterior(
    hmm_fit: HMMFit, feature_window: pd.DataFrame, features: list[str] = REGIME_FEATURES
) -> dict[str, float]:
    """
    Return the LAST row's regime posterior, keyed by human label.

    Addresses: P2, P3 — the single number `RegimeConditionalStrategy` needs
    to decide bull vs. bear at τ. `feature_window` must already be sliced to
    `:τ` by the caller (the engine does this via `extras`) — this function
    only ever looks at the final row of whatever it's given. `features` must
    match whatever list `hmm_fit` was trained on (see `fit_hmm`).

    Returns the neutral `{"bull": 0.5, "bear": 0.5}` if `hmm_fit` did not
    converge (see `fit_hmm`'s failure policy), OR if `feature_window` has no
    usable (non-NaN) rows to predict on — the caller can treat this exactly
    like any other posterior, no special-casing required.
    """
    series = predict_regime_posterior_series(hmm_fit, feature_window, features)
    if series.empty:
        return {"bull": 0.5, "bear": 0.5}
    return {label: float(value) for label, value in series.iloc[-1].items()}
