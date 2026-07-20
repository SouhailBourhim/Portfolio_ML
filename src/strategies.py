"""
strategies.py — Portfolio strategy interface and Markowitz baselines.

The `Strategy` ABC is the single seam between models and the walk-forward
engine (backtest.py): the engine slices data and polices weights; strategies
only ever see their train window. Phase 4's ML models (HMM-conditioned
weights, dynamic-covariance optimizers) plug in through this same interface.

Addresses: P1 — constrained optimization (long-only, per-asset cap) tempers
the instability that noisy covariance estimates cause; the baselines here
deliberately use the naive sample moments so Phase 4's ablation ladder
(Ledoit-Wolf → EWMA → DCC-GARCH) has an honest floor to improve on.
Addresses: P4 — one strict interface means one seam the engine can police
for lookahead.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger("strategies")

TRADING_DAYS_PER_YEAR = 252


class Strategy(ABC):
    """
    Contract for every portfolio strategy, baseline or ML.

    fit() receives ONLY past data — enforced by the engine's slicing, never
    trusted to subclasses — and returns long-only weights summing to 1,
    indexed exactly by the train window's columns.

    `extras` is the Phase 4 seam: a mapping of auxiliary frames (macro
    features, regime labels), each pre-sliced BY THE ENGINE to the train
    window. Baselines accept and ignore it; Phase 4 models consume it.
    """

    name: str = "abstract"

    @abstractmethod
    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        """Return target weights (index == train_returns.columns, sum 1, ≥ 0)."""

    @staticmethod
    def _as_weight_series(values: np.ndarray, assets: pd.Index) -> pd.Series:
        """Clip optimizer dust (−1e-12) to 0 and renormalize to sum exactly 1."""
        w = pd.Series(np.clip(values, 0.0, None), index=assets)
        return w / w.sum()


class EqualWeight(Strategy):
    """
    1/N portfolio.

    Addresses: P1 — the DeMiguel, Garlappi & Uppal (2009) result: naive
    equal weighting beats most optimized portfolios out-of-sample because
    it estimates nothing and therefore cannot overfit estimation noise.
    This is the honest hurdle every other strategy must clear net of costs.
    """

    name = "equal_weight"

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        n = train_returns.shape[1]
        return pd.Series(1.0 / n, index=train_returns.columns)


def _optimize_weights(
    objective: Callable[[np.ndarray], float],
    assets: pd.Index,
    max_weight: float,
    strategy_name: str,
) -> pd.Series:
    """
    Shared SLSQP wrapper: long-only bounds (0, max_weight), Σw = 1, x0 = 1/N.

    Addresses: P1 — the cap stops noisy estimates from producing the
    concentrated corner solutions that collapse out-of-sample.

    Failure policy: retry once from a perturbed start, then fall back to
    equal weights with a WARNING — a logged fallback mid-backtest is honest;
    a crash hides everything after it, and silently bad weights hide worse
    (§13.13: silent loss is a bug, loud degradation is not).

    Raises:
        ValueError: if n_assets × max_weight < 1 (constraints infeasible).
    """
    n = len(assets)
    if n * max_weight < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible constraints: {n} assets × cap {max_weight} < 1 — "
            f"weights cannot sum to 1. Raise max_weight or add assets."
        )

    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    x0 = np.full(n, 1.0 / n)

    for attempt, start in enumerate((x0, x0 + np.random.default_rng(0).normal(0, 0.01, n))):
        start = np.clip(start, 0.0, max_weight)
        start = start / start.sum()
        result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            return Strategy._as_weight_series(result.x, assets)
        log.debug("%s: SLSQP attempt %d failed: %s", strategy_name, attempt + 1, result.message)

    log.warning(
        "%s: optimizer failed twice (%s) — falling back to equal weights for this rebalance.",
        strategy_name, result.message,
    )
    return pd.Series(1.0 / n, index=assets)


class MinVariance(Strategy):
    """
    Minimum-variance portfolio: min wᵀΣw, long-only, per-asset cap.

    Addresses: P1 — ignores expected returns entirely (the noisiest input)
    and still suffers from covariance noise: the textbook case for the
    dynamic-covariance upgrades of Phase 4.
    """

    name = "min_variance"

    def __init__(self, max_weight: float = 0.25) -> None:
        self.max_weight = max_weight

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        cov = train_returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
        return _optimize_weights(
            lambda w: float(w @ cov @ w), train_returns.columns, self.max_weight, self.name
        )


class MinVarianceLW(Strategy):
    """
    Minimum-variance with a Ledoit-Wolf shrunk covariance matrix.

    Addresses: P1 — the first rung of the covariance ablation ladder
    (sample → Ledoit-Wolf shrinkage → EWMA → DCC-GARCH). Shrinkage pulls
    the noisy sample covariance toward a structured target, with the
    shrinkage intensity estimated from the data itself (Ledoit & Wolf
    2004). Comparing this against plain MinVariance isolates how much of
    the P1 problem simple statistical regularization already fixes —
    before any ML is involved. If shrinkage alone closes most of the gap
    to 1/N, that materially changes what Phase 4 has to prove.
    """

    name = "min_variance_lw"

    def __init__(self, max_weight: float = 0.25) -> None:
        self.max_weight = max_weight

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(train_returns.to_numpy())
        cov = lw.covariance_ * TRADING_DAYS_PER_YEAR
        return _optimize_weights(
            lambda w: float(w @ cov @ w), train_returns.columns, self.max_weight, self.name
        )


class MinVarianceEWMA(Strategy):
    """
    Minimum-variance with an exponentially-weighted (EWMA) covariance matrix.

    Addresses: P1, P2 — the second rung of the covariance ablation ladder
    (sample → Ledoit-Wolf → EWMA → DCC-GARCH). Unlike the flat sample window
    MinVariance/MinVarianceLW use, EWMA weights recent observations more
    heavily (RiskMetrics-style decay via `halflife_days`), so the covariance
    estimate reacts to a volatility/correlation shift within the training
    window instead of averaging it away — a direct P2 (non-stationarity) fix
    for covariance estimation, not just P1 regularization.

    Known caveat inherited from Phase 1 (CLAUDE.md §8.4): EEM is
    stationarity-AMBIGUOUS. EWMA's recency-weighting arguably self-mitigates
    a stale structural break better than a flat sample window would, but
    that is not a guarantee — no special-case handling is added here.
    """

    name = "min_variance_ewma"

    def __init__(self, max_weight: float = 0.25, halflife_days: int = 63) -> None:
        self.max_weight = max_weight
        self.halflife_days = halflife_days

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        ewm_cov = train_returns.ewm(halflife=self.halflife_days).cov()
        cov = ewm_cov.loc[train_returns.index[-1]].to_numpy() * TRADING_DAYS_PER_YEAR
        return _optimize_weights(
            lambda w: float(w @ cov @ w), train_returns.columns, self.max_weight, self.name
        )


class DCCGarchStrategy(Strategy):
    """
    Minimum-variance with a DCC-GARCH covariance matrix (Engle 2002).

    Addresses: P1, P2, P3 — the fourth and final rung of the covariance
    ablation ladder (sample → Ledoit-Wolf → EWMA → DCC-GARCH). Unlike EWMA's
    single global decay parameter, DCC-GARCH lets each asset's own
    volatility evolve under its own fitted GARCH(1,1) process and models
    correlation dynamics separately — directly targeting P3 (correlations
    spiking in a crisis) at the estimation level. See `dcc_garch.py` for the
    full two-stage estimator and its non-convergence fallback policy.

    Known caveat inherited from Phase 1 (CLAUDE.md §8.4): EEM is
    stationarity-AMBIGUOUS and the asset most likely to trigger
    `dcc_garch.py`'s Ledoit-Wolf fallback — an expected, monitored case.
    """

    name = "dcc_garch"

    def __init__(
        self,
        max_weight: float = 0.25,
        garch_p: int = 1,
        garch_q: int = 1,
        dcc_a_init: float = 0.02,
        dcc_b_init: float = 0.95,
        rescale_factor: float = 100.0,
    ) -> None:
        self.max_weight = max_weight
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.dcc_a_init = dcc_a_init
        self.dcc_b_init = dcc_b_init
        self.rescale_factor = rescale_factor

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        from dcc_garch import dcc_covariance

        cov = dcc_covariance(
            train_returns,
            garch_p=self.garch_p,
            garch_q=self.garch_q,
            dcc_a_init=self.dcc_a_init,
            dcc_b_init=self.dcc_b_init,
            rescale_factor=self.rescale_factor,
        )
        return _optimize_weights(
            lambda w: float(w @ cov @ w), train_returns.columns, self.max_weight, self.name
        )


class MaxSharpe(Strategy):
    """
    Maximum-Sharpe (tangency) portfolio: max (wᵀμ − rf)/√(wᵀΣw), long-only, cap.

    Addresses: P1 — the classical Markowitz benchmark, using deliberately
    naive sample moments (annualized ×252). Its in-sample optimality and
    out-of-sample fragility is the exact P1 phenomenon the project exists
    to fix; it must be in the comparison for the ML story to mean anything.
    """

    name = "max_sharpe"

    def __init__(self, max_weight: float = 0.25, risk_free_annual: float = 0.0) -> None:
        self.max_weight = max_weight
        self.risk_free_annual = risk_free_annual

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        mu = train_returns.mean().to_numpy() * TRADING_DAYS_PER_YEAR
        cov = train_returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR

        def neg_sharpe(w: np.ndarray) -> float:
            vol = float(np.sqrt(w @ cov @ w))
            if vol < 1e-12:
                return 0.0
            return -(float(w @ mu) - self.risk_free_annual) / vol

        return _optimize_weights(neg_sharpe, train_returns.columns, self.max_weight, self.name)


class RegimeConditionalStrategy(Strategy):
    """
    HMM regime detection gating a bull sub-strategy vs. a bear sub-strategy.

    Addresses: P2, P3 — market parameters estimated in one regime are not
    valid in another (P2); a "bear" regime is exactly where diversification
    breaks down and defensive weighting matters most (P3). Detects the
    regime from causal Phase 3 features (`regime.REGIME_FEATURES`) via a
    2-state HMM (see `regime.py` for why 2 states, not 3), then hands the
    ENTIRE decision to whichever already-tested Phase 2 baseline matches:
    `bull_strategy` (default `MaxSharpe`) or `bear_strategy` (default
    `MinVarianceLW`). This strategy adds no new optimizer — the only new
    surface is regime detection plus a switch, which is what makes it
    defensible line-by-line (CLAUDE.md §12, decision 2).

    Reuses the exact `extras["features"]` key Phase 3 already established
    (`tests/test_phase3_integration.py`) — no new engine contract. Falls
    back to equal weight if `extras["features"]` is missing or empty (e.g.
    a caller that never wired Phase 3 features in), and defers to
    `regime.fit_hmm`'s own neutral-posterior policy for thin/non-converging
    windows — but resolves that neutral case to the DEFENSIVE sub-strategy
    (`bear_strategy`), not an arbitrary tie-break: when the model has no
    confident regime read, guessing bullish is the wrong direction to err.

    Known trade-off, accepted for the MVP (CLAUDE.md §12, decision 2): a
    hard regime switch can move weights sharply right at a regime boundary
    (a turnover/cost spike that day) — monitor via `regime_log` rather than
    adding a hysteresis hyperparameter up front.
    """

    name = "regime_conditional"

    def __init__(
        self,
        bull_strategy: Strategy | None = None,
        bear_strategy: Strategy | None = None,
        n_states: int = 2,
        n_restarts: int = 5,
        random_state_base: int = 0,
        covariance_type: str = "diag",
        min_regime_train_days: int = 252,
        features: list[str] | None = None,
    ) -> None:
        from regime import REGIME_FEATURES

        self.bull_strategy = bull_strategy if bull_strategy is not None else MaxSharpe()
        self.bear_strategy = bear_strategy if bear_strategy is not None else MinVarianceLW()
        self.n_states = n_states
        self.n_restarts = n_restarts
        self.random_state_base = random_state_base
        self.covariance_type = covariance_type
        self.min_regime_train_days = min_regime_train_days
        self.features = features if features is not None else REGIME_FEATURES
        # Diagnostic-only: the engine reuses this same instance across the
        # whole backtest and neither assists nor prevents this kind of
        # internal state (src/backtest.py docstring). Never read by fit().
        self.regime_log: list[dict] = []

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        n = train_returns.shape[1]
        if not extras or "features" not in extras or extras["features"].empty:
            return pd.Series(1.0 / n, index=train_returns.columns)

        from regime import fit_hmm, predict_regime_posterior

        feature_window = extras["features"]
        hmm_fit = fit_hmm(
            feature_window,
            n_states=self.n_states,
            n_restarts=self.n_restarts,
            random_state_base=self.random_state_base,
            covariance_type=self.covariance_type,
            min_regime_train_days=self.min_regime_train_days,
            features=self.features,
        )
        posterior = predict_regime_posterior(hmm_fit, feature_window, features=self.features)

        if hmm_fit.converged:
            regime_label = max(posterior, key=posterior.get)
        else:
            # No confident regime read — default to the defensive
            # sub-strategy rather than an arbitrary tie-break on the
            # neutral 50/50 posterior (see class docstring).
            regime_label = "bear"

        self.regime_log.append(
            {
                "date": train_returns.index[-1],
                "regime": regime_label,
                "posterior": posterior,
                "converged": hmm_fit.converged,
            }
        )

        sub_strategy = self.bull_strategy if regime_label == "bull" else self.bear_strategy
        return sub_strategy.fit(train_returns, extras)
