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
