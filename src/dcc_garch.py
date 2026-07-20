"""
dcc_garch.py — Multivariate dynamic covariance via DCC-GARCH (Engle 2002).

Addresses: P1, P2, P3 — the final rung of the covariance ablation ladder
(sample → Ledoit-Wolf → EWMA → DCC-GARCH). Unlike EWMA's single global decay
parameter, DCC-GARCH lets EACH asset's own volatility evolve under its own
fitted GARCH(1,1) process (P2 — volatility clustering), then models how
CORRELATIONS themselves move over time via a second, shared recursion —
directly targeting P3 (diversification breakdown: correlations spike in
crises) at the covariance-estimation level, not just observed after the fact.

`arch` ships only UNIVARIATE GARCH (CLAUDE.md §3.2) — the multivariate DCC
step below is a standard two-stage quasi-MLE estimator (Engle 2002), not
provided by any dependency and not previously implemented anywhere in this
repo.

Two-stage procedure, run once per walk-forward `fit()` call on the
already-causally-sliced training window (the engine hands us `train_returns
= returns.loc[:τ]`; nothing here ever sees data past τ):

  1. Fit a univariate GARCH(1,1) to each asset's returns independently →
     per-asset conditional volatility σ_i,t.
  2. Standardize: ε_i,t = r_i,t / σ_i,t. Estimate the DCC recursion
     Q_t = (1−a−b)·Q̄ + a·ε_{t-1}ε_{t-1}ᵀ + b·Q_{t-1} via `scipy.optimize` on
     the DCC quasi-likelihood, where Q̄ is the unconditional covariance of
     the standardized residuals computed directly ("correlation targeting",
     Engle 2002's standard practical simplification — only (a, b) are
     numerically optimized, not the full Q̄ matrix). Correlation
     R_t = diag(Q_t)^(−1/2) Q_t diag(Q_t)^(−1/2); covariance
     Σ_t = D_t R_t D_t, annualized ×252.

GARCH is fit with a zero mean (`mean="Zero"`) — daily return means are tiny
relative to volatility, and estimating them adds convergence risk for no
material benefit at this horizon; a deliberate simplification, not an
oversight.

Failure policy, same convention as every other estimator in this codebase
(`strategies._optimize_weights`): a GARCH fit that fails to converge for any
asset, or a DCC optimization that fails, logs a WARNING naming the offending
asset/window and falls back to Ledoit-Wolf shrinkage covariance for that
`fit()` call — never crashes the walk-forward loop over one bad window.
`EEM` (CLAUDE.md §8.4, stationarity-AMBIGUOUS) is the asset most likely to
trigger this path; that is an expected, monitored case, not a bug.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

log = logging.getLogger("dcc_garch")

TRADING_DAYS_PER_YEAR = 252


class DCCGarchNonConvergence(Exception):
    """Internal signal: GARCH or DCC estimation failed to converge on this window."""


def _fit_univariate_garch(
    returns: pd.Series, p: int, q: int, rescale_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit GARCH(p, q) to one asset's returns; return (conditional_vol, std_resid).

    Addresses: P2 — per-asset volatility clustering, the first stage DCC needs
    before it can separately model correlation dynamics.
    """
    from arch import arch_model

    scaled = returns.to_numpy() * rescale_factor
    model = arch_model(scaled, vol="Garch", p=p, q=q, mean="Zero", rescale=False)
    result = model.fit(disp="off", show_warning=False)

    sigma = np.asarray(result.conditional_volatility) / rescale_factor
    std_resid = np.asarray(result.std_resid)

    if result.convergence_flag != 0 or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(std_resid)):
        raise DCCGarchNonConvergence(f"GARCH({p},{q}) failed to converge on asset '{returns.name}'")
    return sigma, std_resid


def _dcc_recursion_final_q(
    std_resid: np.ndarray, a: float, b: float, q_bar: np.ndarray
) -> np.ndarray:
    """Run the Q_t recursion to the end of the window; return the final Q_t."""
    q_t = q_bar.copy()
    for t in range(1, len(std_resid)):
        eps_prev = std_resid[t - 1]
        q_t = (1 - a - b) * q_bar + a * np.outer(eps_prev, eps_prev) + b * q_t
    return q_t


def _dcc_neg_loglikelihood(params: np.ndarray, std_resid: np.ndarray, q_bar: np.ndarray) -> float:
    a, b = params
    if a < 0 or b < 0 or a + b >= 1.0:
        return 1e10

    q_t = q_bar.copy()
    nll = 0.0
    for t in range(1, len(std_resid)):
        eps_prev = std_resid[t - 1]
        q_t = (1 - a - b) * q_bar + a * np.outer(eps_prev, eps_prev) + b * q_t
        d = np.sqrt(np.diag(q_t))
        if np.any(d <= 0):
            return 1e10
        r_t = q_t / np.outer(d, d)
        try:
            sign, logdet = np.linalg.slogdet(r_t)
            if sign <= 0:
                return 1e10
            r_inv = np.linalg.inv(r_t)
        except np.linalg.LinAlgError:
            return 1e10
        eps_t = std_resid[t]
        nll += 0.5 * (logdet + eps_t @ r_inv @ eps_t)

    if not np.isfinite(nll):
        return 1e10
    return nll


def _fit_dcc(
    std_resid: np.ndarray, a_init: float, b_init: float
) -> tuple[float, float, np.ndarray]:
    """
    Estimate (a, b) via QMLE on the DCC likelihood; Q̄ is variance-targeted
    (the unconditional covariance of the standardized residuals), not
    optimized — Engle (2002)'s standard two-stage simplification.
    """
    q_bar = np.cov(std_resid, rowvar=False)
    result = minimize(
        _dcc_neg_loglikelihood,
        x0=np.array([a_init, b_init]),
        args=(std_resid, q_bar),
        method="Nelder-Mead",
        bounds=[(1e-6, 0.3), (1e-6, 0.999)],
    )
    a, b = float(result.x[0]), float(result.x[1])
    if not result.success or a + b >= 1.0:
        raise DCCGarchNonConvergence(f"DCC optimization failed: {result.message}")
    return a, b, q_bar


def dcc_covariance(
    train_returns: pd.DataFrame,
    garch_p: int = 1,
    garch_q: int = 1,
    dcc_a_init: float = 0.02,
    dcc_b_init: float = 0.95,
    rescale_factor: float = 100.0,
) -> np.ndarray:
    """
    Fit DCC-GARCH on `train_returns` and return the latest (τ-dated)
    annualized covariance matrix, in `train_returns.columns` order.

    Addresses: P1, P2, P3 — see module docstring.

    Falls back to annualized Ledoit-Wolf shrinkage covariance, with a logged
    WARNING naming the cause, if any asset's GARCH fit or the DCC
    optimization fails to converge on this window (see `strategies.
    MinVarianceLW` — same estimator, same annualization convention).
    """
    assets = train_returns.columns
    n_obs, n_assets = train_returns.shape

    try:
        sigmas = np.zeros((n_obs, n_assets))
        std_resids = np.zeros((n_obs, n_assets))
        for i, asset in enumerate(assets):
            sigma, std_resid = _fit_univariate_garch(
                train_returns[asset], garch_p, garch_q, rescale_factor
            )
            sigmas[:, i] = sigma
            std_resids[:, i] = std_resid

        a, b, q_bar = _fit_dcc(std_resids, dcc_a_init, dcc_b_init)
        q_t = _dcc_recursion_final_q(std_resids, a, b, q_bar)

        d = np.sqrt(np.diag(q_t))
        r_t = q_t / np.outer(d, d)

        sigma_t = sigmas[-1]
        cov_daily = np.outer(sigma_t, sigma_t) * r_t
        cov_annual = cov_daily * TRADING_DAYS_PER_YEAR

        if not np.all(np.isfinite(cov_annual)):
            raise DCCGarchNonConvergence("Final covariance matrix contains non-finite values")
        return cov_annual

    except (DCCGarchNonConvergence, np.linalg.LinAlgError, ValueError) as exc:
        log.warning(
            "DCC-GARCH failed on this window (%d assets, %d rows): %s — "
            "falling back to Ledoit-Wolf shrinkage covariance for this rebalance.",
            n_assets, n_obs, exc,
        )
        lw = LedoitWolf().fit(train_returns.to_numpy())
        return lw.covariance_ * TRADING_DAYS_PER_YEAR
