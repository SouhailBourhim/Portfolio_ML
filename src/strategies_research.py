"""
strategies_research.py — strategy rungs that no DVC stage depends on.

WHY THIS MODULE EXISTS, and the rule for what belongs in it. `src/strategies.py`
is a declared dependency of SIXTEEN DVC stages, so DVC hashes the whole file and
any edit invalidates all sixteen — three to five hours of recompute. Commit
b9b45a3 paid that toll once for research instruments added to `src/metrics.py`,
moved them to `src/inference.py`, and set the test: *whether a DVC stage calls
the code, not whether it is the same kind of thing*.

A rung here is a real `Strategy` and behaves identically to one in
`strategies.py`. It lives apart only because no pipeline stage constructs it, so
hashing it into the pipeline would invalidate sixteen stages to no effect. If a
rung is ever promoted into `params.yaml` and the DVC graph, move it across and
pay the rebuild deliberately — that rebuild is then real, because its output
would genuinely change.

`grep -c strategies_research dvc.yaml` must stay 0. That is the invariant.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

import telemetry
from strategies import Strategy

log = logging.getLogger("strategies_research")


class MinCVaR(Strategy):
    """
    Mean-CVaR via the Rockafellar-Uryasev linear program.

    Addresses: P1, P3 — every other optimizer in this module targets variance
    or the Sharpe ratio. `metrics` REPORTS maximum drawdown and the Calmar
    ratio, but until now nothing OPTIMIZED for downside, so the drawdown
    numbers in the report belonged to strategies that were never trying to
    achieve them. This rung asks a different question of the same data rather
    than re-estimating the same inputs with a better covariance estimator.

    Rockafellar & Uryasev (2000) show that CVaR at confidence beta is the
    value of a convex program: minimising

        zeta + 1 / ((1 - beta) T) * SUM_t max(0, -w'r_t - zeta)

    over (w, zeta) yields both the optimal portfolio and, at the optimum,
    zeta = VaR. Introducing u_t for the positive part makes it LINEAR, so the
    solve is an LP rather than the SLSQP used elsewhere. That is the practical
    argument for this rung: an LP has a global optimum and a certificate, so
    unlike `_optimize_weights` it needs no perturbed retry and no equal-weight
    degradation path. The fallback below exists for infeasibility and solver
    error, not for the ordinary non-convergence SLSQP hits.

    `return_weight` (lambda) trades expected return against tail risk:
    0.0 is pure minimum-CVaR, the downside analogue of `MinVarianceLW`, and is
    the default because it is the rung comparable to the existing risk
    strategies. Any positive value makes this genuinely "mean-CVaR" and is a
    judgement call that must be reported as one, exactly as the turnover
    penalty is -- it is not estimated here.

    Scenarios are SIMPLE returns, converted from the log returns the engine
    passes. CVaR is defined on realised portfolio loss, and a portfolio's
    simple return is w'(exp(r) - 1); using log returns directly would apply
    the weights to the wrong quantity. The covariance-based rungs make the
    same approximation in the other direction and it is second-order for them,
    but here the tail is the estimand, so it is worth being exact about.

    The constraint matrix is built SPARSE. It has T rows and n + 1 + T columns,
    and on an expanding window T reaches several thousand, where a dense
    representation would cost hundreds of megabytes per rebalance for a matrix
    that is a scaled identity apart from n + 1 dense columns.
    """

    name = "min_cvar"

    def __init__(
        self,
        max_weight: float = 0.25,
        confidence: float = 0.95,
        return_weight: float = 0.0,
    ) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1); got {confidence}")
        if return_weight < 0.0:
            raise ValueError(f"return_weight must be >= 0; got {return_weight}")
        self.max_weight = max_weight
        self.confidence = confidence
        self.return_weight = return_weight

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        from scipy import sparse
        from scipy.optimize import linprog

        assets = train_returns.columns
        n = len(assets)
        if n * self.max_weight < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible constraints: {n} assets x cap {self.max_weight} < 1 — "
                f"weights cannot sum to 1. Raise max_weight or add assets."
            )

        scenarios = np.expm1(train_returns.to_numpy(dtype=float))
        n_scen = scenarios.shape[0]
        tail_scale = 1.0 / ((1.0 - self.confidence) * n_scen)

        # Variables: [w (n) | zeta (1) | u (n_scen)]
        cost = np.concatenate([
            -self.return_weight * scenarios.mean(axis=0),
            np.array([1.0]),
            np.full(n_scen, tail_scale),
        ])
        # Tail constraint, one row per scenario: -w'r_t - zeta - u_t <= 0.
        a_ub = sparse.hstack([
            sparse.csr_matrix(-scenarios),
            sparse.csr_matrix(-np.ones((n_scen, 1))),
            -sparse.eye(n_scen, format="csr"),
        ], format="csr")
        b_ub = np.zeros(n_scen)
        # Budget: sum(w) == 1, touching neither zeta nor u.
        a_eq = sparse.csr_matrix(
            np.concatenate([np.ones(n), np.zeros(1 + n_scen)]).reshape(1, -1)
        )
        bounds = (
            [(0.0, self.max_weight)] * n + [(None, None)] + [(0.0, None)] * n_scen
        )

        result = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=np.array([1.0]),
                         bounds=bounds, method="highs")
        if not result.success:
            log.warning(
                "%s: LP did not solve (%s) — falling back to equal weights for this "
                "rebalance.", self.name, result.message,
            )
            telemetry.record(
                telemetry.FitRecord(
                    model_requested=self.name,
                    model_effective="equal_weight",
                    fit_status=telemetry.STATUS_FALLBACK,
                    n_training_rows=len(train_returns),
                    fallback_reason=f"linprog failed: {result.message}",
                )
            )
            return pd.Series(1.0 / n, index=assets)

        return Strategy._as_weight_series(result.x[:n], assets, self.max_weight)
