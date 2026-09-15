"""
garch_horizon.py — multi-period GARCH variance, for Limit #4.

WHY THIS MODULE EXISTS. `docs/EVALUATION_LIMITS.md` Limit #4 records two defects
in `dcc_garch._dcc_covariance_uncached`, diagnosed by inspection and explicitly
NOT quantified:

  OFF-BY-ONE. `sigma_t = sigmas[-1]` (line 234) is arch's
  `conditional_volatility[T-1]`, which conditions on information through tau-1.
  The object wanted at rebalance tau is the one-step forecast
  sigma^2_{tau+1|tau} = omega + alpha r^2_tau + beta sigma^2_tau. No `forecast(`
  call exists in that module. This adds no lookahead -- it is conservative --
  but it discards the single property GARCH exists for, reacting to the most
  recent shock.

  HORIZON MISMATCH, the larger one. Rebalancing is monthly with a ~21-day hold,
  and the optimizer is handed a ONE-day conditional covariance scaled by 252
  (line 236). GARCH mean-reverts toward its unconditional level across those 21
  days, so a one-day variance systematically overstates how far the portfolio's
  realised risk will sit from the long-run level.

`docs/LITERATURE_IMPROVEMENTS.md` item 2.2 notes that the fix -- averaging the
1..H step forecast variances -- is not an ad-hoc repair but the standard
treatment, and should be presented with Drost & Nijman (1993) rather than as a
bug patch.

THE AGGREGATION, derived rather than guessed. For GARCH(1,1) with persistence
p = alpha + beta < 1 and unconditional variance s2_inf = omega / (1 - p), the
h-step forecast is a geometric decay toward that level:

    s2_{t+h|t} = s2_inf + p^(h-1) * ( s2_{t+1|t} - s2_inf )

A portfolio held for H periods experiences the SUM of those variances, so the
per-period variance that belongs in an annualised optimizer input is their mean:

    mean_{h=1..H} s2_{t+h|t} = s2_inf + ( s2_{t+1|t} - s2_inf ) * A(p, H)
    A(p, H) = (1 - p^H) / ( H * (1 - p) )

A(p, H) is the whole correction in one number: the share of today's deviation
from the long-run level that survives averaging over H periods. It is 1 at H = 1
and falls toward 1/(H(1-p)) as persistence rises. At the project's p ~ 0.95 and
H = 21 it is about 0.63, so a one-day input overstates the conditional deviation
by roughly 1.6x.

SCOPE, stated because it bounds the claim. This corrects the VARIANCE channel.
DCC's correlation matrix mean-reverts toward its own long-run target over the
same horizon, so a fully consistent multi-period covariance would aggregate that
too. `experiments/garch_horizon_quantified.py` measures the variance channel and
reports the correlation channel's persistence separately rather than folding an
unimplemented correction into a single number.

WHY NOT IN `src/dcc_garch.py`. Four DVC stages declare it as a dependency, and
applying this changes every `dcc_garch` result. `grep -c garch_horizon dvc.yaml`
must stay 0 until that re-run is taken deliberately.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("garch_horizon")


def aggregation_factor(persistence: float, horizon: int) -> float:
    """
    A(p, H) = (1 - p^H) / (H (1 - p)): the share of today's deviation from the
    unconditional variance that survives averaging over H periods.

    Returns 1.0 at H = 1 (no aggregation) and 1.0 at p = 1 (a random walk in
    variance never mean-reverts, so every horizon carries the full deviation).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    if not 0.0 <= persistence <= 1.0:
        raise ValueError(f"persistence must be in [0, 1]; got {persistence}")
    if horizon == 1:
        return 1.0
    if np.isclose(persistence, 1.0):
        return 1.0
    return float((1.0 - persistence ** horizon) / (horizon * (1.0 - persistence)))


def one_step_forecast_variance(
    omega: float, alpha: float, beta: float, last_return: float, last_variance: float
) -> float:
    """
    sigma^2_{t+1|t} = omega + alpha * r_t^2 + beta * sigma^2_t.

    The quantity `dcc_garch` should be using at rebalance tau and is not: it
    takes `conditional_volatility[T-1]`, which is sigma^2_{tau|tau-1} and has
    not seen r_tau.
    """
    return float(omega + alpha * last_return ** 2 + beta * last_variance)


def multi_horizon_average_variance(
    omega: float, alpha: float, beta: float, one_step_variance: float, horizon: int
) -> float:
    """
    Mean of the 1..H step forecast variances, in closed form.

    This is the per-period variance a portfolio held for `horizon` periods
    actually experiences, and therefore what belongs in an annualised optimizer
    input under a monthly rebalance.
    """
    persistence = alpha + beta
    if persistence >= 1.0:
        # Integrated GARCH: no unconditional variance exists, and every horizon
        # carries today's level forward unchanged. Returning the one-step value
        # is the correct limit, not a fallback.
        return float(one_step_variance)
    uncond = omega / (1.0 - persistence)
    factor = aggregation_factor(persistence, horizon)
    return float(uncond + (one_step_variance - uncond) * factor)


def forecast_variance_path(
    omega: float, alpha: float, beta: float, one_step_variance: float, horizon: int
) -> np.ndarray:
    """
    The explicit 1..H step forecast path, by direct recursion.

    Exists to check `multi_horizon_average_variance` against the recursion it
    claims to summarise -- a closed form that has drifted from its own
    derivation is the kind of error no output inspection reveals.
    """
    persistence = alpha + beta
    path = np.empty(horizon, dtype=float)
    path[0] = one_step_variance
    for h in range(1, horizon):
        path[h] = omega + persistence * path[h - 1]
    return path
