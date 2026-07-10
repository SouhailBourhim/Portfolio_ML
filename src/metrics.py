"""
metrics.py — Out-of-sample portfolio performance metrics.

All metrics operate on SIMPLE daily returns (the backtest engine's output),
never on log-returns — mixing the two silently misstates performance.

Addresses: P4 — these are the out-of-sample yardsticks every strategy is
judged by; the Deflated Sharpe Ratio explicitly corrects for the number of
strategies tried, the core mechanism of backtest overfitting.
"""

import logging
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger("metrics")

TRADING_DAYS_PER_YEAR = 252

# Euler–Mascheroni constant, used in the expected-maximum-Sharpe term of the DSR
_EULER_GAMMA = 0.5772156649015329


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Geometric annualized return from simple periodic returns.

    Addresses: P4 — geometric (not arithmetic) compounding is what an
    investor actually experiences out-of-sample.
    """
    if returns.empty:
        return float("nan")
    total_growth = float((1 + returns).prod())
    if total_growth <= 0:
        return -1.0  # portfolio wiped out (possible with large negative simple returns)
    return total_growth ** (periods / len(returns)) - 1


def annualized_sharpe(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sharpe ratio of simple periodic returns.

    Addresses: P4 — computed on out-of-sample returns only; the in-sample
    Sharpe of an optimized portfolio is upward-biased by construction.

    Args:
        returns: Simple periodic (daily) returns.
        risk_free_annual: Annual risk-free rate; converted to per-period
            geometrically. MVP default 0.0 (documented simplification).
        periods: Periods per year for annualization.
    """
    if len(returns) < 2 or float(returns.std()) < 1e-12:
        return float("nan")
    rf_periodic = (1 + risk_free_annual) ** (1 / periods) - 1
    excess = returns - rf_periodic
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of the cumulative wealth curve. Always ≤ 0.

    Addresses: P3, P4 — drawdown is where diversification breakdown shows up
    in money terms; a strategy's worst stretch matters more to a real
    portfolio manager than its average.
    """
    if returns.empty:
        return float("nan")
    wealth = (1 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualized return divided by absolute max drawdown.

    Addresses: P4 — rewards return earned per unit of worst-case pain,
    a robustness-first alternative to Sharpe.
    """
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0.0:
        return float("nan")
    return annualized_return(returns, periods) / abs(mdd)


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized mean active return over tracking error, vs a benchmark.

    Addresses: P4 — "did the strategy beat the honest hurdle?" quantified.
    The Phase 2 benchmark is the equal-weight net series (DeMiguel et al.
    2009). A strategy measured against itself has zero tracking error →
    returns NaN by construction (documented, not a bug).
    """
    active = returns.align(benchmark, join="inner")[0] - benchmark.align(returns, join="inner")[0]
    if len(active) < 2 or float(active.std()) == 0.0:
        return float("nan")
    return float(active.mean() / active.std() * np.sqrt(periods))


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes: Sequence[float]) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Probability that the observed Sharpe is genuinely positive after
    correcting for (a) non-normal returns (skew, kurtosis) and (b) the
    number of strategies tried — selection among N trials inflates the
    best observed Sharpe even if all strategies are worthless.

    Addresses: P4 — this is the project's primary defense against
    "we tried many things and reported the best one."

    Honest limitation: N counts the strategies compared in THIS run
    (len(trial_sharpes)); it does not yet accumulate across historical
    experiment sessions. Accumulation policy is a Phase 5 decision —
    until then, treat DSR as directional, not exact.

    Args:
        returns: Simple daily returns of the candidate strategy (OOS).
        trial_sharpes: Per-period (daily, NON-annualized) Sharpe ratios of
            every strategy tried in the comparison, including this one.

    Returns:
        Probability in [0, 1]; values near 1 mean the Sharpe is unlikely
        to be a selection artifact.
    """
    n_trials = len(trial_sharpes)
    if n_trials == 0:
        raise ValueError("trial_sharpes must contain at least the candidate's own Sharpe.")

    r = returns.dropna()
    t_obs = len(r)
    if t_obs < 3 or float(r.std()) == 0.0:
        return float("nan")

    sr = float(r.mean() / r.std())          # per-period, non-annualized
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0            # pandas gives excess kurtosis; formula wants Pearson

    if n_trials == 1:
        # Degenerates to the Probabilistic Sharpe Ratio against SR* = 0.
        warnings.warn(
            "deflated_sharpe_ratio called with a single trial — no selection "
            "correction possible; result is the PSR against SR*=0.",
            UserWarning,
            stacklevel=2,
        )
        sr_star = 0.0
    else:
        var_trials = float(np.var(trial_sharpes, ddof=1))
        sr_star = float(np.sqrt(var_trials)) * (
            (1 - _EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
            + _EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
        )

    denominator = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")

    z = (sr - sr_star) * np.sqrt(t_obs - 1) / denominator
    return float(norm.cdf(z))


def summarize(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    turnover: pd.Series,
    benchmark_net: pd.Series | None = None,
    trial_sharpes: Sequence[float] | None = None,
    risk_free_annual: float = 0.0,
) -> dict[str, float]:
    """
    Full metric panel for one backtest result. All headline metrics are NET
    of transaction costs; gross Sharpe is reported alongside so cost drag
    is visible, never hidden.

    Addresses: P4 — one canonical summary used by the runner, the notebook,
    and MLflow, so no ad-hoc metric recomputation can drift.
    """
    out: dict[str, float] = {
        "ann_return_net":    annualized_return(net_returns),
        "ann_return_gross":  annualized_return(gross_returns),
        "sharpe_net":        annualized_sharpe(net_returns, risk_free_annual),
        "sharpe_gross":      annualized_sharpe(gross_returns, risk_free_annual),
        "max_drawdown_net":  max_drawdown(net_returns),
        "calmar_net":        calmar_ratio(net_returns),
        "avg_turnover":      float(turnover.mean()) if len(turnover) else float("nan"),
        "total_cost_drag":   float((gross_returns - net_returns).sum()),
    }
    if benchmark_net is not None:
        out["information_ratio_net"] = information_ratio(net_returns, benchmark_net)
    if trial_sharpes is not None:
        out["dsr_net"] = deflated_sharpe_ratio(net_returns, trial_sharpes)
        out["n_trials"] = float(len(trial_sharpes))
    return out
