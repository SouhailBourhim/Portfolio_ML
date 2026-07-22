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

    N-accumulation policy (Phase 5, §17.1 gap now closed): the honest N is
    "the number of distinct configurations evaluated in the selection search
    that produced the reported strategy", NOT "all experiments ever" (which
    is ill-defined). `DSRTrialLedger` (below) persists every trial's
    per-period Sharpe across a search; feed `deflated_sharpe_ratio` the
    accumulated pool and N reflects the true breadth of the search (grid size
    × CV folds), deflating the winner correctly. Callers that pass only a
    single run's `trial_sharpes` still get the earlier within-run behavior —
    it is a lower bound on the honest deflation, not a different formula.

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


def block_bootstrap_sharpe_ci(
    returns: pd.Series,
    block_len: int = 21,
    n_boot: int = 1000,
    alpha: float = 0.10,
    risk_free_annual: float = 0.0,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Circular block-bootstrap confidence interval for the annualized Sharpe.

    Addresses: P4 — a point Sharpe hides its own sampling uncertainty; two
    strategies 0.15 apart may be statistically indistinguishable over a
    ~4-year window. This turns "1.12 vs 0.97" into "1.12 (90% CI …) vs
    0.97 (90% CI …)", the form an honest supervisor conversation needs.

    A CIRCULAR BLOCK bootstrap (not IID resampling) because daily returns are
    serially correlated (volatility clusters): resampling contiguous blocks
    of length `block_len` (≈ one trading month) preserves that dependence,
    where IID resampling would destroy it and understate the interval. The
    series is treated as a circle so every observation can start a block,
    avoiding end-effects.

    Deterministic under `seed` (repo convention — every stochastic estimator
    is seeded), so the reported interval is reproducible.

    Args:
        returns: Simple periodic (daily) OOS returns.
        block_len: Block length in periods (21 ≈ one month).
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided miss rate; 0.10 → a 90% CI (5th/95th percentiles).
        risk_free_annual: Excess-return adjustment, applied IDENTICALLY to the
            point and every bootstrap sample so the CI and the point use the
            same (excess, ddof=1) Sharpe convention `annualized_sharpe` uses.
        seed: RNG seed.

    Returns:
        (point_sharpe, lo, hi) — the sample annualized Sharpe and the
        (alpha/2, 1-alpha/2) percentile bounds. NaN triple if the series is
        too short OR has (near-)zero variance (Sharpe undefined), consistent
        with `annualized_sharpe`'s own degenerate-case return.
    """
    r = returns.dropna().to_numpy()
    t = len(r)
    # Zero-variance → Sharpe undefined; return NaN like annualized_sharpe does,
    # rather than a spurious 0.0 CI around an undefined point.
    if t < max(block_len, 3) or float(np.std(r, ddof=1)) < 1e-12:
        return (float("nan"), float("nan"), float("nan"))

    point = annualized_sharpe(pd.Series(r), risk_free_annual)
    # Same per-period excess and ddof=1 as annualized_sharpe, so the bootstrap
    # distribution is in the same units the point estimate lives in.
    rf_periodic = (1 + risk_free_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(t / block_len))
    boot_sharpes = np.empty(n_boot)

    for b in range(n_boot):
        starts = rng.integers(0, t, size=n_blocks)
        # Circular blocks: wrap indices modulo t so no end-effect bias.
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % t
        excess = r[idx[:t]] - rf_periodic
        sd = float(np.std(excess, ddof=1))
        boot_sharpes[b] = (
            excess.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 1e-12 else 0.0
        )

    lo = float(np.percentile(boot_sharpes, 100 * alpha / 2))
    hi = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


class DSRTrialLedger:
    """
    Persistent pool of per-period trial Sharpes, per universe, for honest DSR.

    Addresses: P4 — the N-accumulation policy (see `deflated_sharpe_ratio`'s
    docstring). The Deflated Sharpe Ratio only deflates correctly if N counts
    EVERY configuration the search evaluated, not just the handful compared in
    the final table. A hyperparameter sweep that quietly tried 200 configs and
    reported the best one is exactly the overfitting DSR exists to penalize;
    this ledger makes that count auditable and durable across a run.

    Not a database — a small JSON file (`data/gold/dsr_trial_ledger.json`)
    keyed by universe, holding the per-period (daily, NON-annualized) Sharpe of
    every trial. `record()` appends; `pool()` returns a universe's full list to
    hand to `deflated_sharpe_ratio`. Deliberately simple and inspectable, same
    spirit as the other Gold JSON artifacts.
    """

    def __init__(self, path=None) -> None:
        from pathlib import Path

        self.path = Path(path) if path is not None else None
        self._trials: dict[str, list[float]] = {}
        if self.path is not None and self.path.exists():
            import json

            self._trials = {k: list(v) for k, v in json.loads(self.path.read_text()).items()}

    @staticmethod
    def per_period_sharpe(returns: pd.Series) -> float:
        """Daily, non-annualized Sharpe — the unit `deflated_sharpe_ratio` expects."""
        r = returns.dropna()
        if len(r) < 2 or float(r.std()) < 1e-12:
            return 0.0
        return float(r.mean() / r.std())

    def record(self, universe: str, returns: pd.Series) -> None:
        """Append one trial's per-period Sharpe to `universe`'s pool."""
        self._trials.setdefault(universe, []).append(self.per_period_sharpe(returns))

    def pool(self, universe: str) -> list[float]:
        """Every trial Sharpe recorded for `universe` (empty list if none)."""
        return list(self._trials.get(universe, []))

    def n_trials(self, universe: str) -> int:
        return len(self._trials.get(universe, []))

    def save(self) -> None:
        """Persist to `self.path` (no-op if constructed without a path)."""
        if self.path is None:
            return
        import json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._trials, indent=2))
