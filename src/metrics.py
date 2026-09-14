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
from collections.abc import Mapping, Sequence

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


def certainty_equivalent(
    returns: pd.Series,
    risk_aversion: float = 1.0,
    risk_free_annual: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized certainty-equivalent return for a mean-variance investor.

    `CEQ = mu - (gamma / 2) * sigma^2` on EXCESS returns — the third criterion
    DeMiguel, Garlappi and Uppal (2009) report beside the Sharpe ratio and
    turnover. This project already cites that paper as its honesty hurdle but
    took only its Sharpe comparison. CEQ answers a different question: how much
    certain annual return an investor with risk aversion `gamma` would accept
    in place of the strategy. Being a difference rather than a ratio, it cannot
    be improved by shrinking a denominator.

    Addresses: P4 — `docs/EVALUATION_LIMITS.md` section 2 records that the
    Sharpe RANKING on `full_2021` is not invariant to the risk-free rate: at
    rf = 3.00% `equal_weight` overtakes `regime_conditional`. A ranking that
    flips under a nuisance parameter needs a second criterion computed the same
    way, not a more confident reading of the first.

    Moments are annualized ARITHMETICALLY here, deliberately, unlike
    `annualized_return`'s geometric compounding. CEQ is a mean-variance utility,
    so its mean and its variance must sit on the same footing; pairing a
    geometric mean with an arithmetic variance would not be the quantity
    DeMiguel et al. define. The gap between the two conventions is of order
    sigma^2 / 2 — the same order as the risk penalty itself, so the choice is
    not cosmetic.

    Args:
        returns: Simple periodic (daily) returns.
        risk_aversion: `gamma`. 1.0 is the paper's default. 0.0 is the
            risk-neutral case and returns the annualized mean excess unchanged.
        risk_free_annual: Annual risk-free rate, converted per-period
            geometrically so the excess matches `annualized_sharpe` exactly.
        periods: Periods per year for annualization.

    Returns:
        Annualized CEQ as a decimal. NaN for fewer than two observations,
        matching `annualized_sharpe`'s convention for an unusable sample.

    Raises:
        ValueError: if `risk_aversion` is negative. A negative coefficient
            would reward variance, inverting the utility rather than
            parameterising it — a caller bug, not a degenerate sample.
    """
    if risk_aversion < 0:
        raise ValueError(
            f"risk_aversion must be >= 0 (0 is risk-neutral); got {risk_aversion}"
        )
    if len(returns) < 2:
        return float("nan")
    rf_periodic = (1 + risk_free_annual) ** (1 / periods) - 1
    excess = returns - rf_periodic
    mu_annual = float(excess.mean()) * periods
    var_annual = float(excess.var(ddof=1)) * periods
    return mu_annual - 0.5 * risk_aversion * var_annual


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
    risk_aversion: float = 1.0,
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
        # The DeMiguel et al. (2009) criterion triple is Sharpe + CEQ +
        # turnover; `avg_turnover` was already here, CEQ completes it.
        "ceq_net":           certainty_equivalent(
            net_returns, risk_aversion, risk_free_annual
        ),
        "risk_aversion":     float(risk_aversion),
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
    Auditable record of every configuration a search evaluated, per universe.

    Addresses: P4 — the N-accumulation policy (see `deflated_sharpe_ratio`).
    DSR only deflates correctly if N counts EVERY configuration the search
    evaluated. A sweep that quietly tried 200 configs and reported the best is
    exactly the overfitting DSR exists to penalise.

    SCHEMA 2 — what changed and why. Schema 1 stored a bare list of per-period
    Sharpes. Two defects were found in the Phase 2 audit and both mattered:

      1. It recorded ONLY the portfolio-lever grid. The ML hyperparameter grid
         (6 RF + 9 XGB configurations per universe) never reached it, so the
         recorded N understated the real search by 15 per universe — biasing
         the deflation OPTIMISTICALLY, in the project's own favour.
      2. It stored a scalar per trial and discarded the return series. A
         multiple-testing correction of the White Reality Check / Hansen SPA
         family bootstraps over the candidates' RETURN SERIES; from scalars
         alone, no such correction is constructible. The ledger could not
         support the very claim it existed to license.

    Schema 2 stores, per trial: a human label, a `kind`, the parameters, the
    score in that kind's own units, and the return series where one exists.

    HETEROGENEITY IS EXPLICIT, NOT SMOOTHED OVER. The two kinds are not
    commensurable and are deliberately not merged:

      kind="ml_grid"  scored by INFORMATION COEFFICIENT on validation folds.
                      It has no portfolio return series, because a
                      hyperparameter configuration is not a portfolio.
      kind="lever"    scored by per-period Sharpe of a real walk-forward
                      backtest, and carries that return series.

    So `n_trials()` (the honest size of the search) and `pool()` (the Sharpe
    sample DSR's variance term needs) intentionally differ. Reporting one as
    the other is the mistake schema 1 made; `summary()` returns both.

    Backwards compatible: a schema-1 file still loads, and the legacy
    `record(universe, returns)` call still works.
    """

    SCHEMA_VERSION = 2

    def __init__(self, path=None) -> None:
        from pathlib import Path

        self.path = Path(path) if path is not None else None
        self._trials: dict[str, list[dict]] = {}
        if self.path is not None and self.path.exists():
            import json

            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("schema_version") == self.SCHEMA_VERSION:
                self._trials = {k: list(v) for k, v in raw.get("trials", {}).items()}
            else:
                # Schema 1: {universe: [sharpe, ...]} — lift into schema 2 so an
                # older artifact is readable rather than silently ignored.
                self._trials = {
                    universe: [
                        {"label": f"legacy_{i}", "kind": "lever", "params": {},
                         "per_period_sharpe": float(v), "returns": None}
                        for i, v in enumerate(values)
                    ]
                    for universe, values in raw.items()
                }

    @staticmethod
    def per_period_sharpe(returns: pd.Series) -> float:
        """Daily, non-annualized Sharpe — the unit `deflated_sharpe_ratio` expects."""
        r = returns.dropna()
        if len(r) < 2 or float(r.std()) < 1e-12:
            return 0.0
        return float(r.mean() / r.std())

    def record(
        self,
        universe: str,
        returns: pd.Series | None = None,
        *,
        label: str = "",
        kind: str = "lever",
        params: Mapping | None = None,
        score: float | None = None,
        keep_series: bool = True,
    ) -> None:
        """Append one evaluated configuration to `universe`'s search record.

        Args:
            universe: Ledger key.
            returns: The trial's net-return series, when it has one.
            label: Human-readable identifier, e.g. "rf__max_depth=3".
            kind: "lever" (Sharpe-scored, has a series) or "ml_grid"
                (IC-scored, has none). Any other value is accepted and simply
                excluded from the Sharpe pool.
            params: The configuration evaluated.
            score: The score in this kind's own units — IC for "ml_grid".
                Ignored for "lever", where the Sharpe is computed from
                `returns` so it cannot disagree with the series.
            keep_series: Store the return series. Only turned off by callers
                that would otherwise write a very large artifact.
        """
        entry: dict = {
            "label": label,
            "kind": kind,
            "params": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                       for k, v in dict(params or {}).items()},
        }
        if returns is not None and len(returns) > 0:
            entry["per_period_sharpe"] = self.per_period_sharpe(returns)
            if keep_series:
                clean = returns.dropna()
                # Dated in production; the unit tests use a positional index,
                # and a ledger that only accepts one of those would be brittle
                # for no benefit. Serialise whatever index it has.
                entry["returns"] = {
                    "dates": [
                        d.date().isoformat() if hasattr(d, "date") else str(d)
                        for d in clean.index
                    ],
                    "values": [round(float(v), 10) for v in clean.to_numpy()],
                }
            else:
                entry["returns"] = None
        else:
            entry["per_period_sharpe"] = None
            entry["returns"] = None
            entry["score"] = None if score is None else float(score)
        if score is not None and "score" not in entry:
            entry["score"] = float(score)
        self._trials.setdefault(universe, []).append(entry)

    def pool(self, universe: str) -> list[float]:
        """Per-period Sharpes of trials that HAVE one (the DSR variance sample).

        Deliberately not every trial: an IC-scored hyperparameter config has no
        Sharpe, and inventing one for it would corrupt the variance term.
        """
        return [
            t["per_period_sharpe"] for t in self._trials.get(universe, [])
            if t.get("per_period_sharpe") is not None
        ]

    def n_trials(self, universe: str) -> int:
        """Total configurations evaluated — the honest size of the search."""
        return len(self._trials.get(universe, []))

    def candidate_series(self, universe: str) -> dict[str, pd.Series]:
        """Label → return series, for trials that stored one.

        Addresses: P4 — this is what a White Reality Check / Hansen SPA would
        consume. It exists so that whether such a correction is CONSTRUCTIBLE
        is a question about the data, answerable by inspection, rather than an
        assumption.
        """
        out: dict[str, pd.Series] = {}
        for i, t in enumerate(self._trials.get(universe, [])):
            r = t.get("returns")
            if not r:
                continue
            try:
                # ISO8601 explicitly: a positional index serialises as "0",
                # "1", ... which dateutil would otherwise coerce with a warning.
                index = pd.DatetimeIndex(pd.to_datetime(r["dates"], format="ISO8601"))
            except (ValueError, TypeError):
                index = pd.Index(r["dates"])
            out[t.get("label") or f"trial_{i}"] = pd.Series(r["values"], index=index)
        return out

    def summary(self, universe: str) -> dict:
        """Counts by kind — so a reader can see what N is actually made of."""
        trials = self._trials.get(universe, [])
        by_kind: dict[str, int] = {}
        for t in trials:
            by_kind[t.get("kind", "unknown")] = by_kind.get(t.get("kind", "unknown"), 0) + 1
        return {
            "n_trials_total": len(trials),
            "n_by_kind": by_kind,
            "n_with_sharpe": len(self.pool(universe)),
            "n_with_return_series": len(self.candidate_series(universe)),
        }

    def save(self) -> None:
        """Persist to `self.path` (no-op if constructed without a path)."""
        if self.path is None:
            return
        import json

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"schema_version": self.SCHEMA_VERSION, "trials": self._trials}, indent=2
        ), encoding="utf-8")


def paired_block_bootstrap(
    candidate: pd.Series,
    benchmark: pd.Series,
    candidate_turnover: pd.Series | None = None,
    benchmark_turnover: pd.Series | None = None,
    candidate_cost: pd.Series | None = None,
    benchmark_cost: pd.Series | None = None,
    block_len: int = 21,
    n_boot: int = 2000,
    alpha: float = 0.10,
    risk_free_annual: float = 0.0,
    seed: int = 0,
) -> dict:
    """Paired moving-block bootstrap of the DIFFERENCE between two strategies.

    Addresses: P4 — this is the instrument the project was missing. Marginal
    per-strategy confidence intervals answer "how uncertain is this Sharpe",
    which is NOT the question "do these two strategies differ". Two intervals
    can overlap substantially while the paired difference is consistently
    positive, because the strategies share the same market days and their
    errors are strongly correlated; and non-overlap does not establish a
    difference either. Every "indistinguishable"/"superior" claim previously
    made from overlapping marginal CIs was unlicensed in one direction or the
    other. This function tests the difference directly.

    PAIRED is the operative word: both series are resampled with the SAME
    block indices, so each draw keeps the two strategies on the same calendar
    days. That preserves the serial dependence within a strategy AND the
    same-day cross-correlation between them — the variance reduction that
    makes a paired test more powerful than comparing two marginal intervals.

    NULL-CENTRED p-value. The p-value is NOT the raw fraction of draws below
    zero: that is a descriptive statement about the observed difference, not a
    tail probability under a null. Following the standard bootstrap
    hypothesis-test construction, the resampled differences are recentred on
    zero to simulate the null of no outperformance, and the p-value is the
    share of that null distribution at or beyond the OBSERVED difference
    (one-sided, H1: candidate > benchmark). `prob_sharpe_diff_positive` is
    reported separately and labelled, because it is a useful number that must
    not be mistaken for a p-value.

    Args:
        candidate, benchmark: Daily NET return series on identical dates.
        candidate_turnover, benchmark_turnover: Optional per-rebalance
            turnover, for the economic-impact fields.
        candidate_cost, benchmark_cost: Optional per-rebalance cost fractions.
        block_len: Block length in days (~one trading month by default).
        n_boot, alpha, seed: Resample count, CI level, RNG seed.

    Returns:
        Observed differences, the paired CI, the null p-value, prob_positive,
        and the turnover/cost deltas.

    Raises:
        ValueError: if the indexes are not identical, if either series holds
            NaN, or if there is too little data for one block. Aligning
            silently would change WHICH days are compared, so a mismatch is a
            caller error rather than something to repair here.
    """
    if not candidate.index.equals(benchmark.index):
        only_c = candidate.index.difference(benchmark.index)
        only_b = benchmark.index.difference(candidate.index)
        raise ValueError(
            "paired_block_bootstrap requires identical date indexes; refusing to "
            f"align silently. {len(only_c)} date(s) only in candidate, "
            f"{len(only_b)} only in benchmark. Slice both to the same test window "
            "before comparing."
        )
    if candidate.isna().any() or benchmark.isna().any():
        raise ValueError("paired_block_bootstrap requires NaN-free return series.")

    n = len(candidate)
    if n < 2 or n < block_len:
        raise ValueError(
            f"Not enough observations for a paired block bootstrap: n={n}, "
            f"block_len={block_len}. Need at least one full block."
        )

    c = candidate.to_numpy(dtype=float)
    b = benchmark.to_numpy(dtype=float)

    def _stats(ci: np.ndarray, bi: np.ndarray) -> tuple[float, float]:
        """(annualized return difference, annualized Sharpe difference)."""
        ann_c = float(np.mean(ci)) * TRADING_DAYS_PER_YEAR
        ann_b = float(np.mean(bi)) * TRADING_DAYS_PER_YEAR
        sc, sb = float(np.std(ci, ddof=1)), float(np.std(bi, ddof=1))
        sharpe_c = ((ann_c - risk_free_annual) / (sc * np.sqrt(TRADING_DAYS_PER_YEAR))
                    if sc > 1e-15 else 0.0)
        sharpe_b = ((ann_b - risk_free_annual) / (sb * np.sqrt(TRADING_DAYS_PER_YEAR))
                    if sb > 1e-15 else 0.0)
        return ann_c - ann_b, sharpe_c - sharpe_b

    obs_ret_diff, obs_sharpe_diff = _stats(c, b)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    boot_ret = np.empty(n_boot)
    boot_sharpe = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
        boot_ret[i], boot_sharpe[i] = _stats(c[idx], b[idx])   # SAME idx = paired

    lo_q, hi_q = 100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)

    def _null_p(boot: np.ndarray, observed: float) -> float:
        """One-sided p under H0: no outperformance, via a recentred null."""
        null = boot - float(np.mean(boot))
        # +1 smoothing: a bootstrap p-value should never be exactly 0, which
        # would assert a certainty the resample count cannot support.
        return float((np.sum(null >= observed) + 1) / (len(null) + 1))

    out = {
        "n_observations": int(n),
        "test_start": str(candidate.index.min().date()),
        "test_end": str(candidate.index.max().date()),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "ann_return_diff": round(obs_ret_diff, 6),
        "sharpe_diff": round(obs_sharpe_diff, 6),
        "sharpe_diff_ci": [
            round(float(np.percentile(boot_sharpe, lo_q)), 6),
            round(float(np.percentile(boot_sharpe, hi_q)), 6),
        ],
        "ann_return_diff_ci": [
            round(float(np.percentile(boot_ret, lo_q)), 6),
            round(float(np.percentile(boot_ret, hi_q)), 6),
        ],
        "p_value_no_outperformance": round(_null_p(boot_sharpe, obs_sharpe_diff), 6),
        "prob_sharpe_diff_positive": round(float(np.mean(boot_sharpe > 0.0)), 6),
        "ci_alpha": alpha,
    }

    def _mean_delta(a, bb):
        if a is None or bb is None or len(a) == 0 or len(bb) == 0:
            return None
        return round(float(a.mean()) - float(bb.mean()), 6)

    out["avg_turnover_diff"] = _mean_delta(candidate_turnover, benchmark_turnover)
    out["avg_cost_diff"] = _mean_delta(candidate_cost, benchmark_cost)
    return out


# ── Multiple-testing correction over a whole searched candidate set ──────────
#
# Addresses: P4 — the last stated statistical limitation of this project. Every
# comparison reported so far tests ONE candidate against a benchmark. But the
# candidate was CHOSEN from a search, and the best of N candidates beats a
# benchmark by chance more often than any single candidate does. Without a
# correction, "the selected model beat the hurdle" is not the claim it appears
# to be.
#
# White's Reality Check (2000) and Hansen's SPA (2005) test the composite null
# "NO candidate in the set outperforms the benchmark", accounting for both the
# number of candidates and their cross-correlation — which matters enormously
# here, because 240 configurations that share a data window and differ by one
# hyperparameter are nearly the same strategy 240 times, not 240 independent
# bets.

def _circular_block_indices(
    n: int, block_len: int, rng: np.random.Generator
) -> np.ndarray:
    """One circular-block resample of positions 0..n-1.

    Circular rather than plain moving blocks so every observation has equal
    probability of selection; without the wrap, the first and last block_len-1
    points are systematically under-sampled.
    """
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_len)
    return ((starts[:, None] + offsets[None, :]).ravel() % n)[:n]


def _sharpe_from_array(values: np.ndarray, risk_free_annual: float) -> float:
    std = float(np.std(values, ddof=1))
    if std <= 1e-15:
        return 0.0
    ann_return = float(np.mean(values)) * TRADING_DAYS_PER_YEAR
    return (ann_return - risk_free_annual) / (std * np.sqrt(TRADING_DAYS_PER_YEAR))


def reality_check(
    candidate_returns: Mapping[str, pd.Series],
    benchmark: pd.Series,
    *,
    statistic: str = "mean_return",
    block_len: int = 21,
    n_boot: int = 2000,
    seed: int = 0,
    risk_free_annual: float = 0.0,
) -> dict:
    """White's Reality Check and Hansen's SPA over an entire candidate set.

    Addresses: P4 — tests the composite null that NO candidate outperforms the
    benchmark, rather than testing one pre-chosen candidate as if it had not
    been selected from a search.

    Every candidate is resampled with the SAME block draws as the benchmark and
    as each other. That is not an optimisation: the candidates here differ by a
    single hyperparameter on a shared data window, so they are heavily
    cross-correlated, and resampling them independently would treat 240
    near-identical strategies as 240 independent bets and badly overstate the
    effective breadth of the search.

    Two statistics are offered because the project reports both:

    * ``mean_return`` — the textbook White (2000) formulation, on the per-period
      return differential. The performance measure is a mean, which is what the
      recentering argument assumes.
    * ``sharpe`` — the differential in annualized Sharpe, matching this
      project's headline metric. Defensible and widely used, but the statistic
      is a ratio of moments rather than a mean, so the asymptotic argument is
      weaker than for the first. Reported alongside, never alone.

    Hansen's SPA differs from RC in two ways, both of which make it less
    conservative: it studentizes each candidate by its own bootstrap standard
    error, so a high-variance candidate cannot dominate the max statistic on
    noise alone; and it drops hopelessly poor candidates from the null
    recentering, so padding the search with bad configurations can no longer
    make a good one look significant by lowering the bar.

    Args:
        candidate_returns: Net-return series per candidate, all on the SAME
            index as the benchmark.
        benchmark: The strategy every candidate is measured against.
        statistic: ``"mean_return"`` or ``"sharpe"``.
        block_len, n_boot, seed: Bootstrap configuration.
        risk_free_annual: Used only by the Sharpe statistic.

    Returns:
        Observed best candidate and its differential, the RC p-value, the SPA
        p-value, and the inputs needed to reproduce both.

    Raises:
        ValueError: on an empty set, misaligned indexes, NaN, or too little
            data for one block. Aligning silently would change which days are
            compared.
    """
    if statistic not in ("mean_return", "sharpe"):
        raise ValueError(
            f"Unknown statistic {statistic!r}; expected 'mean_return' or 'sharpe'."
        )
    if not candidate_returns:
        raise ValueError(
            "reality_check needs at least one candidate. An empty set would "
            "return a vacuous p-value of 1.0 that reads like a finding."
        )

    names = sorted(candidate_returns)
    for name in names:
        series = candidate_returns[name]
        if not series.index.equals(benchmark.index):
            raise ValueError(
                f"Candidate {name!r} is not on the benchmark's index; refusing to "
                f"align silently. Every candidate must be evaluated on the SAME "
                f"frozen test dates or the max statistic compares different periods."
            )
        if series.isna().any():
            raise ValueError(f"Candidate {name!r} contains NaN.")
    if benchmark.isna().any():
        raise ValueError("benchmark contains NaN.")

    n = len(benchmark)
    if n < 2 or n < block_len:
        raise ValueError(
            f"Not enough observations: n={n}, block_len={block_len}."
        )

    matrix = np.column_stack([candidate_returns[name].to_numpy(dtype=float) for name in names])
    bench = benchmark.to_numpy(dtype=float)
    n_candidates = len(names)

    def differentials(rows: np.ndarray) -> np.ndarray:
        """Per-candidate performance differential vs the benchmark."""
        sub, sub_bench = matrix[rows], bench[rows]
        if statistic == "mean_return":
            return sub.mean(axis=0) - sub_bench.mean()
        bench_sharpe = _sharpe_from_array(sub_bench, risk_free_annual)
        return np.array([
            _sharpe_from_array(sub[:, k], risk_free_annual) - bench_sharpe
            for k in range(n_candidates)
        ])

    all_rows = np.arange(n)
    observed = differentials(all_rows)
    scale = np.sqrt(n)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, n_candidates), dtype=float)
    for b in range(n_boot):
        boot[b] = differentials(_circular_block_indices(n, block_len, rng))

    # ---- White's Reality Check -------------------------------------------
    # V = max_k sqrt(T)*f_k ; null draws recenter each candidate on its own
    # observed value, which imposes "no candidate outperforms" on every one.
    observed_v = float(np.max(scale * observed))
    null_v = np.max(scale * (boot - observed[None, :]), axis=1)
    rc_p = float((np.sum(null_v >= observed_v) + 1) / (n_boot + 1))

    # ---- Hansen's SPA (consistent variant) --------------------------------
    omega = np.std(scale * boot, axis=0, ddof=1)
    # A zero-variance differential is a DETERMINISTIC edge (or deficit), not an
    # unusable candidate. An earlier version mapped it to omega = inf, which
    # inverted the meaning twice over: a candidate beating the benchmark by a
    # constant got a t-statistic of zero and could never be detected, while one
    # LOSING by a constant was retained in the recentring instead of dropped.
    # The positive-control test caught it. Flooring keeps the sign and the
    # magnitude, so a risk-free edge scores as the extreme case it is.
    omega = np.maximum(omega, 1e-12)
    observed_t = float(np.max(scale * observed / omega))

    # Drop candidates that are so poor they cannot plausibly be the best. The
    # threshold is Hansen's -sqrt(2 log log T) rule; without it, adding bad
    # configurations to the search lowers the bar for the good ones.
    threshold = -np.sqrt(2.0 * np.log(max(np.log(n), 1.0001))) * omega / scale
    retained = observed >= threshold
    recentre = np.where(retained, observed, 0.0)
    null_t = np.max(scale * (boot - recentre[None, :]) / omega[None, :], axis=1)
    spa_p = float((np.sum(null_t >= observed_t) + 1) / (n_boot + 1))

    best = int(np.argmax(observed))
    return {
        "statistic": statistic,
        "n_candidates": n_candidates,
        "n_observations": int(n),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "best_candidate": names[best],
        "best_differential": float(observed[best]),
        "reality_check_p_value": rc_p,
        "spa_p_value": spa_p,
        "spa_candidates_retained": int(np.sum(retained)),
        "n_candidates_beating_benchmark": int(np.sum(observed > 0)),
        "interpretation": (
            f"No evidence that ANY of the {n_candidates} searched candidates "
            f"outperforms the benchmark (RC p = {rc_p:.3f}, SPA p = {spa_p:.3f})."
            if min(rc_p, spa_p) >= 0.05 else
            f"The best of {n_candidates} candidates outperforms the benchmark after "
            f"correcting for the search (RC p = {rc_p:.3f}, SPA p = {spa_p:.3f})."
        ),
    }


def model_confidence_set(
    candidate_returns: Mapping[str, pd.Series],
    *,
    size: float = 0.10,
    block_len: int = 21,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Model Confidence Set of Hansen, Lunde & Nason (2011), Econometrica 79(2).

    Addresses: P4 — `reality_check` can only ever fail to reject. White's RC and
    Hansen's SPA are one-sided tests against a single pre-specified benchmark,
    so when the search turns up nothing they return silence: "no candidate is
    established as superior", and no more. The MCS asks the complementary
    question and returns a SET — the strategies not eliminated at confidence
    1 - `size` — which carries content whichever way it falls. A surviving set
    containing `equal_weight` quantifies DeMiguel et al. (2009) on this
    project's own data; a set that eliminates the ML challengers is evidence
    they are worse rather than merely unproven.

    WORDING (AGENTS.md section 5.2). Survival is not equivalence, and members of
    the returned set must never be called "statistically indistinguishable" —
    that phrase is banned project-wide precisely because failing to eliminate
    is not a demonstration of sameness. Say "retained by the MCS procedure at
    size alpha". `interpretation` below is phrased to comply and should be
    quoted rather than paraphrased.

    STATISTIC. The procedure consumes per-period LOSSES, taken here as negated
    net returns, so this is a mean-return criterion. It is NOT a Sharpe
    criterion: a Sharpe ratio is a ratio of moments with no per-period loss
    representation, so unlike `reality_check` there is no `statistic` argument.
    An MCS reported beside a Sharpe-based RC/SPA compares different quantities,
    and that must be stated wherever the two appear together.

    Implementation note: delegates to `arch.bootstrap.MCS` rather than
    hand-rolling the elimination, but only after verifying it. Its neighbour
    `arch.bootstrap.SPA` accepts a `studentize` flag it never applies — the
    flag only writes a metadata string — so the library is not taken on trust.
    MCS was checked to retain a genuinely best model, eliminate hopeless ones,
    retain everything under exchangeability, and shrink monotonically in `size`.

    Args:
        candidate_returns: Per-strategy simple net return series sharing one
            index. At least two — a confidence set over one model is vacuous.
        size: Test size alpha; the set has confidence 1 - alpha. Default 0.10,
            matching `phase5.bootstrap.alpha`.
        block_len: Circular block length, matching the project's other
            bootstraps (~one trading month).
        n_boot: Bootstrap replications.
        seed: Seeded for reproducibility, like every estimator here.

    Returns:
        dict with `included` / `excluded` (sorted name lists), `n_candidates`,
        `n_included`, `size`, `block_len`, `n_boot`, `seed`, `statistic` and
        `interpretation`.

    Raises:
        ValueError: on fewer than two candidates, misaligned indexes, NaN, or
            too little data for one block — the same input discipline as
            `reality_check`, and for the same reason: silent alignment would
            change which periods are being compared.
    """
    from arch.bootstrap import MCS as _ArchMCS

    names = sorted(candidate_returns)
    if len(names) < 2:
        raise ValueError(
            f"model_confidence_set needs at least two candidates; got {len(names)}."
        )

    reference = candidate_returns[names[0]].index
    for name in names:
        series = candidate_returns[name]
        if not series.index.equals(reference):
            raise ValueError(
                f"model_confidence_set requires identical date indexes; '{name}' "
                f"differs from '{names[0]}' — refusing to align silently."
            )
        if series.isna().any():
            raise ValueError(
                f"model_confidence_set requires NaN-free returns; '{name}' has NaN."
            )

    n = len(reference)
    if n < block_len:
        raise ValueError(
            f"Not enough observations for a block bootstrap: n={n}, block_len={block_len}."
        )

    # Losses, not returns: the MCS eliminates models with HIGHER loss.
    losses = pd.DataFrame(
        {name: -candidate_returns[name].to_numpy(dtype=float) for name in names},
        index=reference,
    )

    mcs = _ArchMCS(
        losses,
        size=size,
        reps=n_boot,
        block_size=block_len,
        bootstrap="circular",
        seed=seed,
    )
    mcs.compute()

    included = sorted(str(x) for x in mcs.included)
    excluded = sorted(str(x) for x in mcs.excluded)

    return {
        "included": included,
        "excluded": excluded,
        "n_candidates": len(names),
        "n_included": len(included),
        "size": float(size),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "statistic": "mean_return",
        "interpretation": (
            f"{len(included)} of {len(names)} strategies are retained by the MCS "
            f"procedure at size {size:g} (mean-return losses). Retention is not "
            f"evidence that the retained strategies perform equally; it means the "
            f"data do not support eliminating them."
        ),
    }


def stepm_superior_models(
    candidate_returns: Mapping[str, pd.Series],
    benchmark: pd.Series,
    *,
    size: float = 0.05,
    block_len: int = 21,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Stepwise multiple testing (Romano & Wolf 2005), Econometrica 73(4).

    Addresses: P4 — `reality_check` collapses a 240-configuration search into
    ONE global p-value: "is the best of them better than the benchmark". When
    that fails to reject, it cannot say which candidates were close, and when
    it does reject it names only the single best. StepM controls the same
    familywise error rate but proceeds in stages — reject the clear cases,
    remove them, re-derive the critical value on what remains, repeat — so it
    returns a PER-STRATEGY verdict and is never less powerful than the
    single-step procedure at the same FWER.

    What it adds over `model_confidence_set`: the MCS asks which strategies
    survive mutual comparison, with no privileged benchmark. StepM keeps the
    project's pre-specified benchmark and asks which candidates beat IT. Both
    are one-sided in the project's favour-free direction; they answer different
    questions and are reported side by side, not as substitutes.

    WORDING (AGENTS.md section 5.2). A named model is one for which the null of
    no outperformance is rejected at FWER `size`. That is a rejection, not a
    demonstration of superiority in general — and an empty list is not evidence
    of no difference. `interpretation` is phrased accordingly.

    STATISTIC. Like the MCS, this consumes per-period losses (negated net
    returns), so it is a MEAN-RETURN criterion and takes no `statistic`
    argument. `reality_check` can also run on Sharpe; this cannot, and mixing
    the two in one table compares different quantities.

    Implementation note: delegates to `arch.bootstrap.StepM`, verified first
    rather than trusted — it wraps `arch.bootstrap.SPA`, whose `studentize`
    flag is accepted and never applied. StepM was checked to name exactly the
    genuinely superior models in a positive control, name none under
    exchangeability, be no less informative than the single-step Reality Check
    on the same data, and be monotone in `size`.

    Args:
        candidate_returns: Per-candidate simple net return series.
        benchmark: The pre-specified comparator, same index as every candidate.
        size: Familywise error rate. Default 0.05.
        block_len: Circular block length (~one trading month).
        n_boot: Bootstrap replications.
        seed: Seeded for reproducibility.

    Returns:
        dict with `superior_models` (sorted names whose null was rejected),
        `n_superior`, `n_candidates`, `size`, `block_len`, `n_boot`, `seed`,
        `statistic` and `interpretation`.

    Raises:
        ValueError: on an empty candidate set, misaligned indexes, NaN, or too
            little data for one block — the input discipline of
            `reality_check`, for the same reason.
    """
    from arch.bootstrap import StepM as _ArchStepM

    names = sorted(candidate_returns)
    if not names:
        raise ValueError("stepm_superior_models requires at least one candidate.")

    for name in names:
        series = candidate_returns[name]
        if not series.index.equals(benchmark.index):
            raise ValueError(
                f"stepm_superior_models requires identical date indexes; '{name}' "
                f"differs from the benchmark — refusing to align silently."
            )
        if series.isna().any():
            raise ValueError(
                f"stepm_superior_models requires NaN-free returns; '{name}' has NaN."
            )
    if benchmark.isna().any():
        raise ValueError("stepm_superior_models requires a NaN-free benchmark.")

    n = len(benchmark)
    if n < block_len:
        raise ValueError(
            f"Not enough observations for a block bootstrap: n={n}, block_len={block_len}."
        )

    # Losses, not returns: a model is "superior" when its loss is lower.
    models = pd.DataFrame(
        {name: -candidate_returns[name].to_numpy(dtype=float) for name in names},
        index=benchmark.index,
    )
    bench_loss = pd.Series(-benchmark.to_numpy(dtype=float), index=benchmark.index)

    step = _ArchStepM(
        bench_loss,
        models,
        size=size,
        block_size=block_len,
        reps=n_boot,
        bootstrap="circular",
        seed=seed,
    )
    step.compute()
    superior = sorted(str(x) for x in step.superior_models)

    return {
        "superior_models": superior,
        "n_superior": len(superior),
        "n_candidates": len(names),
        "size": float(size),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "statistic": "mean_return",
        "interpretation": (
            f"StepM rejects the null of no outperformance for {len(superior)} of "
            f"{len(names)} candidates at familywise error rate {size:g} "
            f"(mean-return losses)."
            if superior else
            f"StepM rejects the null of no outperformance for none of the "
            f"{len(names)} candidates at familywise error rate {size:g} "
            f"(mean-return losses). This does not establish that no candidate "
            f"differs from the benchmark."
        ),
    }


def _newey_west_long_run_cov(centered: np.ndarray, bandwidth: int) -> np.ndarray:
    """Bartlett-kernel HAC estimate of the long-run covariance of a k-vector.

    `centered` is (T, k) with column means already removed. Returns (k, k).
    """
    n_obs = centered.shape[0]
    psi = centered.T @ centered / n_obs
    for lag in range(1, bandwidth + 1):
        gamma = centered[lag:].T @ centered[:-lag] / n_obs
        weight = 1.0 - lag / (bandwidth + 1.0)
        psi = psi + weight * (gamma + gamma.T)
    return psi


def _sharpe_difference_point(r_a: np.ndarray, r_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Per-period Sharpe difference and the delta-method gradient at the estimate.

    Parameterised by the four moments (mu_a, mu_b, gamma_a, gamma_b) where
    gamma is the SECOND RAW moment, following Ledoit & Wolf (2008). With
    sigma^2 = gamma - mu^2 and SR = mu / sigma, the partials are
    d SR / d mu = gamma / sigma^3 and d SR / d gamma = -mu / (2 sigma^3).
    """
    mu_a, mu_b = float(r_a.mean()), float(r_b.mean())
    g_a, g_b = float((r_a**2).mean()), float((r_b**2).mean())
    var_a, var_b = g_a - mu_a**2, g_b - mu_b**2
    if var_a <= 0 or var_b <= 0:
        return float("nan"), np.full(4, np.nan)
    sd_a, sd_b = var_a**1.5, var_b**1.5
    diff = mu_a / np.sqrt(var_a) - mu_b / np.sqrt(var_b)
    gradient = np.array(
        [g_a / sd_a, -g_b / sd_b, -mu_a / (2 * sd_a), mu_b / (2 * sd_b)], dtype=float
    )
    return float(diff), gradient


def sharpe_difference_test(
    candidate: pd.Series,
    benchmark: pd.Series,
    *,
    block_len: int = 21,
    n_boot: int = 2000,
    alpha: float = 0.10,
    risk_free_annual: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
    hac_bandwidth: int | None = None,
    seed: int = 0,
) -> dict:
    """
    Studentized test for a DIFFERENCE of Sharpe ratios (Ledoit & Wolf 2008).

    Addresses: P4 — `paired_block_bootstrap` reports a percentile interval for
    the Sharpe difference. A percentile bootstrap of a non-pivotal statistic is
    not asymptotically refined, and its coverage degrades under exactly the two
    conditions this data has: heavy tails and serial dependence. Ledoit and Wolf
    derive the HAC standard error of the difference by the delta method over the
    four moments (mu_a, mu_b, gamma_a, gamma_b) and use it to STUDENTIZE a
    circular block bootstrap. The studentized statistic is asymptotically
    pivotal, so the bootstrap attains a higher order of accuracy.

    Why this matters here rather than in general: the frozen test on
    `full_2021` is ~455 days and the published intervals are wide enough that
    calibration, not point estimation, is the binding constraint. A
    better-calibrated test on identical data is the cheapest possible
    improvement to the evidence chain.

    The same standard error yields the MINIMUM DETECTABLE DIFFERENCE —
    see `sharpe_difference_mde`. `docs/EVALUATION_LIMITS.md` section 5 records
    that a pre-registered MDE "would have framed the negative result as
    DESIGNED rather than as a disappointment"; this supplies it, and it can be
    computed retrospectively for every comparison already published.

    WORDING (AGENTS.md section 5.2): a point estimate is an OBSERVED
    DIFFERENCE, an interval is UNCERTAINTY QUANTIFICATION, and failing to
    reject establishes nothing about equality.

    SHARPE CONVENTION, and it is a real difference worth stating. The delta
    method is derived on the raw-moment parameterisation, where
    sigma^2 = gamma - mu^2 — a population (ddof=0) variance. `annualized_sharpe`
    elsewhere in this module uses pandas' default ddof=1. The two cannot both
    hold exactly, so this function is internally consistent on the Ledoit-Wolf
    basis: `difference` IS `sharpe_candidate - sharpe_benchmark` exactly, and
    all three use ddof=0. The gap against `annualized_sharpe` is O(1/n) —
    around 7e-4 relative at n = 750 — and is a convention, not a disagreement.
    Reporting a difference that did not equal the difference of the two
    reported Sharpes would be the worse trade: a reader can reconcile a stated
    convention, but not an unexplained arithmetic mismatch.

    Args:
        candidate, benchmark: Simple periodic net returns, identical indexes.
        block_len: Circular block length (~one trading month).
        n_boot: Bootstrap replications.
        alpha: Two-sided level; the interval has coverage 1 - alpha.
        risk_free_annual: Converted per-period geometrically and subtracted
            from both series, matching `annualized_sharpe` exactly.
        periods: Periods per year. Sharpe, the difference and the standard
            error are all reported ANNUALIZED; the t-statistic is invariant to
            that rescaling.
        hac_bandwidth: Bartlett-kernel lag truncation. Defaults to the standard
            Newey-West rule floor(4 (T/100)^(2/9)). Ledoit and Wolf prefer a
            prewhitened QS kernel; Bartlett is the documented simplification.
        seed: Seeded, like every stochastic estimator here.

    Returns:
        dict with annualized `sharpe_candidate`, `sharpe_benchmark`,
        `difference`, `standard_error`, plus `t_statistic`, `p_value_hac`
        (normal approximation), `p_value_studentized_bootstrap`,
        `confidence_interval`, `mde_at_power_80`, and the settings used.

    Raises:
        ValueError: on misaligned indexes, NaN, or too little data for one
            block — the input discipline of `paired_block_bootstrap`.
    """
    if not candidate.index.equals(benchmark.index):
        raise ValueError(
            "sharpe_difference_test requires identical date indexes; "
            "refusing to align silently."
        )
    if candidate.isna().any() or benchmark.isna().any():
        raise ValueError("sharpe_difference_test requires NaN-free return series.")

    n = len(candidate)
    if n < 2 or n < block_len:
        raise ValueError(
            f"Not enough observations: n={n}, block_len={block_len}."
        )

    rf_periodic = (1 + risk_free_annual) ** (1 / periods) - 1
    r_a = candidate.to_numpy(dtype=float) - rf_periodic
    r_b = benchmark.to_numpy(dtype=float) - rf_periodic

    if hac_bandwidth is None:
        hac_bandwidth = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    hac_bandwidth = max(0, min(hac_bandwidth, n - 2))

    def point_and_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        diff, grad = _sharpe_difference_point(a, b)
        if not np.isfinite(diff):
            return float("nan"), float("nan")
        moments = np.column_stack([a, b, a**2, b**2])
        psi = _newey_west_long_run_cov(moments - moments.mean(axis=0), hac_bandwidth)
        variance = float(grad @ psi @ grad) / len(a)
        return diff, float(np.sqrt(variance)) if variance > 0 else float("nan")

    diff_obs, se_obs = point_and_se(r_a, r_b)
    if not np.isfinite(diff_obs) or not np.isfinite(se_obs) or se_obs == 0.0:
        raise ValueError(
            "sharpe_difference_test: degenerate sample — zero variance in a "
            "series, or a non-finite standard error."
        )
    t_obs = diff_obs / se_obs

    rng = np.random.default_rng(seed)
    studentized = np.empty(n_boot)
    for b in range(n_boot):
        rows = _circular_block_indices(n, block_len, rng)
        diff_b, se_b = point_and_se(r_a[rows], r_b[rows])
        studentized[b] = (diff_b - diff_obs) / se_b if np.isfinite(se_b) and se_b > 0 else np.nan
    studentized = studentized[np.isfinite(studentized)]

    p_boot = float((np.sum(np.abs(studentized) >= abs(t_obs)) + 1) / (len(studentized) + 1))
    lo_q, hi_q = np.quantile(studentized, [alpha / 2, 1 - alpha / 2])

    scale = np.sqrt(periods)
    diff_ann, se_ann = diff_obs * scale, se_obs * scale

    return {
        "n_observations": n,
        # ddof=0, NOT ddof=1 — see the "Sharpe convention" note in the docstring.
        "sharpe_candidate": float(np.mean(r_a) / np.std(r_a, ddof=0) * scale),
        "sharpe_benchmark": float(np.mean(r_b) / np.std(r_b, ddof=0) * scale),
        "difference": diff_ann,
        "standard_error": se_ann,
        "t_statistic": float(t_obs),
        "p_value_hac": float(2 * (1 - norm.cdf(abs(t_obs)))),
        "p_value_studentized_bootstrap": p_boot,
        "confidence_interval": (
            float(diff_ann - hi_q * se_ann),
            float(diff_ann - lo_q * se_ann),
        ),
        "mde_at_power_80": sharpe_difference_mde(se_ann, alpha=alpha, power=0.80),
        "ci_alpha": float(alpha),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "hac_bandwidth": int(hac_bandwidth),
        "seed": int(seed),
    }


def sharpe_difference_mde(
    standard_error: float,
    *,
    alpha: float = 0.10,
    power: float = 0.80,
    n_observed: int | None = None,
    n_target: int | None = None,
) -> float:
    """
    Smallest Sharpe difference detectable at `power`, given a standard error.

    Addresses: P4 — `docs/EVALUATION_LIMITS.md` section 5 states that intervals
    containing zero "were the predictable consequence of the design, not a
    discovery about the models", and that a minimum-detectable-effect stated in
    advance "would have framed the negative result as DESIGNED rather than as a
    disappointment". This is that quantity: `(z_{1-alpha/2} + z_{power}) * SE`.

    Report it BEFORE running a comparison, not after. Its value is that it
    distinguishes "the effect is absent" from "this design could never have
    seen it", which a p-value alone cannot.

    Pass `n_observed` and `n_target` to project the standard error onto a
    different sample length under the usual root-n scaling — the honest way to
    answer "how much more data would this question need?".

    Args:
        standard_error: SE of the Sharpe difference, from
            `sharpe_difference_test`. Annualized in, annualized out.
        alpha: Two-sided significance level.
        power: Desired power, conventionally 0.80.
        n_observed, n_target: Optional; supply BOTH to rescale.

    Returns:
        The minimum detectable difference, in the units of `standard_error`.

    Raises:
        ValueError: if only one of `n_observed` / `n_target` is given, or
            either is not positive, or alpha/power are outside (0, 1).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1); got {power}")
    if (n_observed is None) != (n_target is None):
        raise ValueError(
            "supply both n_observed and n_target to rescale, or neither."
        )

    se = float(standard_error)
    if n_observed is not None:
        if n_observed <= 0 or n_target <= 0:
            raise ValueError("n_observed and n_target must be positive.")
        se = se * np.sqrt(n_observed / n_target)

    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def probability_of_backtest_overfitting(
    trial_returns: Mapping[str, pd.Series],
    *,
    n_splits: int = 16,
) -> dict:
    """
    Probability of backtest overfitting via CSCV (Bailey, Borwein, Lopez de
    Prado & Zhu, 2017), Journal of Computational Finance 20(4).

    Addresses: P4 — `deflated_sharpe_ratio` is the other half of this research
    programme and fails differently, which is the point of having both. DSR is
    PARAMETRIC: it needs a trial count and a variance of trial Sharpes, and is
    only as good as those two inputs. CSCV is model-free, non-parametric and
    symmetric. It splits the trial-by-time performance matrix into S blocks,
    forms every balanced in-sample / out-of-sample partition, and asks how
    often the configuration chosen in-sample lands below the OOS median. That
    frequency is the PBO.

    The question it answers is the one this project keeps asking: is the best
    observed candidate a real edge or the luckiest of a wide search? Unlike the
    Reality Check it needs no benchmark, and unlike the DSR it makes no
    distributional assumption about the trial pool.

    CAUTION, and it must be reported with the number: PBO measures the
    SELECTION PROCEDURE, not the strategies alone. It rises toward 1 when
    in-sample advantage is sample-specific — the realistic case for a
    hyperparameter grid fitted repeatedly to one dataset — so a high value over
    a wide grid is not by itself evidence that every candidate is worthless.

    The converse is worth stating too, because it is easy to assume otherwise:
    breadth alone does not inflate PBO. Measured here on INDEPENDENT noise
    series, PBO stays near 0.5 whether the search spans 4 configurations or
    120, because the in-sample winner's out-of-sample rank is then uniform.
    What drives PBO up is dependence between candidates and the selection, not
    the count. Reading a high PBO as "we simply tried too many things" is
    therefore the wrong inference.

    Method. Splits are contiguous in time and disjoint; with S blocks there are
    C(S, S/2) partitions, each using S/2 blocks in-sample and the complement
    out-of-sample. Performance is the per-period Sharpe ratio. For each
    partition the in-sample winner is found, its out-of-sample rank omega among
    all N configurations is taken, and the logit
    lambda = log(omega / (1 - omega)) recorded. PBO is the fraction of
    partitions with lambda <= 0.

    Implementation detail worth knowing: Sharpe ratios for every partition are
    computed from per-block sums of x and x^2 rather than by re-slicing the
    matrix, so the cost is one (C x S) by (S x N) matrix product rather than
    C x N independent passes. At S = 16 that is 12 870 partitions, which is
    seconds rather than hours.

    Args:
        trial_returns: Per-configuration simple return series, identical
            indexes. At least two configurations — ranking one is vacuous.
        n_splits: S, the number of contiguous blocks. Must be even and at
            least 4. The paper uses 16. Observations beyond a multiple of S
            are dropped from the END and the count is reported.

    Returns:
        dict with `pbo`, `n_trials`, `n_splits`, `n_partitions`,
        `n_observations_used`, `n_observations_dropped`, `median_logit`,
        `performance_degradation_slope`, `probability_of_loss`, and
        `interpretation`.

    Raises:
        ValueError: on fewer than two trials, an odd or too-small `n_splits`,
            misaligned indexes, NaN, or too few observations per block.
    """
    from itertools import combinations

    names = sorted(trial_returns)
    if len(names) < 2:
        raise ValueError(
            f"probability_of_backtest_overfitting needs at least two trials; got {len(names)}."
        )
    if n_splits < 4 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even and >= 4; got {n_splits}.")

    reference = trial_returns[names[0]].index
    for name in names:
        series = trial_returns[name]
        if not series.index.equals(reference):
            raise ValueError(
                f"probability_of_backtest_overfitting requires identical date indexes; "
                f"'{name}' differs from '{names[0]}' — refusing to align silently."
            )
        if series.isna().any():
            raise ValueError(
                f"probability_of_backtest_overfitting requires NaN-free returns; "
                f"'{name}' has NaN."
            )

    matrix = np.column_stack([trial_returns[n].to_numpy(dtype=float) for n in names])
    n_obs_total, n_trials = matrix.shape
    block_len = n_obs_total // n_splits
    if block_len < 2:
        raise ValueError(
            f"Too few observations for {n_splits} blocks: {n_obs_total} rows gives "
            f"{block_len} per block; need at least 2."
        )
    used = block_len * n_splits
    trimmed = matrix[:used]

    # Per-block sufficient statistics: a Sharpe over any union of blocks is a
    # function of the summed counts, sums and sums of squares.
    blocks = trimmed.reshape(n_splits, block_len, n_trials)
    block_sum = blocks.sum(axis=1)                 # (S, N)
    block_sumsq = (blocks**2).sum(axis=1)          # (S, N)

    partitions = list(combinations(range(n_splits), n_splits // 2))
    masks = np.zeros((len(partitions), n_splits), dtype=float)
    for i, combo in enumerate(partitions):
        masks[i, list(combo)] = 1.0
    complement = 1.0 - masks

    def sharpes(mask: np.ndarray) -> np.ndarray:
        count = block_len * (n_splits // 2)
        total = mask @ block_sum                   # (C, N)
        total_sq = mask @ block_sumsq              # (C, N)
        mean = total / count
        var = total_sq / count - mean**2
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(var > 0, mean / np.sqrt(np.maximum(var, 1e-300)), 0.0)
        return out

    is_perf = sharpes(masks)
    oos_perf = sharpes(complement)

    winners = np.argmax(is_perf, axis=1)
    rows = np.arange(len(partitions))
    winner_oos = oos_perf[rows, winners]

    # Rank of the in-sample winner among all trials out of sample (1 = worst).
    ranks = (oos_perf < winner_oos[:, None]).sum(axis=1) + 1
    omega = ranks / (n_trials + 1.0)
    omega = np.clip(omega, 1e-12, 1 - 1e-12)
    logits = np.log(omega / (1 - omega))

    pbo = float(np.mean(logits <= 0.0))

    # Performance degradation: OOS performance of the chosen config regressed
    # on its IS performance. A negative slope is the overfitting signature.
    winner_is = is_perf[rows, winners]
    if np.std(winner_is) > 0:
        slope = float(np.polyfit(winner_is, winner_oos, 1)[0])
    else:
        slope = float("nan")

    return {
        "pbo": pbo,
        "n_trials": n_trials,
        "n_splits": int(n_splits),
        "n_partitions": len(partitions),
        "n_observations_used": int(used),
        "n_observations_dropped": int(n_obs_total - used),
        "median_logit": float(np.median(logits)),
        "performance_degradation_slope": slope,
        "probability_of_loss": float(np.mean(winner_oos <= 0.0)),
        "interpretation": (
            f"The configuration selected in-sample falls below the out-of-sample "
            f"median in {pbo:.1%} of {len(partitions)} balanced partitions "
            f"(PBO = {pbo:.4f}, {n_trials} trials). PBO rises toward 1 with the "
            f"size of the search regardless of genuine skill, so it describes the "
            f"selection procedure as much as the strategies."
        ),
    }
