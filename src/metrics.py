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

            raw = json.loads(self.path.read_text())
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
        ))


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
