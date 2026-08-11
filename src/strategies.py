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
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import telemetry

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

    `wants_current_weights` is opt-in: when True, the engine additionally
    puts the portfolio's drifted weights at τ into
    `extras[backtest.CURRENT_WEIGHTS_KEY]`. Only strategies whose objective
    genuinely depends on what is already held (a turnover penalty) should
    set it — see `backtest.CURRENT_WEIGHTS_KEY` for why this is not a
    lookahead channel.
    """

    name: str = "abstract"
    wants_current_weights: bool = False

    @abstractmethod
    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        """Return target weights (index == train_returns.columns, sum 1, ≥ 0)."""

    @staticmethod
    def _as_weight_series(
        values: np.ndarray, assets: pd.Index, max_weight: float | None = None
    ) -> pd.Series:
        """
        Clip optimizer dust (−1e-12) to 0 and renormalize to sum exactly 1,
        without letting that renormalization break the per-asset cap.

        The cap argument is not decoration. SLSQP satisfies its equality
        constraint only to solver tolerance, so it can return Σw = 0.9999998;
        dividing by that sum to force Σw = 1 INFLATES every weight, which can
        lift an asset sitting exactly on the (0, max_weight) bound to
        max_weight + 5e-10 — above the engine's own 1e-9 validation tolerance.
        The optimizer never violated its bounds; the renormalization did.

        Phase 4C surfaced this on real data (`xgb_signal_cost`, GLD at the
        0.25 cap): a turnover penalty rewards NOT trading, which pushes
        solutions hard onto the cap boundary and makes the boundary case
        routine rather than rare.

        Fixed by redistributing any above-cap mass into the assets that still
        have headroom (standard water-filling) instead of by loosening the
        engine's check — the engine validating strategy output is the trust
        boundary this whole project rests on, and a producer bug must be
        fixed in the producer.

        The all-zero fallback (SLSQP returned nothing usable) uses uniform
        1/N, but is NOT returned early — it flows through the same feasibility
        check and cap enforcement below, so it respects `max_weight`
        identically to every other branch (the case Sourcery flagged on
        PR #13). Uniform 1/N already satisfies any feasible cap, so for the
        `_optimize_weights` caller — which guarantees `N × max_weight ≥ 1`
        before it ever calls this — the cap loop is a no-op.

        An INFEASIBLE cap (`N × max_weight < 1`, so not even 1/N fits) raises
        here, exactly as `_optimize_weights` raises for the same condition —
        no combination of long-only weights can both sum to 1 and stay under
        such a cap, so returning a sub-1 vector would only defer a guaranteed
        failure to the engine's validator one step later. Raising at the
        source is this project's loud-failure convention.

        Raises:
            ValueError: if a cap is given that no valid weight vector can
                satisfy (`N × max_weight < 1`).
        """
        n = len(assets)
        w = np.clip(np.asarray(values, dtype=float), 0.0, None)
        total = w.sum()
        w = np.full(n, 1.0 / n) if total <= 0.0 else w / total

        if max_weight is not None:
            if n * max_weight < 1.0 - 1e-9:
                raise ValueError(
                    f"Infeasible cap: {n} assets × max_weight {max_weight} < 1 — "
                    f"no long-only weights summing to 1 can satisfy it."
                )
            # Water-filling: cap the over-limit assets, spread their excess
            # into those with headroom. Each pass freezes at least one more
            # asset at the cap, so it converges in ≤ N passes for any feasible
            # cap; the loop is driven by the `excess` tolerance, and the
            # for-else RAISES rather than silently returning an over-cap
            # vector if convergence ever fails numerically (unreachable given
            # the feasibility guard above, but the trust boundary does not
            # rely on "unreachable").
            for _ in range(n + 1):
                excess = float(np.maximum(w - max_weight, 0.0).sum())
                if excess <= 1e-12:
                    break
                w = np.minimum(w, max_weight)
                headroom = max_weight - w
                room = float(headroom.sum())
                if room <= 1e-15:      # every asset already at the cap
                    break
                w = w + excess * headroom / room
            else:
                if float(np.maximum(w - max_weight, 0.0).sum()) > 1e-9:
                    raise RuntimeError(
                        "cap projection failed to converge — residual weight "
                        "above max_weight after water-filling."
                    )

        return pd.Series(w, index=assets)


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


def _extract_current_weights(
    extras: Mapping[str, pd.DataFrame] | None, assets: pd.Index
) -> np.ndarray | None:
    """
    Read the engine-injected drifted weights out of `extras`, or None.

    Returns None whenever the key is absent (strategy called directly in a
    test, or first rebalance before the engine has any position) — callers
    treat that as "no turnover penalty available this fit", never as an
    error. Reindexed to `assets` so a universe reordering cannot silently
    misalign the penalty against the wrong instrument.
    """
    from backtest import CURRENT_WEIGHTS_KEY

    frame = (extras or {}).get(CURRENT_WEIGHTS_KEY)
    if frame is None or len(frame) == 0:
        return None
    return frame.iloc[-1].reindex(assets).fillna(0.0).to_numpy(dtype=float)


def _smooth_turnover(w: np.ndarray, w_prev: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Differentiable surrogate for Σ|wᵢ − w_prev,i|: Σ√((wᵢ − w_prev,i)² + ε).

    SLSQP is a *gradient-based* method and assumes a smooth objective; a
    raw L1 term is non-differentiable exactly at wᵢ = w_prev,i, which is
    precisely where a turnover-penalized optimum wants to sit (don't trade
    this asset at all). Feeding it raw |·| makes the solver chatter around
    that kink and report spurious failures. With ε = 1e-8 the surrogate is
    within 1e-4 per asset of true absolute value — far below any weight
    difference that matters — while being smooth everywhere.
    """
    return float(np.sqrt((w - w_prev) ** 2 + epsilon).sum())


def _optimize_weights(
    objective: Callable[[np.ndarray], float],
    assets: pd.Index,
    max_weight: float,
    strategy_name: str,
    w_prev: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
    n_training_rows: int = 0,
) -> pd.Series:
    """
    Shared SLSQP wrapper: long-only bounds (0, max_weight), Σw = 1, x0 = 1/N.

    Addresses: P1 — the cap stops noisy estimates from producing the
    concentrated corner solutions that collapse out-of-sample.
    Addresses: P1, P4 — with `turnover_penalty > 0` and `w_prev` supplied,
    the objective becomes `objective(w) + λ·Σ|w − w_prev|`, so the optimizer
    prices the cost of *getting to* a portfolio rather than only the merit
    of being in it. This is the direct fix for the Phase 4B finding that
    `rf_signal` had the best GROSS Sharpe of any strategy on `full_2021`
    (1.240) yet lost 0.178 of it to a 0.885 average turnover: a signal can
    be genuinely informative and still be untradeable if acting on every
    revision costs more than the revision is worth.

    λ is in "objective units per unit of turnover" — for the Sharpe
    objective, roughly "annualized Sharpe sacrificed to fully replace the
    portfolio". It is a hyperparameter, NOT estimated from data here;
    Phase 5's purged CV is the honest place to tune it, and until then any
    chosen value is a judgement call that must be reported as one.

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

    if turnover_penalty > 0.0 and w_prev is not None:
        base_objective = objective

        def objective(w: np.ndarray) -> float:  # noqa: F811 — deliberate wrap
            return base_objective(w) + turnover_penalty * _smooth_turnover(w, w_prev)

    bounds = [(0.0, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    x0 = np.full(n, 1.0 / n)

    for attempt, start in enumerate((x0, x0 + np.random.default_rng(0).normal(0, 0.01, n))):
        start = np.clip(start, 0.0, max_weight)
        start = start / start.sum()
        result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            # A "successful" solve can still return a vector with no positive
            # mass, which `_as_weight_series` silently renormalizes to 1/N.
            # That substitution is invisible in the returned Series, so it is
            # recorded here rather than at the projection: this is the only
            # place that still knows the request came from `strategy_name`.
            if np.clip(np.asarray(result.x, dtype=float), 0.0, None).sum() <= 0.0:
                telemetry.record(
                    telemetry.FitRecord(
                        model_requested=strategy_name,
                        model_effective="equal_weight",
                        fit_status=telemetry.STATUS_FALLBACK,
                        n_training_rows=n_training_rows,
                        fallback_reason=(
                            "SLSQP reported success but returned a degenerate "
                            "weight vector with no positive mass"
                        ),
                    )
                )
            return Strategy._as_weight_series(result.x, assets, max_weight)
        log.debug("%s: SLSQP attempt %d failed: %s", strategy_name, attempt + 1, result.message)

    log.warning(
        "%s: optimizer failed twice (%s) — falling back to equal weights for this rebalance.",
        strategy_name, result.message,
    )
    telemetry.record(
        telemetry.FitRecord(
            model_requested=strategy_name,
            model_effective="equal_weight",
            fit_status=telemetry.STATUS_FALLBACK,
            n_training_rows=n_training_rows,
            fallback_reason=f"SLSQP did not converge in 2 attempts: {result.message}",
        )
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
            lambda w: float(w @ cov @ w), train_returns.columns, self.max_weight, self.name,
            n_training_rows=len(train_returns),
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
            lambda w: float(w @ cov @ w), train_returns.columns, self.max_weight, self.name,
            n_training_rows=len(train_returns),
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
            lambda w: float(w @ cov @ w), train_returns.columns, self.max_weight, self.name,
            n_training_rows=len(train_returns),
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
            lambda w: float(w @ cov @ w), train_returns.columns, self.max_weight, self.name,
            n_training_rows=len(train_returns),
        )


def estimate_covariance(
    train_returns: pd.DataFrame,
    estimator: str = "ledoit_wolf",
    halflife_days: int = 63,
    garch_p: int = 1,
    garch_q: int = 1,
    dcc_a_init: float = 0.02,
    dcc_b_init: float = 0.95,
    rescale_factor: float = 100.0,
) -> np.ndarray:
    """
    Annualized covariance from any rung of the Phase 4 ablation ladder.

    Addresses: P1, P2 — factored out so a strategy's covariance choice
    becomes a parameter rather than a hardcoded import. `MinVarianceLW`,
    `MinVarianceEWMA` and `DCCGarchStrategy` each pin one rung as their
    identity and are left alone; this exists for strategies whose identity
    is their `mu` (F7's signal models), so "best predicted mu + best
    covariance" is a config change instead of a new class.

    Args:
        train_returns: `:τ`-sliced log-return window.
        estimator: `"sample"`, `"ledoit_wolf"`, `"ewma"` or `"dcc_garch"`.
        halflife_days: EWMA decay, used only by `"ewma"`.
        garch_p, garch_q, dcc_a_init, dcc_b_init, rescale_factor: forwarded
            to `dcc_garch.dcc_covariance`, used only by `"dcc_garch"`.

    Raises:
        ValueError: on an unknown estimator name — a misspelled config key
            must not silently resolve to a different risk model.
    """
    if estimator == "sample":
        return train_returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
    if estimator == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        return LedoitWolf().fit(train_returns.to_numpy()).covariance_ * TRADING_DAYS_PER_YEAR
    if estimator == "ewma":
        ewm_cov = train_returns.ewm(halflife=halflife_days).cov()
        return ewm_cov.loc[train_returns.index[-1]].to_numpy() * TRADING_DAYS_PER_YEAR
    if estimator == "dcc_garch":
        from dcc_garch import dcc_covariance

        return dcc_covariance(
            train_returns,
            garch_p=garch_p,
            garch_q=garch_q,
            dcc_a_init=dcc_a_init,
            dcc_b_init=dcc_b_init,
            rescale_factor=rescale_factor,
        )
    raise ValueError(
        f"Unknown covariance estimator: {estimator!r} — expected 'sample', "
        f"'ledoit_wolf', 'ewma' or 'dcc_garch'."
    )


def _neg_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free_annual: float) -> float:
    """
    Shared Sharpe-maximization objective: -(wᵀμ − rf)/√(wᵀΣw).

    Addresses: P1 — factored out of `MaxSharpe` so every strategy that only
    changes HOW `mu` is estimated (naive sample mean, or an F7 ML signal
    model) reuses the identical objective and optimizer, rather than each
    duplicating this closure. Mirrors how the covariance ladder
    (`MinVarianceLW`/`MinVarianceEWMA`/`DCCGarchStrategy`) all share one
    min-variance objective and only swap `cov`.
    """
    vol = float(np.sqrt(w @ cov @ w))
    if vol < 1e-12:
        return 0.0
    return -(float(w @ mu) - risk_free_annual) / vol


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
        return _optimize_weights(
            lambda w: _neg_sharpe(w, mu, cov, self.risk_free_annual),
            train_returns.columns, self.max_weight, self.name,
            n_training_rows=len(train_returns),
        )


class _MLSignalStrategy(Strategy):
    """
    Shared implementation for F7's adaptive ML signal strategies
    (`RandomForestSignalStrategy`, `XGBoostSignalStrategy`): predicted
    `mu` from `ml_signals.fit_predict_expected_returns`, a covariance from
    any rung of the Phase 4 ladder, both fed into the identical
    `_neg_sharpe` objective `MaxSharpe` uses. Subclasses set only `name`
    and `model_type`.

    Addresses: P1, P2, P3 — see `ml_signals.py`'s module docstring. This
    class adds ZERO new optimizer code — it is "swap one moment into the
    unmodified engine," the same pattern every prior Phase 4 addition used.

    Phase 4C adds three levers, all defaulting OFF so the Phase 4B result
    stays exactly reproducible as the honest floor to compare against:
      - `turnover_penalty` (+ `wants_current_weights`) — prices the cost of
        trading into the target, the direct fix for Phase 4B's finding that
        `rf_signal` had the best gross Sharpe on `full_2021` (1.240) and
        lost 0.178 of it to 0.885 turnover.
      - `mu_transform`/`shrinkage_weight` — regularizes the predicted `mu`
        (Chopra & Ziemba 1993), see `ml_signals.apply_mu_transform`.
      - `cov_estimator` — pairs a predicted `mu` with any covariance rung,
        making "best mu + best cov" a config change, not a new class.
    """

    model_type: str = "abstract"

    def __init__(
        self,
        max_weight: float = 0.25,
        risk_free_annual: float = 0.0,
        model_params: Mapping | None = None,
        min_train_rows: int = 504,
        short_window: int = 21,
        long_window: int = 63,
        momentum_windows: Sequence[int] = (5, 21, 63),
        condition_on_regime: bool = True,
        n_states: int = 2,
        n_restarts: int = 5,
        random_state_base: int = 0,
        covariance_type: str = "diag",
        min_regime_train_days: int = 252,
        cov_estimator: str = "ledoit_wolf",
        halflife_days: int = 63,
        garch_p: int = 1,
        garch_q: int = 1,
        dcc_a_init: float = 0.02,
        dcc_b_init: float = 0.95,
        rescale_factor: float = 100.0,
        turnover_penalty: float = 0.0,
        mu_transform: str = "none",
        shrinkage_weight: float = 0.5,
        name: str | None = None,
    ) -> None:
        self.max_weight = max_weight
        self.risk_free_annual = risk_free_annual
        self.model_params = model_params
        self.min_train_rows = min_train_rows
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_windows = momentum_windows
        self.condition_on_regime = condition_on_regime
        self.n_states = n_states
        self.n_restarts = n_restarts
        self.random_state_base = random_state_base
        self.covariance_type = covariance_type
        self.min_regime_train_days = min_regime_train_days
        self.cov_estimator = cov_estimator
        self.halflife_days = halflife_days
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.dcc_a_init = dcc_a_init
        self.dcc_b_init = dcc_b_init
        self.rescale_factor = rescale_factor
        self.turnover_penalty = turnover_penalty
        self.mu_transform = mu_transform
        self.shrinkage_weight = shrinkage_weight
        # Per-instance label so one class can appear as several distinct rows
        # in an ablation table (rf_signal vs rf_signal_cost vs ...) without a
        # subclass per configuration. Falls back to the class attribute.
        if name is not None:
            self.name = name
        # Only ask the engine for portfolio state if it is actually used —
        # an unused injection would be a silent widening of what this
        # strategy can see.
        self.wants_current_weights = turnover_penalty > 0.0

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        from ml_signals import fit_predict_expected_returns

        mu = fit_predict_expected_returns(
            train_returns,
            extras,
            model_type=self.model_type,
            model_params=self.model_params,
            min_train_rows=self.min_train_rows,
            short_window=self.short_window,
            long_window=self.long_window,
            momentum_windows=self.momentum_windows,
            condition_on_regime=self.condition_on_regime,
            n_states=self.n_states,
            n_restarts=self.n_restarts,
            random_state_base=self.random_state_base,
            covariance_type=self.covariance_type,
            min_regime_train_days=self.min_regime_train_days,
            mu_transform=self.mu_transform,
            shrinkage_weight=self.shrinkage_weight,
        ).to_numpy()

        cov = estimate_covariance(
            train_returns,
            estimator=self.cov_estimator,
            halflife_days=self.halflife_days,
            garch_p=self.garch_p,
            garch_q=self.garch_q,
            dcc_a_init=self.dcc_a_init,
            dcc_b_init=self.dcc_b_init,
            rescale_factor=self.rescale_factor,
        )
        return _optimize_weights(
            lambda w: _neg_sharpe(w, mu, cov, self.risk_free_annual),
            train_returns.columns, self.max_weight, self.name,
            w_prev=_extract_current_weights(extras, train_returns.columns),
            turnover_penalty=self.turnover_penalty,
            n_training_rows=len(train_returns),
        )


class RandomForestSignalStrategy(_MLSignalStrategy):
    """
    F7 — RandomForest return-prediction signal feeding the Sharpe objective.

    Addresses: P1, P2, P3 — see `_MLSignalStrategy` and `ml_signals.py`.
    """

    name = "rf_signal"
    model_type = "random_forest"


class XGBoostSignalStrategy(_MLSignalStrategy):
    """
    F7 — XGBoost return-prediction signal feeding the Sharpe objective.

    Addresses: P1, P2, P3 — see `_MLSignalStrategy` and `ml_signals.py`.
    """

    name = "xgb_signal"
    model_type = "xgboost"


# LSTMSignalStrategy (a third F7 model family, torch-based) was built and
# passed its own isolated test suite, but was DROPPED from this pass after
# a segfault appeared when running the full suite — torch and xgboost both
# load native/OpenMP-linked libraries, and having both loaded in the same
# process crashed on this machine. Not a code defect in either library;
# deferred to a run on more capable hardware rather than shipped unstable.
# See CLAUDE.md's Phase 4B notes for the full account.


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
            telemetry.record(
                telemetry.FitRecord(
                    model_requested=self.name,
                    model_effective="equal_weight",
                    fit_status=telemetry.STATUS_FALLBACK,
                    n_training_rows=len(train_returns),
                    fallback_reason=(
                        "no regime features supplied — the HMM was never fitted "
                        "and no regime signal entered this allocation"
                    ),
                )
            )
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
            # The dispatch is the substitution: on this rebalance the result
            # is produced entirely by `bear_strategy`, with no regime signal
            # in it. Recorded here rather than in `regime.fit_hmm` because
            # this is where the substitution happens and where the effective
            # model has a name — `fit_hmm` is also called by F7's feature
            # builder, where a neutral posterior is not a portfolio fallback.
            telemetry.record(
                telemetry.FitRecord(
                    model_requested=self.name,
                    model_effective=self.bear_strategy.name,
                    fit_status=telemetry.STATUS_FALLBACK,
                    n_training_rows=len(train_returns),
                    fallback_reason=(
                        "HMM did not converge — dispatched to the defensive "
                        "sub-strategy on a neutral 50/50 posterior"
                    ),
                )
            )

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
