"""
test_phase4c_cost_aware.py — Phase 4C: cost-aware optimization + mu regularization.

Phase 4B's honest negative result had a specific, diagnosable shape rather
than "the model doesn't work": on `full_2021`, `rf_signal` produced the
BEST gross Sharpe of any strategy in the comparison (1.240, above the
Phase 4 winner's 1.204) and then lost 0.178 of it to a 0.885 average
turnover. The signal was informative; acting on every revision of it was
not affordable. Phase 4C attacks exactly that, on two fronts:

  1. Price the trade — a turnover-penalized objective, so the optimizer
     weighs the cost of REACHING a portfolio, not only the merit of
     holding it.
  2. Distrust the magnitudes — shrink the predicted `mu` toward the naive
     sample mean, or keep only its cross-sectional ordering (Chopra &
     Ziemba 1993: error in expected returns hurts a mean-variance
     optimizer roughly an order of magnitude more than equivalent error in
     the covariance).

Every test here is named after the rule it locks in, not the function it
calls, per this repo's convention — the suite doubles as the record of WHY
each mechanism is shaped the way it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import CURRENT_WEIGHTS_KEY, run_backtest
from ml_signals import apply_mu_transform
from strategies import (
    RandomForestSignalStrategy,
    Strategy,
    _extract_current_weights,
    _smooth_turnover,
    estimate_covariance,
)

# Deliberately tiny so RF fits stay in the milliseconds; these tests prove
# MECHANISM, never performance — the real numbers come from run_phase4c.py.
FAST_MODEL = {"n_estimators": 12, "max_depth": 3, "random_state": 0}


@pytest.fixture()
def small_returns() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-04", periods=520, name="Date")
    return pd.DataFrame(
        rng.normal(0.0004, 0.011, size=(520, 4)),
        index=dates,
        columns=["A", "B", "C", "D"],
    )


def _signal_strategy(**overrides) -> RandomForestSignalStrategy:
    kwargs = dict(
        min_train_rows=80,
        condition_on_regime=False,
        model_params=FAST_MODEL,
        max_weight=0.5,
    )
    kwargs.update(overrides)
    return RandomForestSignalStrategy(**kwargs)


# ────────────────────────────────────────────────────────────────────────
# The turnover penalty
# ────────────────────────────────────────────────────────────────────────


class TestTurnoverPenalty:
    def test_penalty_reduces_realized_turnover_through_the_real_engine(self, small_returns):
        """The load-bearing claim of Phase 4C: penalizing turnover reduces it.

        Measured end-to-end on the real engine (not on the objective in
        isolation), because that is the quantity the Phase 4B result was
        actually lost to.
        """
        unpenalized = run_backtest(
            small_returns, _signal_strategy(turnover_penalty=0.0),
            min_train_days=252, max_weight=0.5,
        )
        penalized = run_backtest(
            small_returns, _signal_strategy(turnover_penalty=1.0),
            min_train_days=252, max_weight=0.5,
        )
        assert penalized.turnover.mean() < unpenalized.turnover.mean()

    def test_turnover_falls_monotonically_as_the_penalty_rises(self, small_returns):
        """λ is a dial, not a switch — a non-monotone response would mean the
        smooth surrogate or the solver, not the penalty, is driving the result."""
        turnovers = []
        for penalty in (0.0, 0.5, 2.0):
            result = run_backtest(
                small_returns, _signal_strategy(turnover_penalty=penalty),
                min_train_days=252, max_weight=0.5,
            )
            turnovers.append(result.turnover.mean())
        assert turnovers == sorted(turnovers, reverse=True)

    def test_zero_penalty_reproduces_the_unpenalized_weights_exactly(self, small_returns):
        """Phase 4B's result must stay bit-reproducible as the honest floor.

        A "fix" that quietly changes the baseline it is measured against is
        not a fix; it's a moved goalpost.
        """
        window = small_returns.iloc[:300]
        default = RandomForestSignalStrategy(
            min_train_rows=80, condition_on_regime=False,
            model_params=FAST_MODEL, max_weight=0.5,
        )
        explicit_zero = _signal_strategy(turnover_penalty=0.0)
        pd.testing.assert_series_equal(
            default.fit(window), explicit_zero.fit(window)
        )

    def test_penalty_is_inert_without_portfolio_state(self, small_returns):
        """Called directly (no engine, so no `current_weights`), a penalized
        strategy must still return valid weights rather than crash — the same
        "degrade, never crash" rule every estimator in this codebase follows."""
        window = small_returns.iloc[:300]
        weights = _signal_strategy(turnover_penalty=1.0).fit(window, extras=None)
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()

    def test_smooth_surrogate_approximates_absolute_value(self):
        """SLSQP needs a differentiable objective; the surrogate must still be
        numerically indistinguishable from true L1 at weight-scale differences."""
        w = np.array([0.4, 0.3, 0.2, 0.1])
        w_prev = np.array([0.1, 0.3, 0.25, 0.35])
        assert _smooth_turnover(w, w_prev) == pytest.approx(
            np.abs(w - w_prev).sum(), abs=1e-3
        )

    def test_smooth_surrogate_is_finite_and_differentiable_at_the_kink(self):
        """The exact point a turnover-penalized optimum wants to sit (trade
        nothing) is where raw |·| is non-differentiable."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        value = _smooth_turnover(w, w.copy())
        assert np.isfinite(value)
        assert value == pytest.approx(0.0, abs=1e-3)


# ────────────────────────────────────────────────────────────────────────
# The engine's current_weights channel
# ────────────────────────────────────────────────────────────────────────


class _WeightStateSpy(Strategy):
    """Records what portfolio state the engine handed it, per rebalance."""

    name = "weight_state_spy"

    def __init__(self, wants: bool) -> None:
        self.wants_current_weights = wants
        self.seen: list[pd.DataFrame | None] = []

    def fit(self, train_returns, extras=None):
        self.seen.append((extras or {}).get(CURRENT_WEIGHTS_KEY))
        n = train_returns.shape[1]
        return pd.Series(1.0 / n, index=train_returns.columns)


class TestCurrentWeightsChannel:
    def test_engine_injects_portfolio_state_only_when_the_strategy_asks(self, small_returns):
        """Opt-in: a strategy that never requested portfolio state must not be
        handed it. Widening what a strategy can see, silently, is exactly the
        class of change this project's no-lookahead guarantee cannot tolerate."""
        opted_out = _WeightStateSpy(wants=False)
        run_backtest(small_returns, opted_out, min_train_days=252)
        assert all(seen is None for seen in opted_out.seen)

        opted_in = _WeightStateSpy(wants=True)
        run_backtest(small_returns, opted_in, min_train_days=252)
        assert all(isinstance(seen, pd.DataFrame) for seen in opted_in.seen)

    def test_injected_state_is_never_dated_after_its_rebalance(self, small_returns):
        """The new channel obeys the SAME `index.max() <= tau` invariant the
        no-lookahead suite already enforces on every other extras frame — it is
        checked by that rule, not exempted from it."""
        spy = _WeightStateSpy(wants=True)
        result = run_backtest(small_returns, spy, min_train_days=252)
        for frame, tau in zip(spy.seen, result.rebalance_dates):
            assert frame.index.max() <= tau

    def test_first_rebalance_sees_cash_not_a_phantom_position(self, small_returns):
        """Turnover from cash is 1.0 by convention; the state that produces it
        must actually be zeros, not an uninitialized carry-over."""
        spy = _WeightStateSpy(wants=True)
        run_backtest(small_returns, spy, min_train_days=252)
        assert spy.seen[0].to_numpy().sum() == pytest.approx(0.0)

    def test_extract_returns_none_when_state_is_absent(self, small_returns):
        assert _extract_current_weights(None, small_returns.columns) is None
        assert _extract_current_weights({}, small_returns.columns) is None

    def test_extract_realigns_to_the_asset_order_it_is_given(self):
        """A universe reordering must not silently penalize the wrong instrument."""
        assets = pd.Index(["A", "B", "C"])
        frame = pd.DataFrame(
            [[0.5, 0.2, 0.3]],
            index=pd.DatetimeIndex(["2024-01-31"]),
            columns=["C", "A", "B"],
        )
        extracted = _extract_current_weights({CURRENT_WEIGHTS_KEY: frame}, assets)
        np.testing.assert_allclose(extracted, [0.2, 0.3, 0.5])

    def test_signal_strategy_requests_state_only_when_penalized(self):
        assert _signal_strategy(turnover_penalty=0.0).wants_current_weights is False
        assert _signal_strategy(turnover_penalty=0.5).wants_current_weights is True


# ────────────────────────────────────────────────────────────────────────
# mu regularization
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def mu_pair() -> tuple[pd.Series, pd.Series]:
    predicted = pd.Series({"A": 0.30, "B": -0.10, "C": 0.05, "D": 0.12})
    naive = pd.Series({"A": 0.02, "B": 0.03, "C": 0.01, "D": 0.04})
    return predicted, naive


class TestMuTransforms:
    def test_none_passes_the_prediction_through_untouched(self, mu_pair):
        predicted, naive = mu_pair
        pd.testing.assert_series_equal(
            apply_mu_transform(predicted, naive, "none"), predicted
        )

    def test_shrink_with_zero_weight_is_exactly_the_naive_mean(self, mu_pair):
        """The degenerate end of the dial must land exactly on `MaxSharpe`'s own
        estimator — that is what makes the blend interpretable."""
        predicted, naive = mu_pair
        pd.testing.assert_series_equal(
            apply_mu_transform(predicted, naive, "shrink", 0.0), naive
        )

    def test_shrink_with_unit_weight_is_exactly_the_prediction(self, mu_pair):
        predicted, naive = mu_pair
        pd.testing.assert_series_equal(
            apply_mu_transform(predicted, naive, "shrink", 1.0), predicted
        )

    def test_shrink_pulls_strictly_between_the_two_estimates(self, mu_pair):
        predicted, naive = mu_pair
        blended = apply_mu_transform(predicted, naive, "shrink", 0.5)
        lower = pd.concat([predicted, naive], axis=1).min(axis=1)
        upper = pd.concat([predicted, naive], axis=1).max(axis=1)
        assert (blended >= lower - 1e-12).all()
        assert (blended <= upper + 1e-12).all()

    def test_rank_preserves_the_prediction_ordering(self, mu_pair):
        """The whole point of the rank tilt: keep WHAT the model ranked highest,
        discard its claim about BY HOW MUCH."""
        predicted, naive = mu_pair
        ranked = apply_mu_transform(predicted, naive, "rank")
        assert list(predicted.rank()) == list(ranked.rank())

    def test_rank_borrows_level_and_dispersion_from_the_naive_estimate(self, mu_pair):
        predicted, naive = mu_pair
        ranked = apply_mu_transform(predicted, naive, "rank")
        assert ranked.mean() == pytest.approx(naive.mean())
        assert ranked.std(ddof=0) == pytest.approx(naive.std(ddof=0))

    def test_rank_discards_the_predicted_magnitudes(self, mu_pair):
        """A prediction 10x larger but identically ordered must produce the
        SAME tilt — otherwise magnitudes are still leaking through."""
        predicted, naive = mu_pair
        modest = apply_mu_transform(predicted, naive, "rank")
        extreme = apply_mu_transform(predicted * 10.0, naive, "rank")
        pd.testing.assert_series_equal(modest, extreme)

    def test_flat_prediction_falls_back_to_naive_under_rank(self, mu_pair):
        """A model with no cross-sectional opinion must not be amplified into
        one by a divide-by-(near-)zero."""
        _, naive = mu_pair
        flat = pd.Series(0.05, index=naive.index)
        pd.testing.assert_series_equal(
            apply_mu_transform(flat, naive, "rank"), naive
        )

    def test_unknown_transform_raises_rather_than_silently_choosing_one(self, mu_pair):
        predicted, naive = mu_pair
        with pytest.raises(ValueError, match="mu_transform"):
            apply_mu_transform(predicted, naive, "definitely_not_a_transform")


class TestMuTransformReachesTheStrategy:
    def test_transform_choice_changes_the_resulting_weights(self, small_returns):
        """Proves the parameter is actually threaded through
        `_MLSignalStrategy.fit` → `fit_predict_expected_returns`, rather than
        being accepted and dropped."""
        window = small_returns.iloc[:400]
        raw = _signal_strategy(mu_transform="none").fit(window)
        ranked = _signal_strategy(mu_transform="rank").fit(window)
        assert not np.allclose(raw.to_numpy(), ranked.to_numpy())

    def test_full_shrinkage_to_naive_matches_the_naive_mu_strategy(self, small_returns):
        """`shrinkage_weight=0` must reproduce a strategy that never consulted
        the model at all — the cleanest possible check that shrinkage is doing
        arithmetic on the right two vectors."""
        from strategies import _neg_sharpe, _optimize_weights

        window = small_returns.iloc[:400]
        shrunk_to_naive = _signal_strategy(
            mu_transform="shrink", shrinkage_weight=0.0
        ).fit(window)

        mu = window.mean().to_numpy() * 252
        cov = estimate_covariance(window, "ledoit_wolf")
        expected = _optimize_weights(
            lambda w: _neg_sharpe(w, mu, cov, 0.0),
            window.columns, 0.5, "reference",
        )
        np.testing.assert_allclose(
            shrunk_to_naive.to_numpy(), expected.to_numpy(), atol=1e-6
        )


# ────────────────────────────────────────────────────────────────────────
# Covariance estimator selection (best-mu + best-cov)
# ────────────────────────────────────────────────────────────────────────


class TestCovarianceEstimatorSelection:
    @pytest.mark.parametrize("estimator", ["sample", "ledoit_wolf", "ewma", "dcc_garch"])
    def test_every_estimator_returns_a_valid_covariance_matrix(self, estimator, small_returns):
        cov = estimate_covariance(small_returns, estimator)
        n = small_returns.shape[1]
        assert cov.shape == (n, n)
        np.testing.assert_allclose(cov, cov.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(cov) > -1e-8)   # PSD

    def test_unknown_estimator_raises_rather_than_defaulting(self, small_returns):
        with pytest.raises(ValueError, match="covariance estimator"):
            estimate_covariance(small_returns, "not_a_real_estimator")

    def test_estimator_choice_changes_the_resulting_weights(self, small_returns):
        """If swapping the risk model left weights identical, the parameter
        would be decorative."""
        window = small_returns.iloc[:400]
        lw = _signal_strategy(cov_estimator="ledoit_wolf").fit(window)
        ewma = _signal_strategy(cov_estimator="ewma").fit(window)
        assert not np.allclose(lw.to_numpy(), ewma.to_numpy())


# ────────────────────────────────────────────────────────────────────────
# End-to-end: the composed variant still satisfies every engine invariant
# ────────────────────────────────────────────────────────────────────────


class TestComposedVariantThroughTheEngine:
    def test_all_levers_together_survive_validate_weights(self, small_returns):
        """The engine validates weights rather than trusting them; a strategy
        stacking every Phase 4C lever at once must still clear that boundary at
        every rebalance, including its fallback paths."""
        strategy = _signal_strategy(
            turnover_penalty=1.0,
            mu_transform="shrink",
            shrinkage_weight=0.3,
            cov_estimator="ewma",
            name="rf_signal_everything",
        )
        result = run_backtest(
            small_returns, strategy, min_train_days=252,
            cost_bps=10.0, max_weight=0.5,
        )
        assert result.strategy_name == "rf_signal_everything"
        assert len(result.rebalance_dates) > 0
        weights = result.target_weights
        assert np.allclose(weights.sum(axis=1).to_numpy(), 1.0)
        assert (weights.to_numpy() >= -1e-12).all()
        assert weights.to_numpy().max() <= 0.5 + 1e-9

    def test_per_instance_name_does_not_mutate_the_class(self):
        """Ablation variants share one class; a per-instance label must not
        leak into every other instance's identity."""
        variant = _signal_strategy(name="rf_signal_cost")
        plain = _signal_strategy()
        assert variant.name == "rf_signal_cost"
        assert plain.name == "rf_signal"
        assert RandomForestSignalStrategy.name == "rf_signal"

    def test_future_corruption_cannot_change_past_weights_with_portfolio_state(self):
        """The no-lookahead gate, re-run for the NEW input channel.

        Phase 4C widens what a strategy can see — it now receives portfolio
        state. That state is a function of past fit() outputs and returns
        strictly before τ, so it cannot carry future information; but
        "cannot" is an argument, and this project's standard is a test.
        Corrupting the future of the returns matrix must still leave every
        past rebalance's weights bit-identical, while genuinely changing
        future ones (so the unchanged-past assertion isn't vacuous).
        """
        rng = np.random.default_rng(5)
        index = pd.bdate_range("2019-01-01", periods=400, name="Date")
        returns = pd.DataFrame(
            rng.normal(0.0003, 0.011, size=(400, 4)),
            index=index, columns=[f"A{i}" for i in range(4)],
        )

        def penalized():
            # Uncapped so different signals produce visibly different
            # weights, matching test_phase4b_integration's reasoning.
            return _signal_strategy(
                turnover_penalty=1.0, max_weight=1.0, min_train_rows=50,
            )

        clean = run_backtest(
            returns, penalized(), rebalance_freq="ME",
            min_train_days=120, universe_name="test",
        )

        cutoff = clean.rebalance_dates[1]
        corrupted = returns.copy()
        mask = corrupted.index > cutoff
        noise = np.random.default_rng(123)
        corrupted.loc[mask, "A0"] = 0.02 + noise.normal(0, 0.02, mask.sum())
        corrupted.loc[mask, "A1"] = -0.02 + noise.normal(0, 0.02, mask.sum())

        poisoned = run_backtest(
            corrupted, penalized(), rebalance_freq="ME",
            min_train_days=120, universe_name="test",
        )

        pre = clean.target_weights.index <= cutoff
        post = clean.target_weights.index > cutoff
        assert pre.sum() >= 1 and post.sum() >= 1

        pd.testing.assert_frame_equal(
            clean.target_weights.loc[pre], poisoned.target_weights.loc[pre]
        )
        assert not clean.target_weights.loc[post].equals(
            poisoned.target_weights.loc[post]
        )
