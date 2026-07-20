"""
test_phase4_integration.py — RegimeConditionalStrategy ↔ Phase 2 engine seam.

Mirrors tests/test_phase3_integration.py: unit tests already prove regime.py
and strategies.py work in isolation, but nothing proves the ACTUAL Phase 4
strategy — not a toy probe — reaches the engine without leaking. These tests
feed synthetic returns with an engineered bull→bear shift through a real
`RegimeConditionalStrategy` via `run_backtest` and check the same end-to-end
no-lookahead guarantee Phase 3 established, now proven for a genuine
composition strategy (regime detection + sub-strategy dispatch), not a spy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import run_backtest
from ml_features import build_ml_feature_set
from strategies import MaxSharpe, MinVarianceLW, RegimeConditionalStrategy

FEATURE_CONFIG = {
    "volatility_short_window": 21,
    "volatility_long_window": 63,
    "correlation_window": 63,
    "correlation_min_periods": 42,
    "macro_lag_days": 1,
}


def _regime_shift_returns(n_bull: int = 250, n_bear: int = 250, seed: int = 99) -> pd.DataFrame:
    """
    4 assets over a synthetic bull→bear shift (shared factor, low→high vol
    and low→high correlation), with asset-level heterogeneity (HI: high
    mean/vol, LOW: low vol, F1/F2: filler) so MaxSharpe and MinVarianceLW
    give MEASURABLY different weights on the same data — otherwise a test
    asserting "dispatch changed the weights" would be vacuous.
    """
    rng = np.random.default_rng(seed)
    n = n_bull + n_bear
    dates = pd.bdate_range("2018-01-02", periods=n, name="Date")
    factor_bull = rng.normal(0.0006, 0.006, n_bull)
    factor_bear = rng.normal(-0.001, 0.02, n_bear)
    factor = np.concatenate([0.3 * factor_bull, 0.8 * factor_bear])
    idio_scale = np.concatenate([np.full(n_bull, 0.008), np.full(n_bear, 0.008 * 0.3)])

    hi = 0.0015 + factor + rng.normal(0, 0.010, n)
    low = 0.0000 + 0.3 * factor + rng.normal(0, 0.002, n)
    f1 = factor + rng.normal(0, idio_scale)
    f2 = factor + rng.normal(0, idio_scale)
    return pd.DataFrame({"HI": hi, "LOW": low, "F1": f1, "F2": f2}, index=dates)


def _macro(returns: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {"VIX": 18 + np.cumsum(rng.normal(0, 0.3, len(returns)))}, index=returns.index
    )


def _build_features(returns: pd.DataFrame) -> pd.DataFrame:
    return build_ml_feature_set(returns, _macro(returns), FEATURE_CONFIG)


def _discriminating_strategy() -> RegimeConditionalStrategy:
    # Uncapped sub-strategies so bull/bear dispatch is visibly different —
    # the default 0.25 cap forces near-uniform weights on a 4-asset universe
    # regardless of objective, which would make several assertions vacuous.
    return RegimeConditionalStrategy(
        bull_strategy=MaxSharpe(max_weight=1.0),
        bear_strategy=MinVarianceLW(max_weight=1.0),
        min_regime_train_days=60,
        n_restarts=3,
    )


class TestPhase4RegimeStrategyFeedsPhase2Engine:
    def test_engine_slices_features_to_train_window(self):
        returns = _regime_shift_returns()
        features = _build_features(returns)
        strategy = _discriminating_strategy()

        result = run_backtest(
            returns, strategy, rebalance_freq="ME", min_train_days=150,
            extras={"features": features}, universe_name="test",
        )

        assert strategy.regime_log, "strategy never received the feature frame"
        for entry, tau in zip(strategy.regime_log, result.rebalance_dates):
            # Each fit() logged train_returns.index[-1] as the feature-window
            # end it saw — must equal the rebalance date itself, never later.
            assert entry["date"] == tau

    def test_weights_match_dispatched_substrategy_for_a_known_confident_regime(self):
        returns = _regime_shift_returns()
        features = _build_features(returns)
        strategy = _discriminating_strategy()

        result = run_backtest(
            returns, strategy, rebalance_freq="ME", min_train_days=150,
            extras={"features": features}, universe_name="test",
        )

        confident = [e for e in strategy.regime_log if e["posterior"].get(e["regime"], 0) > 0.95]
        assert confident, "no confidently-classified rebalance to test against"

        for entry in confident:
            tau = entry["date"]
            train = returns.loc[:tau]
            feats = features.loc[:tau]
            sub = MaxSharpe(max_weight=1.0) if entry["regime"] == "bull" else MinVarianceLW(max_weight=1.0)
            expected = sub.fit(train, {"features": feats})
            actual = result.target_weights.loc[tau]
            pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_future_feature_corruption_cannot_change_past_weights(self):
        # The end-to-end lookahead gate for a REAL Phase 4 strategy (not the
        # toy _FeatureConsumer Phase 3 used). Corrupting the future of the
        # feature frame must never change a past rebalance's weights, while
        # genuinely changing future ones — proving the strategy is really
        # feature-driven, so the unchanged-past assertion means something.
        returns = _regime_shift_returns()
        features = _build_features(returns)

        clean_strategy = _discriminating_strategy()
        clean = run_backtest(
            returns, clean_strategy, rebalance_freq="ME", min_train_days=150,
            extras={"features": features}, universe_name="test",
        )

        cutoff = clean.rebalance_dates[1]
        corrupted = features.copy()
        corrupted.loc[corrupted.index > cutoff] = 99.0

        poisoned_strategy = _discriminating_strategy()
        poisoned = run_backtest(
            returns, poisoned_strategy, rebalance_freq="ME", min_train_days=150,
            extras={"features": corrupted}, universe_name="test",
        )

        pre = clean.target_weights.index <= cutoff
        post = clean.target_weights.index > cutoff
        assert pre.sum() >= 1, "need at least one rebalance before the cutoff"
        assert post.sum() >= 1, "need at least one rebalance after the cutoff"

        pd.testing.assert_frame_equal(
            clean.target_weights.loc[pre], poisoned.target_weights.loc[pre]
        )
        assert not clean.target_weights.loc[post].equals(poisoned.target_weights.loc[post])

    def test_default_constructor_survives_validate_weights_including_fallback_path(self):
        # Default constructor (no args) end-to-end smoke test: out-of-the-box
        # wiring (MaxSharpe/MinVarianceLW defaults with the standard 0.25
        # cap, equal-weight fallback, defensive-bear-on-non-convergence
        # fallback) must all produce weights the engine's zero-trust
        # validation (_validate_weights) accepts. A short window is used
        # deliberately to exercise the non-convergence fallback path too.
        returns = _regime_shift_returns(n_bull=60, n_bear=60)
        features = _build_features(returns)
        strategy = RegimeConditionalStrategy()

        result = run_backtest(
            returns, strategy, rebalance_freq="ME", min_train_days=60,
            extras={"features": features}, universe_name="test",
        )
        assert len(result.rebalance_dates) >= 1
        assert result.target_weights.sum(axis=1).sub(1.0).abs().max() < 1e-6

    def test_feature_index_is_a_subset_of_returns_index(self):
        returns = _regime_shift_returns()
        features = _build_features(returns)
        assert features.index.isin(returns.index).all()
        assert features.index.is_monotonic_increasing
