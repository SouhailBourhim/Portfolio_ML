"""
test_phase3_integration.py — Phase 3 features ↔ Phase 2 engine seam.

The Phase 2 engine promises to slice every `extras` frame to `:τ` before each
fit; Phase 3 promises to produce causal features. Each side is unit-tested in
isolation, but nothing proved the two actually connect. These tests feed a REAL
Phase 3 feature matrix through `run_backtest` as `extras` and check that:

  1. a feature-consuming strategy only ever receives feature rows dated ≤ τ, and
  2. corrupting the FUTURE of the feature frame cannot change any past weight —
     the end-to-end no-lookahead guarantee, across both modules at once.

This is the wiring Phase 4's HMM / dynamic-covariance strategies will depend on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest
from ml_features import ML_CORE_FEATURES, build_ml_feature_set
from strategies import Strategy

FEATURE_CONFIG = {
    "volatility_short_window": 21,
    "volatility_long_window": 63,
    "correlation_window": 63,
    "correlation_min_periods": 42,
    "macro_lag_days": 1,
}


def _returns(periods: int = 400, assets: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    index = pd.bdate_range("2019-01-01", periods=periods, name="Date")
    values = rng.normal(0.0003, 0.011, size=(periods, assets))
    return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(assets)])


def _macro(returns: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"VIX": 18 + np.cumsum(rng.normal(0, 0.4, len(returns)))},
        index=returns.index,
    )


class _FeatureConsumer(Strategy):
    """Reads the Phase 3 feature frame from `extras`, records what it was shown,
    and actually TILTS weights by trailing volatility — so its output genuinely
    depends on feature values. A strategy that read features but ignored them
    would make the lookahead test below vacuous (it would pass even if the
    engine leaked), so the tilt is load-bearing, not decorative."""

    name = "feature_consumer"

    def __init__(self) -> None:
        self.seen_feature_ends: list[pd.Timestamp] = []
        self.core_was_complete: list[bool] = []

    def fit(self, train_returns, extras=None):
        assets = train_returns.columns
        n = len(assets)
        if not extras or "features" not in extras:
            return pd.Series(1.0 / n, index=assets)

        feats = extras["features"]
        self.seen_feature_ends.append(feats.index.max())
        latest = feats.iloc[-1]
        self.core_was_complete.append(bool(latest[ML_CORE_FEATURES].notna().all()))

        # Load-bearing consumption: the weight on asset 0 is a deterministic,
        # bounded function of the latest short-vol feature (tanh keeps it in
        # (0.25, 0.75)); the rest is split equally. Long-only and sums to 1 by
        # construction — and, crucially, the weights MOVE when the feature moves.
        vol = float(latest["MARKET_VOL_SHORT"])
        w0 = 0.25 + 0.5 * float(np.tanh(vol))
        rest = (1.0 - w0) / (n - 1)
        weights = pd.Series(rest, index=assets)
        weights.iloc[0] = w0
        return weights


def _build_features(returns: pd.DataFrame) -> pd.DataFrame:
    return build_ml_feature_set(returns, _macro(returns), FEATURE_CONFIG)


class TestPhase3FeedsPhase2Engine:
    def test_engine_slices_features_to_train_window(self):
        returns = _returns()
        features = _build_features(returns)
        probe = _FeatureConsumer()

        result = run_backtest(
            returns, probe, min_train_days=200,
            extras={"features": features}, universe_name="test",
        )

        assert probe.seen_feature_ends, "strategy never received the feature frame"
        for feat_end, tau in zip(probe.seen_feature_ends, result.rebalance_dates):
            assert feat_end <= tau  # never a feature dated after the decision date

    def test_core_features_are_dense_by_first_rebalance(self):
        # min_train_days (200) exceeds the feature warm-up, so a fitting strategy
        # must never see NaN core features. This is the invariant Phase 4 relies
        # on — if it ever fails, min_train_days is too small for the config.
        returns = _returns()
        features = _build_features(returns)
        probe = _FeatureConsumer()
        run_backtest(returns, probe, min_train_days=200,
                     extras={"features": features}, universe_name="test")
        assert all(probe.core_was_complete)

    def test_future_feature_values_cannot_change_past_weights(self):
        # The end-to-end lookahead gate spanning BOTH modules. The strategy tilts
        # on MARKET_VOL_SHORT, so corrupting future feature rows genuinely WOULD
        # change post-cutoff weights if those rows reached a fit — which lets us
        # assert both directions: past weights unchanged (no leak) AND future
        # weights changed (proof the strategy really consumes features, so the
        # unchanged-past result is meaningful rather than vacuous).
        returns = _returns()
        features = _build_features(returns)

        clean = run_backtest(returns, _FeatureConsumer(), min_train_days=200,
                             extras={"features": features}, universe_name="test")

        corrupted = features.copy()
        cutoff = corrupted.index[len(corrupted) // 2]
        corrupted.loc[corrupted.index > cutoff] = 99.0
        poisoned = run_backtest(returns, _FeatureConsumer(), min_train_days=200,
                                extras={"features": corrupted}, universe_name="test")

        pre = clean.target_weights.index <= cutoff
        post = clean.target_weights.index > cutoff
        assert pre.any(), "need at least one rebalance before the cutoff"
        assert post.any(), "need at least one rebalance after the cutoff"

        # Past decisions are untouched by future corruption (the no-lookahead guarantee)
        pd.testing.assert_frame_equal(
            clean.target_weights.loc[pre], poisoned.target_weights.loc[pre],
        )
        # Future decisions DO move — the strategy is genuinely feature-driven,
        # so the unchanged-past assertion above is a real guard, not a no-op.
        assert not clean.target_weights.loc[post].equals(poisoned.target_weights.loc[post])

    def test_feature_index_is_a_subset_of_returns_index(self):
        # Label-based slicing in the engine is robust to the feature frame having
        # fewer (warm-up-trimmed) rows, but only if its dates are a clean subset.
        returns = _returns()
        features = _build_features(returns)
        assert features.index.isin(returns.index).all()
        assert features.index.is_monotonic_increasing
