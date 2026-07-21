"""
test_phase4b_integration.py — F7 adaptive ML signal strategies ↔ Phase 2
engine seam.

Mirrors tests/test_phase4_integration.py's pattern, but proves a materially
different claim: `ml_signals.py`'s unit tests already prove
`build_asset_features` is causal and `build_supervised_dataset` structurally
excludes the current rebalance date from training. What ISN'T proven in
isolation is that a REAL trained model, run through the REAL engine, cannot
have its PAST decisions changed by corrupting the FUTURE of the returns
matrix its own labels are derived from — exactly where a subtle leak would
hide, since F7's features and labels come from the same source (unlike
Phase 3/4's features, which were a separate frame from `returns`).

Parametrized over both RandomForestSignalStrategy and XGBoostSignalStrategy
— they share the identical `_MLSignalStrategy.fit()` implementation and
`ml_signals.fit_predict_expected_returns` orchestration, differing only in
`model_type`, so one shared suite covers both rather than duplicating it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import run_backtest
from strategies import RandomForestSignalStrategy, XGBoostSignalStrategy

SIGNAL_STRATEGY_CLASSES = [RandomForestSignalStrategy, XGBoostSignalStrategy]


def _returns(periods: int = 250, assets: int = 4, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-01", periods=periods, name="Date")
    values = rng.normal(0.0003, 0.011, size=(periods, assets))
    return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(assets)])


@pytest.mark.parametrize("strategy_cls", SIGNAL_STRATEGY_CLASSES, ids=lambda c: c.name)
class TestPhase4BSignalStrategyFeedsPhase2Engine:
    def test_default_constructor_survives_validate_weights(self, strategy_cls):
        # Smoke test with the out-of-the-box constructor (capped, regime-
        # conditioning on) -- confirms the whole pipeline (feature build,
        # regime attach, model fit/predict, Ledoit-Wolf cov, SLSQP) produces
        # weights the engine's zero-trust `_validate_weights` accepts.
        returns = _returns()
        strategy = strategy_cls(min_train_rows=50)
        result = run_backtest(
            returns, strategy, rebalance_freq="ME", min_train_days=120,
            universe_name="test",
        )
        assert len(result.rebalance_dates) >= 1
        assert result.target_weights.sum(axis=1).sub(1.0).abs().max() < 1e-6

    def test_fallback_path_survives_validate_weights_on_a_short_window(self, strategy_cls):
        # min_train_rows set impossibly high -> every rebalance falls back
        # to the naive sample mean; must still produce valid weights.
        returns = _returns(periods=150)
        strategy = strategy_cls(min_train_rows=1_000_000)
        result = run_backtest(
            returns, strategy, rebalance_freq="ME", min_train_days=100,
            universe_name="test",
        )
        assert len(result.rebalance_dates) >= 1
        assert result.target_weights.sum(axis=1).sub(1.0).abs().max() < 1e-6

    def test_future_return_corruption_cannot_change_past_weights(self, strategy_cls):
        # The end-to-end no-lookahead gate for a REAL Phase 4B strategy.
        # Corrupting the future of the SAME returns matrix that both
        # features and labels are derived from must never change a past
        # rebalance's weights, while genuinely changing future ones --
        # proving the strategy is really data-driven, so the unchanged-past
        # assertion means something.
        returns = _returns()

        def uncapped() -> object:
            # Uncapped so genuinely different predicted-mu signals produce
            # visibly different weights -- the default 0.25 cap on a
            # 4-asset universe forces near-uniform weights regardless of
            # the signal, which would make the assertion below vacuous.
            return strategy_cls(max_weight=1.0, min_train_rows=50, condition_on_regime=False)

        clean = run_backtest(
            returns, uncapped(), rebalance_freq="ME",
            min_train_days=120, universe_name="test",
        )

        cutoff = clean.rebalance_dates[1]
        corrupted = returns.copy()
        mask = corrupted.index > cutoff
        rng = np.random.default_rng(123)
        # A controlled, plausible-magnitude corruption (not an absurd value
        # like 99.0, which overflows exp(cumsum(...)) inside build_asset_
        # features and only exercises the NaN-fallback path, not the real
        # label-leak question): a strong divergent drift for two assets.
        corrupted.loc[mask, "A0"] = 0.02 + rng.normal(0, 0.02, mask.sum())
        corrupted.loc[mask, "A1"] = -0.02 + rng.normal(0, 0.02, mask.sum())

        poisoned = run_backtest(
            corrupted, uncapped(), rebalance_freq="ME",
            min_train_days=120, universe_name="test",
        )

        pre = clean.target_weights.index <= cutoff
        post = clean.target_weights.index > cutoff
        assert pre.sum() >= 1, "need at least one rebalance before the cutoff"
        assert post.sum() >= 1, "need at least one rebalance after the cutoff"

        pd.testing.assert_frame_equal(
            clean.target_weights.loc[pre], poisoned.target_weights.loc[pre]
        )
        assert not clean.target_weights.loc[post].equals(poisoned.target_weights.loc[post])

    def test_engine_never_shows_the_model_a_future_label(self, strategy_cls):
        # A second angle on the same guarantee, using an instrumented
        # subclass: records the max date seen in train_returns at every
        # fit() call and asserts it never exceeds the rebalance date --
        # the same spy pattern test_backtest.py::TestNoLookahead uses,
        # now specifically for a strategy that builds its own labels
        # from that same train_returns frame.
        seen_train_ends = []

        class _Spy(strategy_cls):
            def fit(self, train_returns, extras=None):
                seen_train_ends.append(train_returns.index.max())
                return super().fit(train_returns, extras)

        returns = _returns()
        strategy = _Spy(max_weight=1.0, min_train_rows=50, condition_on_regime=False)
        result = run_backtest(
            returns, strategy, rebalance_freq="ME", min_train_days=120,
            universe_name="test",
        )

        assert seen_train_ends, "strategy was never fit"
        for train_end, tau in zip(seen_train_ends, result.rebalance_dates):
            assert train_end <= tau
