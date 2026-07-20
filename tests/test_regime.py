"""
test_regime.py — Tests for HMM regime detection.

Uses tiny synthetic feature windows throughout (never real Gold data) so the
suite stays fast and offline, matching repo convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import regime
from regime import REGIME_FEATURES, fit_hmm, label_regimes, predict_regime_posterior


def _two_regime_features(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """80 'bull' rows (positive return, low vol, low correlation) followed by
    80 'bear' rows (negative return, high vol, high correlation) — a
    synthetic stand-in for the causal Phase 3 core features."""
    rng = np.random.default_rng(seed)
    bull = pd.DataFrame(
        {
            "MARKET_RETURN": rng.normal(0.001, 0.003, n),
            "MARKET_VOL_SHORT": rng.normal(0.08, 0.01, n),
            "AVG_PAIRWISE_CORR": rng.normal(0.15, 0.03, n),
        }
    )
    bear = pd.DataFrame(
        {
            "MARKET_RETURN": rng.normal(-0.002, 0.006, n),
            "MARKET_VOL_SHORT": rng.normal(0.30, 0.03, n),
            "AVG_PAIRWISE_CORR": rng.normal(0.65, 0.05, n),
        }
    )
    features = pd.concat([bull, bear], ignore_index=True)
    features.index = pd.bdate_range("2020-01-01", periods=len(features), name="Date")
    return features


class TestFitHMMSeparatesRegimes:
    def test_separates_two_synthetically_distinct_blocks(self):
        n = 80
        features = _two_regime_features(n)
        hmm_fit = fit_hmm(features, min_regime_train_days=50)

        assert hmm_fit.converged
        X = hmm_fit.scaler.transform(features[REGIME_FEATURES].to_numpy())
        states = hmm_fit.model.predict(X)

        # Each block should be assigned overwhelmingly to a single state.
        first_block_counts = np.bincount(states[:n], minlength=2)
        second_block_counts = np.bincount(states[n:], minlength=2)
        assert first_block_counts.max() / n > 0.9
        assert second_block_counts.max() / n > 0.9
        # And the two blocks must land in DIFFERENT states.
        assert np.argmax(first_block_counts) != np.argmax(second_block_counts)

    def test_posteriors_correctly_identify_bull_and_bear_rows(self):
        n = 80
        features = _two_regime_features(n)
        hmm_fit = fit_hmm(features, min_regime_train_days=50)

        bull_posterior = predict_regime_posterior(hmm_fit, features.iloc[: n // 2])
        bear_posterior = predict_regime_posterior(hmm_fit, features)  # ends in a bear row

        assert bull_posterior["bull"] > 0.9
        assert bear_posterior["bear"] > 0.9

    def test_fixed_seed_is_reproducible(self):
        features = _two_regime_features()
        first = fit_hmm(features, min_regime_train_days=50, random_state_base=0)
        second = fit_hmm(features, min_regime_train_days=50, random_state_base=0)

        assert first.converged and second.converged
        assert first.seed_used == second.seed_used
        assert first.log_likelihood == pytest.approx(second.log_likelihood)
        assert first.label_map == second.label_map


class TestLabelRegimes:
    class _FakeModel:
        def __init__(self, means: np.ndarray) -> None:
            self.means_ = means
            self.n_components = means.shape[0]

    def test_higher_mean_return_state_is_labeled_bull_index_zero(self):
        # State 0 has the higher MARKET_RETURN mean.
        model = self._FakeModel(np.array([[0.01, 0.1, 0.2], [-0.01, 0.3, 0.6]]))
        labels = label_regimes(model)
        assert labels == {0: "bull", 1: "bear"}

    def test_higher_mean_return_state_is_labeled_bull_index_one(self):
        # Same as above but with state indices swapped — the mapping must
        # follow the DATA, never a hardcoded index, since hmmlearn doesn't
        # guarantee stable state ordering across fits.
        model = self._FakeModel(np.array([[-0.01, 0.3, 0.6], [0.01, 0.1, 0.2]]))
        labels = label_regimes(model)
        assert labels == {1: "bull", 0: "bear"}

    def test_rejects_non_two_state_models(self):
        model = self._FakeModel(np.array([[0.01, 0.1, 0.2], [0.0, 0.2, 0.4], [-0.01, 0.3, 0.6]]))
        with pytest.raises(ValueError, match="2-state"):
            label_regimes(model)


class TestFailurePolicy:
    def test_short_window_returns_neutral_posterior_and_warns(self, caplog):
        features = _two_regime_features(n=10)  # 20 rows, well under any reasonable min
        with caplog.at_level("WARNING", logger="regime"):
            hmm_fit = fit_hmm(features, min_regime_train_days=252)

        assert not hmm_fit.converged
        assert hmm_fit.model is None
        assert any("returning neutral" in r.message for r in caplog.records)

        posterior = predict_regime_posterior(hmm_fit, features)
        assert posterior == {"bull": 0.5, "bear": 0.5}

    def test_neutral_fallback_never_raises(self):
        # Degenerate window: far below min_regime_train_days, but the
        # function must still return cleanly, never crash the backtest.
        features = _two_regime_features(n=2)
        hmm_fit = fit_hmm(features, min_regime_train_days=252)
        posterior = predict_regime_posterior(hmm_fit, features)
        assert posterior == {"bull": 0.5, "bear": 0.5}


class TestScalerIsFitFreshPerCall:
    def test_scaler_reflects_only_its_own_window(self):
        # Regression guard for the no-global-standardization rule
        # (CLAUDE.md §15.14): two windows with different scales must produce
        # two INDEPENDENTLY-fit scalers, not a shared/cached one.
        low_scale = _two_regime_features(seed=1) * 1.0
        high_scale = _two_regime_features(seed=2) * 5.0

        fit_low = fit_hmm(low_scale, min_regime_train_days=50)
        fit_high = fit_hmm(high_scale, min_regime_train_days=50)

        assert not np.allclose(fit_low.scaler.mean_, fit_high.scaler.mean_)
        assert not np.allclose(fit_low.scaler.scale_, fit_high.scaler.scale_)
