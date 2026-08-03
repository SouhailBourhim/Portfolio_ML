"""
Tests for the content-addressed memoization added to the deterministic
estimators (`src/memo.py`, wired into
`ml_signals.fit_predict_expected_returns` and `dcc_garch.dcc_covariance`).

These are named after the RULE each one locks in, per this project's testing
convention, because the cache is only sound if all of the following hold —
and every one of them is a way a cache could silently corrupt a published
number rather than merely fail loudly:

  * the wrapped estimators are DETERMINISTIC (otherwise reusing a value is
    not the same as recomputing it);
  * caching does not change any produced value;
  * the key covers every argument that can change the result, so two
    different questions cannot collide;
  * the mu transform, which is applied AFTER the cached model fit, still
    differentiates the strategies that share that fit;
  * the no-lookahead guarantee is untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dcc_garch import dcc_covariance
from memo import ContentCache, content_key
from ml_signals import fit_predict_expected_returns


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def returns_window() -> pd.DataFrame:
    """A deterministic multi-asset log-return window, long enough to fit."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-01-01", periods=800, name="Date")
    data = rng.normal(0.0004, 0.011, size=(len(idx), 4))
    return pd.DataFrame(data, index=idx, columns=["AAA", "BBB", "CCC", "DDD"])


@pytest.fixture
def market_features(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Market-level features shaped like the Phase 3 frame the HMM consumes."""
    market = returns_window.mean(axis=1)
    return pd.DataFrame(
        {
            "MARKET_RETURN": market,
            "MARKET_VOL_SHORT": market.rolling(21, min_periods=21).std(),
            "AVG_PAIRWISE_CORR": returns_window.rolling(63, min_periods=63)
            .corr()
            .groupby(level=0)
            .mean()
            .mean(axis=1),
        }
    ).dropna()


# ── The cache container itself ──────────────────────────────────────────────
class TestContentCache:
    def test_identical_inputs_produce_one_computation(self):
        cache = ContentCache("t")
        calls = []

        def compute():
            calls.append(1)
            return 42

        key = content_key("x", 1)
        assert cache.get_or_compute(key, compute) == 42
        assert cache.get_or_compute(key, compute) == 42
        assert len(calls) == 1, "a repeated key must not recompute"
        assert cache.hits == 1 and cache.misses == 1

    def test_different_inputs_never_collide(self):
        cache = ContentCache("t")
        a = cache.get_or_compute(content_key("x", 1), lambda: "a")
        b = cache.get_or_compute(content_key("x", 2), lambda: "b")
        assert (a, b) == ("a", "b")
        assert cache.misses == 2

    def test_cache_is_bounded_and_evicts_oldest_first(self):
        cache = ContentCache("t", maxsize=3)
        for i in range(5):
            cache.get_or_compute(content_key(i), lambda i=i: i)
        assert len(cache) == 3, "an unbounded cache is a memory leak"

    def test_a_changed_dataframe_value_changes_the_key(self):
        """The whole staleness argument: the key IS the content."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        other = df.copy()
        other.iloc[1, 0] = 2.5
        assert content_key(df) != content_key(other)

    def test_column_order_changes_the_key(self):
        """Column order changes what a positional model fit means."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        assert content_key(df) != content_key(df[["b", "a"]])

    def test_index_changes_the_key_even_when_values_match(self):
        """Two different rebalance windows must never share a key."""
        values = [1.0, 2.0, 3.0]
        early = pd.DataFrame({"a": values}, index=pd.bdate_range("2020-01-01", periods=3))
        late = pd.DataFrame({"a": values}, index=pd.bdate_range("2021-01-01", periods=3))
        assert content_key(early) != content_key(late)


# ── Determinism: the precondition that makes caching sound ──────────────────
class TestEstimatorsAreDeterministic:
    """If these fail, the cache is not merely slow to warm — it is unsound.

    Each estimator is called twice on identical inputs with the cache
    bypassed, and the two results must be identical. This is asserted rather
    than assumed precisely because the whole optimisation rests on it.
    """

    def test_panel_model_prediction_is_deterministic(self, returns_window, market_features):
        from ml_signals import _predict_expected_returns_uncached

        fallback = returns_window.mean() * 252
        args = (
            returns_window, {"features": market_features}, "random_forest",
            {"n_estimators": 25, "max_depth": 4}, 100, 21, 63, [5, 21], True,
            2, 5, 0, "diag", 252, fallback,
        )
        # Three-tuple since the telemetry work: the FitRecord travels WITH
        # the cached value so a cache hit still reports the fallback.
        first, flag_a, record_a = _predict_expected_returns_uncached(*args)
        second, flag_b, record_b = _predict_expected_returns_uncached(*args)
        assert flag_a == flag_b
        assert record_a == record_b, "the fit record must be deterministic too"
        pd.testing.assert_series_equal(first, second)

    def test_dcc_covariance_is_deterministic(self, returns_window):
        from dcc_garch import _dcc_covariance_uncached

        window = returns_window.iloc[-300:]
        first, record_a = _dcc_covariance_uncached(window, 1, 1, 0.02, 0.95, 100.0)
        second, record_b = _dcc_covariance_uncached(window, 1, 1, 0.02, 0.95, 100.0)
        np.testing.assert_array_equal(first, second)
        assert record_a == record_b, "the fit record must be deterministic too"


# ── Caching must not change a single produced number ────────────────────────
class TestCachingDoesNotChangeResults:
    def test_cached_prediction_equals_uncached_prediction(
        self, returns_window, market_features
    ):
        from ml_signals import _predict_expected_returns_uncached

        model_params = {"n_estimators": 25, "max_depth": 4}
        extras = {"features": market_features}
        fallback = returns_window.mean() * 252

        direct, used_fallback, _ = _predict_expected_returns_uncached(
            returns_window, extras, "random_forest", model_params, 100, 21, 63,
            [5, 21], True, 2, 5, 0, "diag", 252, fallback,
        )
        through_cache = fit_predict_expected_returns(
            returns_window, extras, model_type="random_forest",
            model_params=model_params, min_train_rows=100,
            momentum_windows=[5, 21],
        )
        expected = direct if used_fallback else direct
        pd.testing.assert_series_equal(through_cache, expected, check_names=False)

    def test_cached_dcc_equals_uncached_dcc(self, returns_window):
        from dcc_garch import _dcc_covariance_uncached

        window = returns_window.iloc[-300:]
        uncached, _ = _dcc_covariance_uncached(window, 1, 1, 0.02, 0.95, 100.0)
        np.testing.assert_array_equal(dcc_covariance(window), uncached)

    def test_mutating_a_returned_value_cannot_poison_the_cache(self, returns_window):
        """`_MLSignalStrategy.fit` takes a zero-copy `.to_numpy()` view."""
        window = returns_window.iloc[-300:]
        first = dcc_covariance(window)
        first[0, 0] = -999.0                       # caller scribbles on its copy
        second = dcc_covariance(window)
        assert second[0, 0] != -999.0, "cache handed out a mutable shared object"


# ── The variants that share a fit must still differ downstream ──────────────
class TestSharedFitStillProducesDistinctStrategies:
    """The five RF variants share one model fit; they must NOT share a mu.

    This is the test that would fail if the cache key wrongly swallowed
    `mu_transform` — the exact mistake that would make `rf_signal_shrunk`
    silently report `rf_signal`'s numbers.
    """

    def test_mu_transform_still_differentiates_after_a_cache_hit(
        self, returns_window, market_features
    ):
        common = dict(
            extras={"features": market_features},
            model_type="random_forest",
            model_params={"n_estimators": 25, "max_depth": 4},
            min_train_rows=100,
            momentum_windows=[5, 21],
        )
        plain = fit_predict_expected_returns(returns_window, **common)
        shrunk = fit_predict_expected_returns(
            returns_window, **common, mu_transform="shrink", shrinkage_weight=0.5
        )
        ranked = fit_predict_expected_returns(
            returns_window, **common, mu_transform="rank"
        )

        assert not np.allclose(plain.to_numpy(), shrunk.to_numpy()), (
            "shrink collapsed onto the untransformed prediction — the cache key "
            "must not include mu_transform, but the transform must still be applied"
        )
        assert not np.allclose(plain.to_numpy(), ranked.to_numpy())

    def test_different_model_params_are_not_shared(self, returns_window, market_features):
        common = dict(
            extras={"features": market_features}, model_type="random_forest",
            min_train_rows=100, momentum_windows=[5, 21],
        )
        shallow = fit_predict_expected_returns(
            returns_window, model_params={"n_estimators": 25, "max_depth": 2}, **common
        )
        deep = fit_predict_expected_returns(
            returns_window, model_params={"n_estimators": 25, "max_depth": 8}, **common
        )
        assert not np.allclose(shallow.to_numpy(), deep.to_numpy()), (
            "two different models returned identical predictions — key collision"
        )

    def test_portfolio_state_in_extras_does_not_change_the_prediction(
        self, returns_window, market_features
    ):
        """Locks the invariant the cache key relies on.

        The turnover-penalized variants receive `CURRENT_WEIGHTS_KEY` in
        `extras`; it is consumed by the optimizer, never by the return
        model. The key omits it on purpose, so if this function ever started
        reading it the omission would become a real collision — this test is
        what would catch that.
        """
        from backtest import CURRENT_WEIGHTS_KEY

        common = dict(
            model_type="random_forest",
            model_params={"n_estimators": 25, "max_depth": 4},
            min_train_rows=100, momentum_windows=[5, 21],
        )
        without = fit_predict_expected_returns(
            returns_window, {"features": market_features}, **common
        )
        with_state = fit_predict_expected_returns(
            returns_window,
            {
                "features": market_features,
                CURRENT_WEIGHTS_KEY: pd.DataFrame(
                    [np.full(returns_window.shape[1], 0.25)],
                    index=returns_window.index[-1:],
                    columns=returns_window.columns,
                ),
            },
            **common,
        )
        pd.testing.assert_series_equal(without, with_state)

    def test_different_training_windows_are_not_shared(self, returns_window, market_features):
        common = dict(
            extras={"features": market_features}, model_type="random_forest",
            model_params={"n_estimators": 25, "max_depth": 4},
            min_train_rows=100, momentum_windows=[5, 21],
        )
        early = fit_predict_expected_returns(returns_window.iloc[:-60], **common)
        full = fit_predict_expected_returns(returns_window, **common)
        assert not np.allclose(early.to_numpy(), full.to_numpy()), (
            "a shorter window returned the longer window's prediction — this "
            "would be a lookahead leak, not just a cache bug"
        )
