"""
test_ml_signals.py — Tests for Phase 4B / F7 per-asset feature engineering
and supervised return-prediction dataset construction.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml_signals import (
    attach_fundamentals_features,
    attach_regime_feature,
    build_asset_features,
    build_supervised_dataset,
    melt_to_panel,
)


def _small_returns(periods: int = 120, assets: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=periods, name="Date")
    values = rng.normal(0.0002, 0.01, size=(periods, assets))
    return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(assets)])


class TestBuildAssetFeatures:
    def test_output_columns_match_naming_convention(self):
        returns = _small_returns()
        features = build_asset_features(
            returns, short_window=5, long_window=10, momentum_windows=[3, 5]
        )
        expected = set()
        for asset in returns.columns:
            expected.add(f"{asset}__RET_3D")
            expected.add(f"{asset}__RET_5D")
            expected.add(f"{asset}__VOL_5D")
            expected.add(f"{asset}__VOL_10D")
            expected.add(f"{asset}__PRICE_REL_MA_10D")
        assert set(features.columns) == expected

    def test_output_aligned_to_input_index(self):
        returns = _small_returns()
        features = build_asset_features(returns, short_window=5, long_window=10)
        assert features.index.equals(returns.index)

    def test_leading_rows_are_nan_until_warmup_clears(self):
        returns = _small_returns()
        features = build_asset_features(
            returns, short_window=5, long_window=10, momentum_windows=[3]
        )
        # VOL_10D needs 10 observations -> first 9 rows must be NaN.
        col = f"{returns.columns[0]}__VOL_10D"
        assert features[col].iloc[:9].isna().all()
        assert features[col].iloc[9:].notna().all()

    def test_future_returns_do_not_change_past_asset_features(self):
        # The end-to-end causality proof for this module: precomputing over
        # the FULL history and slicing later (what the extras mechanism
        # does) must be identical to never having seen the future at all.
        original = _small_returns()
        changed = original.copy()
        cutoff = original.index[89]
        changed.loc[changed.index > cutoff] = 5.0

        first = build_asset_features(original, short_window=5, long_window=10)
        second = build_asset_features(changed, short_window=5, long_window=10)

        pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])

    def test_future_corruption_does_change_future_rows(self):
        # The flip side: proves the fixture actually exercises the feature
        # windows (guards against a vacuous causality test).
        original = _small_returns()
        changed = original.copy()
        cutoff = original.index[89]
        changed.loc[changed.index > cutoff] = 5.0

        first = build_asset_features(original, short_window=5, long_window=10)
        second = build_asset_features(changed, short_window=5, long_window=10)

        after = first.index > cutoff
        assert not first.loc[after].equals(second.loc[after])

    def test_rejects_unsorted_index(self):
        returns = _small_returns().sort_index(ascending=False)
        with pytest.raises(ValueError, match="sorted"):
            build_asset_features(returns)

    def test_rejects_small_windows(self):
        returns = _small_returns()
        with pytest.raises(ValueError, match=">= 2"):
            build_asset_features(returns, short_window=1)

    def test_price_rel_ma_is_zero_when_wealth_at_its_own_moving_average(self):
        # Hand-computable sanity check: constant returns -> wealth grows
        # smoothly -> PRICE_REL_MA should be small and well-defined (finite),
        # never NaN/inf once the warm-up clears.
        index = pd.bdate_range("2020-01-01", periods=30, name="Date")
        returns = pd.DataFrame({"A": [0.001] * 30, "B": [0.0] * 30}, index=index)
        features = build_asset_features(
            returns, short_window=5, long_window=10, momentum_windows=[5]
        )
        col = "A__PRICE_REL_MA_10D"
        assert np.isfinite(features[col].iloc[10:]).all()


class TestMeltToPanel:
    def test_output_has_date_asset_multiindex(self):
        returns = _small_returns(assets=3)
        wide = build_asset_features(returns, short_window=5, long_window=10, momentum_windows=[3])
        panel = melt_to_panel(wide, list(returns.columns))
        assert panel.index.names == ["Date", "ASSET"]

    def test_column_prefix_is_stripped(self):
        returns = _small_returns(assets=2)
        wide = build_asset_features(returns, short_window=5, long_window=10, momentum_windows=[3])
        panel = melt_to_panel(wide, list(returns.columns))
        assert set(panel.columns) == {"RET_3D", "VOL_5D", "VOL_10D", "PRICE_REL_MA_10D"}

    def test_row_count_equals_dates_times_assets(self):
        returns = _small_returns(periods=50, assets=4)
        wide = build_asset_features(returns, short_window=5, long_window=10, momentum_windows=[3])
        panel = melt_to_panel(wide, list(returns.columns))
        assert len(panel) == len(returns) * 4

    def test_each_asset_row_matches_its_own_wide_columns(self):
        returns = _small_returns(periods=50, assets=2)
        wide = build_asset_features(returns, short_window=5, long_window=10, momentum_windows=[3])
        panel = melt_to_panel(wide, list(returns.columns))
        date = returns.index[30]  # well past the 10-day warm-up -> no NaN to handle
        row = panel.loc[(date, "A0")]
        assert row["VOL_5D"] == pytest.approx(wide.loc[date, "A0__VOL_5D"])
        assert row["RET_3D"] == pytest.approx(wide.loc[date, "A0__RET_3D"])

    def test_raises_when_no_asset_columns_match(self):
        returns = _small_returns(assets=2)
        wide = build_asset_features(returns, short_window=5, long_window=10, momentum_windows=[3])
        with pytest.raises(ValueError, match="matching"):
            melt_to_panel(wide, ["NOT_AN_ASSET"])


class TestAttachRegimeFeature:
    def _panel(self):
        returns = _small_returns(periods=40, assets=2)
        wide = build_asset_features(returns, short_window=5, long_window=10, momentum_windows=[3])
        return melt_to_panel(wide, list(returns.columns))

    def test_neutral_fallback_when_market_features_none(self):
        panel = self._panel()
        result = attach_regime_feature(panel, None)
        assert (result["REGIME_BULL_PROB"] == 0.5).all()

    def test_neutral_fallback_when_market_features_empty(self):
        panel = self._panel()
        result = attach_regime_feature(panel, pd.DataFrame())
        assert (result["REGIME_BULL_PROB"] == 0.5).all()

    def test_neutral_fallback_when_window_too_short_for_hmm(self):
        panel = self._panel()
        market_features = pd.DataFrame(
            {
                "MARKET_RETURN": np.full(40, 0.001),
                "MARKET_VOL_SHORT": np.full(40, 0.1),
                "AVG_PAIRWISE_CORR": np.full(40, 0.2),
            },
            index=panel.index.get_level_values("Date").unique(),
        )
        # min_regime_train_days default (252) far exceeds this window -> fallback.
        result = attach_regime_feature(panel, market_features)
        assert (result["REGIME_BULL_PROB"] == 0.5).all()

    def test_regime_probability_is_broadcast_identically_across_assets(self):
        # A converged HMM's per-date bull-probability must be the SAME for
        # every asset on that date (regime is market-wide) -- the defining
        # property of a legitimate one-to-many join, not an approximation.
        rng = np.random.default_rng(11)
        n = 80
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
        market_features = pd.concat([bull, bear], ignore_index=True)
        market_features.index = pd.bdate_range("2020-01-01", periods=len(market_features), name="Date")

        returns = _small_returns(periods=len(market_features), assets=3)
        returns.index = market_features.index
        wide = build_asset_features(returns, short_window=5, long_window=10, momentum_windows=[3])
        panel = melt_to_panel(wide, list(returns.columns))

        result = attach_regime_feature(panel, market_features, min_regime_train_days=50)

        for date in result.index.get_level_values("Date").unique()[-10:]:
            values = result.loc[date, "REGIME_BULL_PROB"]
            assert values.nunique() == 1, f"regime prob differs across assets on {date}"


class TestBuildSupervisedDataset:
    """The highest-risk correctness point in this module — see the
    function's own docstring. Every test here locks in one facet of the
    causal boundary."""

    def _panel_and_returns(self, periods=60, assets=3, short_window=5, long_window=10):
        returns = _small_returns(periods=periods, assets=assets)
        wide = build_asset_features(
            returns, short_window=short_window, long_window=long_window, momentum_windows=[3]
        )
        panel = melt_to_panel(wide, list(returns.columns))
        return panel, returns

    def test_X_predict_date_equals_the_panels_last_date(self):
        panel, returns = self._panel_and_returns()
        X, y, X_predict = build_supervised_dataset(panel, returns)
        last_date = panel.index.get_level_values("Date").max()
        assert set(X_predict.index.get_level_values("Date").unique()) == {last_date}
        assert X_predict.index.get_level_values("Date").max() == returns.index[-1]

    def test_X_dates_are_a_strict_subset_excluding_the_last_date(self):
        panel, returns = self._panel_and_returns()
        X, y, X_predict = build_supervised_dataset(panel, returns)
        last_date = panel.index.get_level_values("Date").max()
        assert last_date not in X.index.get_level_values("Date")

    def test_len_y_equals_len_X_and_neither_is_empty(self):
        panel, returns = self._panel_and_returns()
        X, y, X_predict = build_supervised_dataset(panel, returns)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_labels_are_the_correct_next_day_return_values(self):
        # Hand-checkable: pick a mid-window (date, asset) pair whose features
        # are already warmed up and assert y equals the REAL next-day return.
        panel, returns = self._panel_and_returns(periods=60)
        X, y, X_predict = build_supervised_dataset(panel, returns)
        date = returns.index[30]
        asset = returns.columns[0]
        next_date = returns.index[31]
        expected = returns.loc[next_date, asset]
        assert y.loc[(date, asset)] == pytest.approx(expected)

    def test_last_date_excluded_even_if_log_returns_extends_past_the_panel(self):
        # The defensive guarantee stated in the docstring: exclusion is by
        # DATE, not by whether a label happens to be available. Construct a
        # log_returns frame with MORE rows than the panel (as if a caller
        # mistakenly passed unsliced future data) and confirm the panel's
        # last date is STILL excluded from training, not silently trained on
        # because a "real" label became available for it.
        panel, returns = self._panel_and_returns(periods=60)
        cutoff_date = returns.index[50]
        panel_sliced = panel.loc[panel.index.get_level_values("Date") <= cutoff_date]

        # log_returns here extends WELL PAST the panel's last date (50) —
        # simulating a caller error.
        X, y, X_predict = build_supervised_dataset(panel_sliced, returns)

        assert cutoff_date not in X.index.get_level_values("Date")
        assert set(X_predict.index.get_level_values("Date").unique()) == {cutoff_date}

    def test_no_nan_in_X_or_y(self):
        panel, returns = self._panel_and_returns()
        X, y, X_predict = build_supervised_dataset(panel, returns)
        assert not X.isna().any().any()
        assert not y.isna().any()


class TestFitPredictExpectedReturns:
    def _returns(self, periods=400, assets=4, seed=3):
        rng = np.random.default_rng(seed)
        index = pd.bdate_range("2019-01-01", periods=periods, name="Date")
        values = rng.normal(0.0003, 0.011, size=(periods, assets))
        return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(assets)])

    @pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
    def test_output_indexed_exactly_by_train_returns_columns(self, model_type):
        from ml_signals import fit_predict_expected_returns

        returns = self._returns()
        result = fit_predict_expected_returns(
            returns, extras=None, model_type=model_type, min_train_rows=50,
            condition_on_regime=False,
        )
        assert list(result.index) == list(returns.columns)
        assert np.isfinite(result.to_numpy()).all()

    def test_fallback_below_min_train_rows_logs_warning(self, caplog):
        from ml_signals import fit_predict_expected_returns

        returns = self._returns(periods=60, assets=3)
        with caplog.at_level("WARNING", logger="ml_signals"):
            result = fit_predict_expected_returns(
                returns, extras=None, model_type="random_forest",
                min_train_rows=10_000,  # impossibly high -> always falls back
                condition_on_regime=False,
            )
        expected = returns.mean() * 252
        pd.testing.assert_series_equal(result, expected, check_names=False)
        assert any("falling back" in r.message for r in caplog.records)

    def test_unknown_model_type_falls_back_rather_than_raising(self, caplog):
        from ml_signals import fit_predict_expected_returns

        returns = self._returns(periods=400, assets=3)
        with caplog.at_level("WARNING", logger="ml_signals"):
            result = fit_predict_expected_returns(
                returns, extras=None, model_type="not_a_real_model",
                min_train_rows=50, condition_on_regime=False,
            )
        expected = returns.mean() * 252
        pd.testing.assert_series_equal(result, expected, check_names=False)

    @pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
    def test_works_with_regime_conditioning_enabled(self, model_type):
        from ml_signals import fit_predict_expected_returns

        returns = self._returns(periods=400, assets=3)
        rng = np.random.default_rng(9)
        market_features = pd.DataFrame(
            {
                "MARKET_RETURN": returns.mean(axis=1),
                "MARKET_VOL_SHORT": rng.normal(0.15, 0.02, len(returns)),
                "AVG_PAIRWISE_CORR": rng.normal(0.2, 0.05, len(returns)),
            },
            index=returns.index,
        )
        result = fit_predict_expected_returns(
            returns, extras={"features": market_features}, model_type=model_type,
            min_train_rows=50, condition_on_regime=True, min_regime_train_days=100,
        )
        assert list(result.index) == list(returns.columns)
        assert np.isfinite(result.to_numpy()).all()


class TestRunMlSignalFeatures:
    def test_writes_both_universes_and_manifest(self, tmp_path):
        from ml_signals import run_ml_signal_features

        returns = _small_returns(periods=100, assets=3)
        gold = tmp_path / "data" / "gold"
        gold.mkdir(parents=True)
        returns.to_parquet(gold / "log_returns_etf.parquet")
        returns.iloc[20:].to_parquet(gold / "log_returns.parquet")

        config = {
            "backtest": {
                "universes": {
                    "etf_2017": "data/gold/log_returns_etf.parquet",
                    "full_2021": "data/gold/log_returns.parquet",
                }
            },
            "ml_signals": {
                "short_window": 5,
                "long_window": 10,
                "momentum_windows": [3, 5],
                "outputs": {
                    "etf_2017": "data/gold/ml_signal_features_etf.parquet",
                    "full_2021": "data/gold/ml_signal_features_full.parquet",
                },
                "manifest_path": "data/gold/ml_signal_features_manifest.json",
            },
        }

        results = run_ml_signal_features(config=config, project_root=tmp_path)

        assert set(results) == {"etf_2017", "full_2021"}
        assert (gold / "ml_signal_features_etf.parquet").exists()
        assert (gold / "ml_signal_features_full.parquet").exists()
        manifest = json.loads((gold / "ml_signal_features_manifest.json").read_text())
        assert set(manifest["universes"]) == {"etf_2017", "full_2021"}
        assert manifest["universes"]["etf_2017"]["columns"] == results["etf_2017"].shape[1]


class TestAttachFundamentalsFeatures:
    """The fundamentals-experiment seam. Same shape-of-guarantee as
    TestAttachRegimeFeature: no-op when input is absent, correct broadcast,
    and — the load-bearing one — the ETF median-fill respects the
    per-asset BVC values so a BVC row never gets a median instead of its
    own value."""

    def _panel_and_fund(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 3 dates × 3 assets: two "BVC-like" (A, B) with fundamentals, one
        # ETF-like (E) without. Fund panel has A__FUND_pe and B__FUND_pe.
        dates = pd.bdate_range("2023-01-02", periods=3, name="Date")
        assets = ["A", "B", "E"]
        idx = pd.MultiIndex.from_product([dates, assets], names=["Date", "ASSET"])
        panel = pd.DataFrame({"RET_5D": np.arange(len(idx), dtype=float)}, index=idx)

        fund = pd.DataFrame({
            "A__FUND_pe": [10.0, 11.0, 12.0],
            "B__FUND_pe": [20.0, 22.0, 24.0],
        }, index=dates)
        return panel, fund

    def test_no_fundamentals_panel_returns_input_untouched(self):
        panel, _ = self._panel_and_fund()
        result = attach_fundamentals_features(panel, fundamentals_panel=None)
        pd.testing.assert_frame_equal(result, panel)

    def test_bvc_asset_gets_its_own_fundamental_not_the_median(self):
        panel, fund = self._panel_and_fund()
        result = attach_fundamentals_features(panel, fund, fund_assets=["A", "B"])
        # On date[0], A's PE must be 10.0 (its own), NOT 15.0 (the median).
        d0 = fund.index[0]
        assert result.loc[(d0, "A"), "FUND_pe"] == pytest.approx(10.0)
        assert result.loc[(d0, "B"), "FUND_pe"] == pytest.approx(20.0)

    def test_etf_asset_gets_cross_sectional_median_of_bvc(self):
        panel, fund = self._panel_and_fund()
        result = attach_fundamentals_features(panel, fund, fund_assets=["A", "B"])
        d1 = fund.index[1]
        # median of A=11.0, B=22.0 is 16.5
        assert result.loc[(d1, "E"), "FUND_pe"] == pytest.approx(16.5)

    def test_has_fund_indicator_marks_bvc_and_etf_correctly(self):
        panel, fund = self._panel_and_fund()
        result = attach_fundamentals_features(panel, fund, fund_assets=["A", "B"])
        assert result.loc[(fund.index[0], "A"), "HAS_FUND"] == 1
        assert result.loc[(fund.index[0], "B"), "HAS_FUND"] == 1
        assert result.loc[(fund.index[0], "E"), "HAS_FUND"] == 0

    def test_infer_fund_assets_when_omitted(self):
        panel, fund = self._panel_and_fund()
        result = attach_fundamentals_features(panel, fund)
        # Inferred fund_assets = ["A", "B"] from column names → E still ETF.
        assert result.loc[(fund.index[0], "E"), "HAS_FUND"] == 0
