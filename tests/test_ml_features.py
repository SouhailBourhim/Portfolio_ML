"""Tests for Phase 3 causal ML feature engineering."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml_features import (
    ML_CORE_FEATURES,
    average_pairwise_rolling_correlation,
    build_lagged_macro_signals,
    build_ml_feature_set,
    build_return_features,
    run_phase3,
)


def _small_returns(periods: int = 120, assets: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    index = pd.bdate_range("2020-01-01", periods=periods, name="Date")
    values = rng.normal(0.0002, 0.01, size=(periods, assets))
    return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(assets)])


def _config() -> dict:
    return {
        "volatility_short_window": 5,
        "volatility_long_window": 10,
        "correlation_window": 10,
        "correlation_min_periods": 5,
        "macro_lag_days": 1,
    }


class TestReturnFeatures:
    def test_future_observations_do_not_change_past_features(self):
        original = _small_returns()
        changed = original.copy()
        cutoff = original.index[89]
        changed.loc[changed.index > cutoff] = 5.0

        first = build_return_features(original, 5, 10, 10, 5)
        second = build_return_features(changed, 5, 10, 10, 5)

        pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])

    def test_short_volatility_matches_manual_calculation(self):
        returns = pd.DataFrame(
            {
                "A": [0.01, 0.02, -0.01, 0.00],
                "B": [0.03, 0.00, 0.01, -0.02],
            },
            index=pd.bdate_range("2020-01-01", periods=4, name="Date"),
        )
        result = build_return_features(returns, 3, 3, 3, 3)
        market = returns.mean(axis=1)
        expected = market.iloc[:3].std(ddof=1) * np.sqrt(252.0)
        assert result.iloc[2]["MARKET_VOL_SHORT"] == pytest.approx(expected)

    def test_average_pairwise_correlation_matches_manual_calculation(self):
        returns = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [2.0, 4.0, 6.0, 8.0],
                "C": [4.0, 3.0, 2.0, 1.0],
            },
            index=pd.bdate_range("2020-01-01", periods=4, name="Date"),
        )
        result = average_pairwise_rolling_correlation(returns, window=3, min_periods=3)
        corr = returns.iloc[:3].corr().to_numpy()
        expected = corr[np.triu_indices(3, k=1)].mean()
        assert result.iloc[2] == pytest.approx(expected)

    def test_output_has_required_columns(self):
        result = build_return_features(_small_returns(), 5, 10, 10, 5)
        assert set(ML_CORE_FEATURES).issubset(result.columns)

    def test_rejects_unsorted_index(self):
        returns = _small_returns().sort_index(ascending=False)
        with pytest.raises(ValueError, match="sorted"):
            build_return_features(returns, 5, 10, 10, 5)


class TestMacroSignals:
    def test_macro_signal_is_differenced_then_lagged(self):
        index = pd.bdate_range("2020-01-01", periods=5, name="Date")
        macro = pd.DataFrame({"VIX": [10.0, 11.0, 13.0, 16.0, 20.0]}, index=index)
        result = build_lagged_macro_signals(macro, index, lag_days=1)
        assert result.loc[index[2], "VIX_DIFF_L1"] == pytest.approx(1.0)
        assert result.loc[index[3], "VIX_DIFF_L1"] == pytest.approx(2.0)

    def test_rejects_zero_lag(self):
        index = pd.bdate_range("2020-01-01", periods=5, name="Date")
        macro = pd.DataFrame({"VIX": range(5)}, index=index)
        with pytest.raises(ValueError, match="lookahead"):
            build_lagged_macro_signals(macro, index, lag_days=0)

    def test_interior_nan_from_multi_source_calendars_is_forward_filled(self):
        # Two macro sources on different calendars, concatenated, leave interior
        # NaN holes on dates only one source published. Those must be carried
        # forward (causal), not scattered through the differenced signal — else
        # a Phase 4 fit at any date could hit a NaN macro feature.
        index = pd.bdate_range("2020-01-01", periods=6, name="Date")
        macro = pd.DataFrame(
            {"VIX": [10.0, np.nan, 12.0, np.nan, 15.0, 16.0]},  # interior holes
            index=index,
        )
        result = build_lagged_macro_signals(macro, index, lag_days=1)
        # No interior NaN survives after the first valid observation
        signal = result["VIX_DIFF_L1"]
        assert not signal.loc[signal.first_valid_index():].isna().any()

    def test_alignment_does_not_backfill_before_first_macro_date(self):
        returns_index = pd.bdate_range("2020-01-01", periods=6, name="Date")
        macro = pd.DataFrame(
            {"VIX": [10.0, 12.0, 13.0]},
            index=returns_index[3:],
        )
        result = build_lagged_macro_signals(macro, returns_index, lag_days=1)
        assert result.loc[returns_index[:4], "VIX_DIFF_L1"].isna().all()

    def test_future_macro_does_not_change_past_signals(self):
        index = pd.bdate_range("2020-01-01", periods=20, name="Date")
        macro = pd.DataFrame({"VIX": np.arange(20, dtype=float)}, index=index)
        changed = macro.copy()
        cutoff = index[12]
        changed.loc[changed.index > cutoff, "VIX"] = 9999.0
        first = build_lagged_macro_signals(macro, index, lag_days=1)
        second = build_lagged_macro_signals(changed, index, lag_days=1)
        pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])


class TestFeatureSetAndPipeline:
    def test_feature_set_is_deterministic_and_has_no_infinity(self):
        returns = _small_returns()
        macro = pd.DataFrame(
            {"VIX": np.linspace(10, 30, len(returns))},
            index=returns.index,
        )
        first = build_ml_feature_set(returns, macro, _config())
        second = build_ml_feature_set(returns, macro, _config())
        pd.testing.assert_frame_equal(first, second)
        assert np.isfinite(first[ML_CORE_FEATURES].to_numpy()).all()

    def test_run_phase3_writes_both_universes_and_manifest(self, tmp_path):
        returns = _small_returns()
        gold = tmp_path / "data" / "gold"
        bronze = tmp_path / "data" / "bronze"
        gold.mkdir(parents=True)
        bronze.mkdir(parents=True)

        returns.to_parquet(gold / "log_returns_etf.parquet")
        returns.iloc[20:].to_parquet(gold / "log_returns.parquet")
        pd.DataFrame(
            {"VIX": np.linspace(10, 30, len(returns))},
            index=returns.index,
        ).to_parquet(bronze / "raw_macro.parquet")

        config = {
            "backtest": {
                "universes": {
                    "etf_2017": "data/gold/log_returns_etf.parquet",
                    "full_2021": "data/gold/log_returns.parquet",
                }
            },
            "ml_features": {
                **_config(),
                "outputs": {
                    "etf_2017": "data/gold/ml_features_etf.parquet",
                    "full_2021": "data/gold/ml_features_full.parquet",
                },
                "manifest_path": "data/gold/ml_features_manifest.json",
            },
        }

        results = run_phase3(config=config, project_root=tmp_path)

        assert set(results) == {"etf_2017", "full_2021"}
        assert (gold / "ml_features_etf.parquet").exists()
        assert (gold / "ml_features_full.parquet").exists()
        manifest = json.loads((gold / "ml_features_manifest.json").read_text())
        assert manifest["global_standardization"] is False
        assert set(manifest["universes"]) == {"etf_2017", "full_2021"}

    def test_manifest_records_feature_warmup(self, tmp_path):
        # The warm-up must be reported so Phase 4 can guard min_train_days
        # against it instead of discovering NaN features at fit time.
        returns = _small_returns()
        gold = tmp_path / "data" / "gold"
        bronze = tmp_path / "data" / "bronze"
        gold.mkdir(parents=True)
        bronze.mkdir(parents=True)
        returns.to_parquet(gold / "log_returns_etf.parquet")
        returns.iloc[20:].to_parquet(gold / "log_returns.parquet")
        # macro that starts late → a genuine leading-NaN warm-up on its column
        late_macro = pd.DataFrame(
            {"VIX": np.linspace(10, 30, len(returns) - 30)},
            index=returns.index[30:],
        )
        late_macro.to_parquet(bronze / "raw_macro.parquet")

        config = {
            "backtest": {"universes": {
                "etf_2017": "data/gold/log_returns_etf.parquet",
                "full_2021": "data/gold/log_returns.parquet",
            }},
            "ml_features": {**_config(), "outputs": {
                "etf_2017": "data/gold/ml_features_etf.parquet",
                "full_2021": "data/gold/ml_features_full.parquet",
            }, "manifest_path": "data/gold/ml_features_manifest.json"},
        }
        run_phase3(config=config, project_root=tmp_path)
        manifest = json.loads((gold / "ml_features_manifest.json").read_text())

        assert "warmup_policy" in manifest
        etf = manifest["universes"]["etf_2017"]
        assert "max_leading_nan" in etf and "leading_nan_by_column" in etf
        # core features carry no leading NaN in the output; the late macro does
        assert etf["leading_nan_by_column"]["MARKET_RETURN"] == 0
        # The late-starting macro series MUST show a strictly positive warm-up and
        # must drive max_leading_nan — otherwise warm-up accounting has silently
        # stopped treating late macro history as leading NaN.
        assert etf["leading_nan_by_column"]["VIX_DIFF_L1"] > 0
        assert etf["max_leading_nan"] >= etf["leading_nan_by_column"]["VIX_DIFF_L1"]
        assert etf["max_leading_nan"] > 0
