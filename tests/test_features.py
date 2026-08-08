"""
test_features.py — Tests for Gold layer feature engineering.
"""

import json

import numpy as np
import pandas as pd
import pytest

import features
from features import build_macro_features, run_stationarity_tests, write_currency_manifest


class TestStationarity:
    def test_returns_dataframe_with_correct_index(self, synthetic_log_returns):
        result = run_stationarity_tests(synthetic_log_returns)
        assert set(result.index) == set(synthetic_log_returns.columns)

    def test_conclusion_column_exists(self, synthetic_log_returns):
        result = run_stationarity_tests(synthetic_log_returns)
        assert "conclusion" in result.columns

    def test_valid_conclusions(self, synthetic_log_returns):
        result = run_stationarity_tests(synthetic_log_returns)
        valid = {"STATIONARY", "NON-STATIONARY", "AMBIGUOUS"}
        assert set(result["conclusion"]).issubset(valid)

    def test_gbm_returns_are_stationary(self, synthetic_log_returns):
        result = run_stationarity_tests(synthetic_log_returns)
        # GBM-simulated returns should be stationary
        assert (result["conclusion"] == "STATIONARY").all()


class TestBuildMacroFeatures:
    def test_output_shape(self, synthetic_macro, synthetic_log_returns):
        result = build_macro_features(synthetic_macro, synthetic_log_returns.index)
        assert result.shape[1] == synthetic_macro.shape[1]

    def test_lag_prevents_same_day_data(self, synthetic_macro, synthetic_log_returns):
        # With lag=1, row 0 of features should be NaN (dropped)
        # and features should be 1 day behind macro
        result = build_macro_features(synthetic_macro, synthetic_log_returns.index, lag_days=1)
        # First available feature date must be strictly after the first macro date
        assert result.index[0] > synthetic_log_returns.index[0]

    def test_rejects_zero_lag(self, synthetic_macro, synthetic_log_returns):
        with pytest.raises(ValueError, match="lookahead"):
            build_macro_features(synthetic_macro, synthetic_log_returns.index, lag_days=0)

    def test_features_are_standardized(self, synthetic_macro, synthetic_log_returns):
        result = build_macro_features(synthetic_macro, synthetic_log_returns.index)
        # After standardization, values should be roughly in [-5, 5]
        assert result.abs().max().max() < 10


class TestCurrencyManifest:
    """The Gold layer must state its own numéraire.

    AGENTS.md §7 makes Gold the only layer modelling code reads, so a currency
    recorded solely in a Silver report is invisible to every consumer that
    matters. A returns matrix carries no unit of its own.
    """

    @pytest.fixture()
    def dirs(self, tmp_path, monkeypatch):
        silver, gold = tmp_path / "silver", tmp_path / "gold"
        silver.mkdir()
        gold.mkdir()
        monkeypatch.setattr(features, "SILVER_DIR", silver)
        monkeypatch.setattr(features, "GOLD_DIR", gold)
        return silver, gold

    def _write_report(self, silver, name, currency):
        payload = {"n_trading_days": 10}
        if currency is not None:
            payload["currency"] = currency
        (silver / name).write_text(json.dumps(payload))

    def test_it_copies_the_currency_block_from_both_silver_reports(self, dirs):
        silver, gold = dirs
        block = {
            "converted": True,
            "base_currency": "MAD",
            "fx_series": "USDMAD",
            "hedge_status": "unhedged",
        }
        self._write_report(silver, "validation_report.json", block)
        self._write_report(silver, "validation_report_log_returns_etf.json", block)

        manifest = write_currency_manifest()

        assert (gold / "currency_manifest.json").exists()
        assert set(manifest["universes"]) == {"full_2021", "etf_2017"}
        for universe in ("full_2021", "etf_2017"):
            assert manifest["universes"][universe]["base_currency"] == "MAD"
            assert manifest["universes"][universe]["hedge_status"] == "unhedged"

    def test_a_silver_report_predating_the_correction_is_flagged_not_omitted(self, dirs):
        """Absence of the key must not read as 'not applicable' — an artifact
        with no numéraire is exactly the state being corrected."""
        silver, _ = dirs
        self._write_report(silver, "validation_report.json", None)
        manifest = write_currency_manifest()
        assert manifest["universes"]["full_2021"]["converted"] is None
        assert "predates" in manifest["universes"]["full_2021"]["note"]

    def test_it_is_copied_rather_than_recomputed(self, dirs):
        """Recomputing would create a second source of truth able to disagree
        with the artifact it describes — the §17.1 failure class."""
        silver, _ = dirs
        sentinel = {"converted": True, "base_currency": "SENTINEL"}
        self._write_report(silver, "validation_report.json", sentinel)
        manifest = write_currency_manifest()
        assert manifest["universes"]["full_2021"]["base_currency"] == "SENTINEL"
