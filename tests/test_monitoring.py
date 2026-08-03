"""test_monitoring.py — the drift metrics must be quiet on stable data and loud on shifted data.

Both halves matter equally. A detector that never fires is useless; one that
fires on unchanged data is worse, because it trains its readers to ignore it.
Every metric here is therefore tested against BOTH a stable and a shifted
version of the same synthetic input.

Offline and seeded throughout — no artifact and no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import monitoring


RNG = np.random.default_rng(0)
STABLE_A = RNG.normal(0.0, 1.0, 4000)
STABLE_B = np.random.default_rng(1).normal(0.0, 1.0, 4000)
SHIFTED_LOCATION = np.random.default_rng(2).normal(1.5, 1.0, 4000)
SHIFTED_SCALE = np.random.default_rng(3).normal(0.0, 3.0, 4000)


class TestPopulationStabilityIndex:
    def test_identical_distributions_score_near_zero(self):
        psi = monitoring.population_stability_index(STABLE_A, STABLE_A)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_two_samples_of_the_same_law_stay_in_the_stable_band(self):
        psi = monitoring.population_stability_index(STABLE_A, STABLE_B)
        assert psi < monitoring.PSI_MODERATE
        assert monitoring.interpret_psi(psi) == "stable"

    def test_a_location_shift_is_flagged_as_significant(self):
        psi = monitoring.population_stability_index(STABLE_A, SHIFTED_LOCATION)
        assert psi > monitoring.PSI_SIGNIFICANT
        assert monitoring.interpret_psi(psi) == "significant_shift"

    def test_a_scale_shift_is_flagged_even_with_the_mean_unchanged(self):
        """Variance drift is the failure mode a mean comparison misses."""
        assert abs(SHIFTED_SCALE.mean() - STABLE_A.mean()) < 0.2
        psi = monitoring.population_stability_index(STABLE_A, SHIFTED_SCALE)
        assert psi > monitoring.PSI_MODERATE

    def test_psi_increases_monotonically_with_the_size_of_the_shift(self):
        scores = [
            monitoring.population_stability_index(
                STABLE_A, np.random.default_rng(10 + i).normal(delta, 1.0, 4000)
            )
            for i, delta in enumerate((0.0, 0.5, 1.0, 2.0))
        ]
        assert scores == sorted(scores), f"PSI not monotone in shift size: {scores}"

    def test_bins_come_from_the_reference_not_the_pooled_data(self):
        """Binning on the union lets the evaluation window redefine its own test.

        A reference concentrated in [0, 1] against an evaluation concentrated in
        [10, 11] must score as a large shift. If edges were computed from the
        combined sample, both windows would fall into separate bins of a wide
        grid and the statistic could be made to look tame.
        """
        reference = np.random.default_rng(4).uniform(0.0, 1.0, 2000)
        evaluation = np.random.default_rng(5).uniform(10.0, 11.0, 2000)
        assert monitoring.population_stability_index(reference, evaluation) > 1.0

    def test_an_empty_window_scores_zero_rather_than_nan(self):
        """A NaN would propagate silently into a health report."""
        assert monitoring.population_stability_index([], STABLE_A) == 0.0
        assert monitoring.population_stability_index(STABLE_A, []) == 0.0

    def test_a_constant_reference_scores_zero_rather_than_dividing_by_zero(self):
        assert monitoring.population_stability_index([2.0] * 100, STABLE_A) == 0.0

    def test_an_empty_bin_stays_finite(self):
        """Without the epsilon floor a vacated bin sends PSI to infinity."""
        reference = np.concatenate([np.zeros(500), np.ones(500)])
        evaluation = np.zeros(500)
        psi = monitoring.population_stability_index(reference, evaluation)
        assert np.isfinite(psi) and psi > monitoring.PSI_SIGNIFICANT

    def test_nan_values_are_dropped_not_propagated(self):
        with_nans = np.concatenate([STABLE_A, np.full(50, np.nan)])
        assert np.isfinite(monitoring.population_stability_index(STABLE_A, with_nans))


class TestStoredReferenceSummary:
    """The baseline must be self-sufficient: no raw training data required."""

    def test_summary_stores_edges_not_observations(self):
        summary = monitoring.summarize_reference(STABLE_A)
        assert summary["count"] == len(STABLE_A)
        assert len(summary["edges"]) <= monitoring.DEFAULT_BINS + 1
        assert "values" not in summary and "observations" not in summary

    def test_psi_from_summary_agrees_with_psi_from_raw_reference(self):
        summary = monitoring.summarize_reference(STABLE_A)
        raw = monitoring.population_stability_index(STABLE_A, SHIFTED_LOCATION)
        stored = monitoring.psi_from_reference_summary(summary, SHIFTED_LOCATION)
        assert stored == pytest.approx(raw, abs=0.05)

    def test_psi_from_summary_is_quiet_on_stable_data(self):
        summary = monitoring.summarize_reference(STABLE_A)
        assert monitoring.psi_from_reference_summary(summary, STABLE_B) < monitoring.PSI_MODERATE

    def test_an_empty_summary_scores_zero(self):
        assert monitoring.psi_from_reference_summary({"count": 0, "edges": []}, STABLE_A) == 0.0


class TestCategoricalShift:
    def test_an_unchanged_regime_mix_is_stable(self):
        labels = ["bull"] * 70 + ["bear"] * 30
        result = monitoring.categorical_shift(labels, labels)
        assert result["interpretation"] == "stable"

    def test_a_regime_flip_is_flagged(self):
        reference = ["bull"] * 80 + ["bear"] * 20
        evaluation = ["bull"] * 20 + ["bear"] * 80
        result = monitoring.categorical_shift(reference, evaluation)
        assert result["interpretation"] == "significant_shift"
        assert result["evaluation_share"]["bear"] == pytest.approx(0.8)

    def test_a_label_absent_from_one_window_stays_finite(self):
        result = monitoring.categorical_shift(["bull", "bear"] * 50, ["bear"] * 100)
        assert np.isfinite(result["psi"])
        assert result["reference_share"]["bull"] == pytest.approx(0.5)


class TestHealthMetrics:
    def test_missingness_reports_the_fraction_per_column(self):
        frame = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan], "B": [1.0, 2.0, 3.0, 4.0]})
        result = monitoring.feature_missingness(frame)
        assert result["A"] == pytest.approx(0.5)
        assert result["B"] == 0.0

    def test_equal_weights_give_effective_n_equal_to_asset_count(self):
        result = monitoring.allocation_concentration(pd.Series([0.25] * 4))
        assert result["effective_n"] == pytest.approx(4.0)
        assert result["herfindahl"] == pytest.approx(0.25)

    def test_a_concentrated_book_reports_lower_effective_breadth(self):
        spread = monitoring.allocation_concentration(pd.Series([0.25] * 4))
        concentrated = monitoring.allocation_concentration(pd.Series([0.7, 0.1, 0.1, 0.1]))
        assert concentrated["effective_n"] < spread["effective_n"]
        assert concentrated["max_weight"] == pytest.approx(0.7)

    def test_cap_binding_uses_tolerance_because_slsqp_converges_not_lands(self):
        """A weight of 0.2499999996 is on the cap in every sense that matters."""
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-31"] * 4),
            "asset": list("ABCD"),
            "weight": [0.25 - 1e-9, 0.25, 0.3, 0.2],
        })
        result = monitoring.cap_binding_rate(frame, max_weight=0.25)
        assert result["mean_positions_at_cap"] == pytest.approx(3.0)
        assert result["date_rate"] == 1.0

    def test_no_position_at_the_cap_reports_zero(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-31"] * 2),
            "asset": ["A", "B"], "weight": [0.1, 0.2],
        })
        assert monitoring.cap_binding_rate(frame, max_weight=0.25)["date_rate"] == 0.0

    def test_turnover_is_the_absolute_weight_change_between_rebalances(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]),
            "asset": ["A", "B", "A", "B"],
            "weight": [0.5, 0.5, 0.7, 0.3],
        })
        result = monitoring.turnover_summary(frame)
        assert result["mean"] == pytest.approx(0.4)  # |0.7-0.5| + |0.3-0.5|
        assert result["n_rebalances"] == 2

    def test_a_single_rebalance_has_no_turnover_to_measure(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-31"] * 2),
            "asset": ["A", "B"], "weight": [0.5, 0.5],
        })
        assert monitoring.turnover_summary(frame)["mean"] == 0.0

    def test_fallback_rate_counts_non_converged_rebalances(self):
        frame = pd.DataFrame({"converged": [True, True, False, True]})
        result = monitoring.fallback_rate(frame)
        assert result["fallback_rate"] == pytest.approx(0.25)
        assert result["n_rebalances"] == 4


class TestReportAssembly:
    def test_stable_inputs_produce_no_warnings(self):
        report = monitoring.compare_distributions({"f": STABLE_A}, {"f": STABLE_B})
        assert monitoring.build_warnings(report) == []

    def test_shifted_inputs_produce_an_alert_naming_the_metric(self):
        report = monitoring.compare_distributions({"f": STABLE_A}, {"f": SHIFTED_LOCATION})
        warnings = monitoring.build_warnings(report)
        assert len(warnings) == 1
        assert warnings[0]["severity"] == "alert"
        assert warnings[0]["metric"].endswith("f")

    def test_a_missing_feature_is_reported_as_schema_change_not_drift(self):
        """A feature that vanished is a pipeline break, not a moving market."""
        report = monitoring.compare_distributions(
            {"kept": STABLE_A, "vanished": STABLE_A}, {"kept": STABLE_B}
        )
        assert report["__schema_mismatch__"]["missing_from_evaluation"] == ["vanished"]
        warnings = monitoring.build_warnings(report)
        assert any("schema mismatch" in w["detail"] for w in warnings)

    def test_alerts_sort_ahead_of_warnings(self):
        report = {
            "mild": {"psi": 0.15, "interpretation": "moderate_shift"},
            "severe": {"psi": 0.90, "interpretation": "significant_shift"},
        }
        severities = [w["severity"] for w in monitoring.build_warnings(report)]
        assert severities == ["alert", "warning"]


class TestOperationalPosture:
    """Monitoring warns. It must never be able to change what the model does."""

    def test_the_module_exposes_no_write_or_fit_path(self):
        import inspect

        source = inspect.getsource(monitoring)
        for forbidden in ("write_text(", "to_parquet(", ".fit(", "to_json("):
            assert forbidden not in source, (
                f"monitoring.py contains {forbidden!r}. This module must observe "
                f"only: it emits warnings and never alters model behaviour or "
                f"artifacts (docs/MODEL_GOVERNANCE.md §8)."
            )


class TestFallbackMonitoringCoversEveryPath:
    """The gap this closes: monitoring saw one of three fallback paths.

    `fallback_rate` reads the regime timeline's `converged` column. It is blind
    to DCC-GARCH degrading to Ledoit-Wolf and to an ML signal degrading to the
    naive sample mean, so a baseline built on it alone would understate how
    often a labelled model was not the model that produced the number.
    """

    @staticmethod
    def _reports() -> pd.DataFrame:
        return pd.DataFrame({
            "universe": ["u"] * 6,
            "strategy": ["dcc_garch"] * 3 + ["rf_signal"] * 3,
            "Date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"] * 2),
            "fit_status": ["ok", "fallback", "ok", "ok", "ok", "fallback"],
            "model_effective": [
                "dcc_garch", "dcc_garch [via ledoit_wolf]", "dcc_garch",
                "rf_signal", "rf_signal", "rf_signal [via naive_sample_mean]",
            ],
            "fallback_reason": [None, "GARCH non-convergence", None,
                                None, None, "thin panel"],
        })

    def test_rates_are_computed_per_strategy_not_pooled(self):
        rates = monitoring.model_fallback_rates(self._reports())
        assert set(rates) == {"u/dcc_garch", "u/rf_signal"}
        assert rates["u/dcc_garch"]["fallback_rate"] == pytest.approx(1 / 3)

    def test_the_effective_model_histogram_names_the_substitute(self):
        """'dcc_garch fell back' is far less useful than what it fell back TO."""
        rates = monitoring.model_fallback_rates(self._reports())
        assert "dcc_garch [via ledoit_wolf]" in rates["u/dcc_garch"]["models_effective"]

    def test_a_window_restricts_the_rows_considered(self):
        rates = monitoring.model_fallback_rates(
            self._reports(), start="2024-03-01", end="2024-03-31"
        )
        assert rates["u/dcc_garch"]["rebalances"] == 1
        assert rates["u/dcc_garch"]["fallback_rate"] == 0.0

    def test_a_rising_rate_is_flagged_a_falling_one_is_not(self):
        """A rise means the estimator fails more on newer data — model validity."""
        rising = monitoring.fallback_rate_shift(
            {"u/dcc_garch": {"fallback_rate": 0.05}},
            {"u/dcc_garch": {"fallback_rate": 0.40}},
        )
        assert rising["u/dcc_garch"]["interpretation"] == "significant_shift"

        falling = monitoring.fallback_rate_shift(
            {"u/dcc_garch": {"fallback_rate": 0.40}},
            {"u/dcc_garch": {"fallback_rate": 0.05}},
        )
        assert falling["u/dcc_garch"]["interpretation"] == "stable"

    def test_a_strategy_missing_from_one_window_is_not_comparable(self):
        """Silently treating an absent strategy as 0.0 would invent a result."""
        shift = monitoring.fallback_rate_shift(
            {"u/dcc_garch": {"fallback_rate": 0.1}}, {"u/rf_signal": {"fallback_rate": 0.1}}
        )
        assert shift["u/dcc_garch"]["interpretation"] == "not_comparable"
        assert shift["u/dcc_garch"]["delta"] is None

    def test_an_absent_artifact_reports_not_measured_never_zero(self):
        """The distinction that matters: 'nothing ran' vs 'nothing fell back'."""
        import run_monitoring_baseline as runner

        block = runner._fallback_block(None, "u", None, None)
        assert block["status"] == "not_measured"
        assert block["by_strategy"] == {}
        assert "NOT a zero fallback rate" in block["note"]
