"""
test_global_2004_q2.py — Q2's protocol controls, made executable.

Q2 is the SEARCH, so its integrity rests on things that are easy to get
subtly wrong and hard to notice afterwards: how many candidates were actually
corrected for, whether any were quietly dropped, whether the benchmark is the
same realization Q1 used, and — once four p-values exist in one artifact —
which of them the verdict is allowed to read.

The controls locked in here:

  1. The reachable ledger is derived from FROZEN CONFIG and matches what ran.
  2. No candidate is deduplicated on observed performance.
  3. The benchmark reproduces Q1 EXACTLY.
  4. The deployable challenger is selected forward-only, never on the test set.
  5. The concordance rule is exactly "RC AND SPA on the PRIMARY endpoint",
     and the secondary endpoint can never promote itself.
  6. No statistic shopping: the verdict must not track the smallest p-value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_global_2004_q2 as q2  # noqa: E402
from run_reality_check import _label, candidate_configs  # noqa: E402
from utils import load_params  # noqa: E402

ARTIFACT = ROOT / "data" / "gold" / "global_2004_q2_results.json"
Q1_ARTIFACT = ROOT / "data" / "gold" / "global_2004_q1_results.json"


def _artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip("Q2 artifact absent — run src/run_global_2004_q2.py.")
    return json.loads(ARTIFACT.read_text())


# ── 1. The ledger is derived, not observed ───────────────────────────────────

class TestReachableLedger:
    def test_expected_count_is_derivable_from_frozen_config_alone(self):
        """No return, no artifact, no observed performance is consulted."""
        params = load_params()
        assert q2.expected_candidate_count(params) == len(candidate_configs(params))

    def test_expected_count_matches_the_documented_240(self):
        """(6 RF + 9 XGB) x 4 shrink x 4 penalty. A change here silently
        changes the multiplicity every Q2 p-value is corrected for."""
        assert q2.expected_candidate_count(load_params()) == 240

    def test_labels_are_unique_across_the_whole_ledger(self):
        labels = [_label(c) for c in candidate_configs(load_params())]
        assert len(set(labels)) == len(labels)

    def test_executed_ledger_equals_the_expected_count(self):
        art = _artifact()["candidate_ledger"]
        assert art["executed_count"] == art["expected_reachable_count"]
        assert art["executed_count"] == q2.expected_candidate_count(load_params())

    def test_no_deduplication_on_performance(self):
        assert "NONE" in _artifact()["candidate_ledger"]["deduplication"]

    def test_every_candidate_series_is_persisted(self):
        """The full frozen-test return matrix must survive the run.

        Without it the correction is unauditable: a reader cannot recompute
        the composite null from the artifact alone.
        """
        import pandas as pd

        art = _artifact()
        path = ROOT / art["candidate_ledger"]["series_artifact"]
        if not path.is_file():
            pytest.skip("series parquet absent")
        frame = pd.read_parquet(path)
        names = set(frame["candidate"].unique())
        # every candidate plus the benchmark
        assert len(names) == art["candidate_ledger"]["executed_count"] + 1
        assert q2.BENCHMARK in names


# ── 2. The benchmark is the SAME realization Q1 measured ─────────────────────

class TestBenchmarkReproducesQ1:
    def test_benchmark_metrics_match_q1_exactly(self):
        if not Q1_ARTIFACT.is_file():
            pytest.skip("Q1 artifact absent")
        q1 = json.loads(Q1_ARTIFACT.read_text())["candidate"]
        q2_bench = _artifact()["benchmark"]

        for field in ("net_sharpe", "net_geometric_annual_return", "max_drawdown",
                      "avg_turnover", "n_rebalances_in_test", "n_test_days"):
            assert q2_bench[field] == q1[field], (
                f"{field} differs between Q1 and Q2. The two questions must be "
                "answered against the identical benchmark realization."
            )

    def test_the_artifact_asserts_the_reproduction(self):
        assert _artifact()["benchmark"]["reproduces_q1_exactly"] is True


# ── 3. Forward-only selection of the deployable challenger ───────────────────

class TestDeployableChallengerSelection:
    def test_both_families_have_a_selected_configuration(self):
        deployable = _artifact()["deployable_challengers"]
        assert set(deployable) == {"rf_signal_tuned", "xgb_signal_tuned"}

    def test_selection_is_documented_as_forward_only_and_test_blind(self):
        for spec in _artifact()["deployable_challengers"].values():
            text = spec["selection"].lower()
            assert "forward-only" in text
            assert "never shown" in text or "never" in text
            assert "test segment" in text

    def test_selected_configs_are_inside_the_reachable_ledger(self):
        """A deployable config outside the corrected family would be an
        uncorrected selection — the correction must cover what was chosen."""
        params = load_params()
        reachable = {
            (c["model_type"], tuple(sorted(c["ml_params"].items())),
             c["shrinkage_weight"], c["turnover_penalty"])
            for c in candidate_configs(params)
        }
        for spec in _artifact()["deployable_challengers"].values():
            key = (spec["model_type"], tuple(sorted(spec["ml_params"].items())),
                   spec["shrinkage_weight"], spec["turnover_penalty"])
            assert key in reachable, f"{key} is outside the corrected ledger"


# ── 4. The interpretation rule, exactly as amended ───────────────────────────

class TestConcordanceRule:
    def test_both_endpoints_and_both_tests_are_persisted(self):
        tests = _artifact()["family_tests"]
        assert set(tests) == {"primary_sharpe", "secondary_mean_return"}
        for block in tests.values():
            for field in ("reality_check_p_value", "spa_p_value",
                          "rc_rejects_at_alpha", "spa_rejects_at_alpha"):
                assert field in block

    def test_primary_endpoint_is_the_sharpe_differential(self):
        art = _artifact()
        assert art["family_tests"]["primary_sharpe"]["statistic"] == "sharpe"
        assert "Sharpe" in art["verdict"]["primary_endpoint"]

    def test_concordance_equals_rc_and_spa_on_the_primary_endpoint(self):
        art = _artifact()
        primary = art["family_tests"]["primary_sharpe"]
        expected = primary["rc_rejects_at_alpha"] and primary["spa_rejects_at_alpha"]
        assert art["verdict"]["concordant_evidence_of_family_outperformance"] == expected

    def test_the_secondary_endpoint_cannot_promote_itself(self):
        """Even if the secondary endpoint rejects on both tests, the verdict
        must not follow it. This is the statistic-shopping guard with teeth."""
        art = _artifact()
        secondary = art["family_tests"]["secondary_mean_return"]
        primary = art["family_tests"]["primary_sharpe"]
        if (secondary["rc_rejects_at_alpha"] and secondary["spa_rejects_at_alpha"]
                and not (primary["rc_rejects_at_alpha"] and primary["spa_rejects_at_alpha"])):
            assert art["verdict"]["concordant_evidence_of_family_outperformance"] is False

    def test_evidence_status_matches_the_number_of_rejections(self):
        art = _artifact()
        primary = art["family_tests"]["primary_sharpe"]
        n = int(primary["rc_rejects_at_alpha"]) + int(primary["spa_rejects_at_alpha"])
        status = art["verdict"]["evidence_status"]
        assert status == {
            2: "concordant_evidence_of_family_outperformance",
            1: "test_dependent_evidence",
            0: "no_evidence_of_family_outperformance",
        }[n]

    def test_alpha_is_the_inherited_level(self):
        art = _artifact()
        assert art["verdict"]["alpha"] == load_params()["phase5"]["bootstrap"]["alpha"] == 0.10

    def test_no_single_family_p_value_is_reported(self):
        """RC and SPA are reported separately and never merged."""
        art = _artifact()
        assert "p_value" not in art["verdict"]
        assert "family_p_value" not in art["verdict"]

    def test_best_candidate_is_labelled_as_not_evidence(self):
        for block in _artifact()["family_tests"].values():
            assert "NOT evidence" in block["best_candidate_note"]


# ── 5. Telemetry and limitations ─────────────────────────────────────────────

class TestTelemetryAndLimitations:
    def test_fallback_telemetry_covers_every_candidate(self):
        art = _artifact()
        telemetry = art["fallback_telemetry"]["candidates"]
        assert len(telemetry) == art["candidate_ledger"]["executed_count"]
        for entry in telemetry.values():
            assert "fallback_count" in entry
            assert "effective_models" in entry

    def test_benchmark_telemetry_is_recorded(self):
        assert "effective_models" in _artifact()["fallback_telemetry"]["benchmark"]

    def test_both_required_limitations_are_attached(self):
        text = " ".join(_artifact()["limitations"]).lower()
        assert "macro-feature policy" in text, "attribution confound missing"
        assert "outer" in text and "construct" in text, "outer selection missing"

    def test_generated_from_a_clean_revision(self):
        rev = _artifact()["provenance"]["git_revision"]
        assert not rev.endswith("-dirty")
