"""test_fit_reports.py — the persisted fallback artifact must not mislead.

Two failure modes are guarded here, and the second is the one that motivated
the whole exercise.

1. **The artifact must describe the PUBLISHED strategies.** A fallback rate
   measured on a differently-configured lookalike answers a different
   question. An early draft of `_build_strategies` omitted the `ml_signals`
   block from params.yaml and produced an rf_signal scoring 0.146 against the
   published 1.123 — same name, different strategy, and a zero fallback rate
   that meant nothing. The cross-check against `dashboard_showcase.json` is
   what caught it.

2. **A missing number must never render as a plausible one.** With zero
   non-fallback days there is no excluding-fallback Sharpe; reporting 0, blank
   or a normal-looking figure would recreate the exact opacity the fallback
   rate exists to remove, one level up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"

SUMMARY = GOLD / "fit_report_summary.json"
REPORTS = GOLD / "fit_reports.parquet"

REQUIRED_FIELDS = (
    "model_requested", "model_effective", "fit_status",
    "n_training_rows", "fallback_reason", "convergence_warning",
)


def _summary() -> dict:
    if not SUMMARY.is_file():
        pytest.skip("fit_report_summary.json not present — run `dvc repro fit_reports`.")
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _reports() -> pd.DataFrame:
    if not REPORTS.is_file():
        pytest.skip("fit_reports.parquet not present — run `dvc repro fit_reports`.")
    return pd.read_parquet(REPORTS)


class TestArtifactShape:
    def test_every_rebalance_carries_all_six_fields(self):
        frame = _reports()
        for field in REQUIRED_FIELDS:
            assert field in frame.columns, f"{field} missing from fit_reports.parquet"
        assert {"universe", "strategy", "Date"} <= set(frame.columns)
        assert not frame.empty

    def test_every_fallback_row_names_a_substitute_and_a_reason(self):
        """The acceptance criterion, checked against the persisted data."""
        frame = _reports()
        fell_back = frame[frame["fit_status"] == "fallback"]
        for _, row in fell_back.iterrows():
            assert row["fallback_reason"], f"{row['strategy']} @ {row['Date']}: no reason"
            assert row["model_effective"] != row["model_requested"], (
                "a fallback whose effective model equals the requested one hides "
                "the substitution it is supposed to reveal"
            )

    def test_summary_and_raw_rows_agree_on_the_fallback_count(self):
        """The table and its source artifact must not drift apart."""
        frame, summary = _reports(), _summary()
        for entry in summary["results"]:
            block = frame[
                (frame["universe"] == entry["universe"])
                & (frame["strategy"] == entry["strategy_requested"])
            ]
            assert len(block) == entry["rebalances"]
            assert int((block["fit_status"] == "fallback").sum()) == entry["fallback_rebalances"]


class TestMeasuresThePublishedStrategies:
    """A rate measured on a lookalike answers a different question."""

    def test_full_period_sharpe_matches_the_published_dashboard_figure(self):
        showcase_path = GOLD / "dashboard_showcase.json"
        if not showcase_path.is_file():
            pytest.skip("dashboard_showcase.json not present.")
        showcase = json.loads(showcase_path.read_text(encoding="utf-8"))

        checked = 0
        for entry in _summary()["results"]:
            published = (
                showcase["universes"].get(entry["universe"], {})
                .get("strategies", {}).get(entry["strategy_requested"])
            )
            if published is None:
                continue  # not a dashboard strategy; nothing to cross-check against
            measured = entry["performance"]["full_period_hybrid"]["net_sharpe"]
            assert measured == pytest.approx(published["sharpe_net"], abs=5e-4), (
                f"{entry['universe']}/{entry['strategy_requested']}: fit_reports says "
                f"{measured}, the published dashboard says {published['sharpe_net']}. "
                f"The runner is not reproducing the published strategy."
            )
            checked += 1
        assert checked, "no strategy could be cross-checked — the guard is vacuous"


class TestAMissingNumberNeverRendersAsAPlausibleOne:
    def test_zero_observation_blocks_are_not_estimable_with_a_reason(self):
        for entry in _summary()["results"]:
            for block in entry["performance"].values():
                if block["n_days"] == 0:
                    assert block["status"] == "not_estimable"
                    assert block["reason"], "an unavailable number must say why"
                    assert block["net_sharpe"] is None, (
                        "a period with no observations must report None — not 0, "
                        "which a reader would take for a measured result"
                    )

    def test_excluding_fallback_is_withheld_below_the_active_day_floor(self):
        summary = _summary()
        floor = summary["min_active_days_for_sharpe"]
        for entry in summary["results"]:
            block = entry["performance"]["excluding_fallback"]
            if block["status"] == "estimated":
                assert block["n_days"] >= floor
            else:
                assert block["net_sharpe"] is None
                assert str(block["n_days"]) in block["reason"] or block["n_days"] == 0, (
                    "a withheld number must disclose how far short the sample fell"
                )

    def test_a_strategy_that_always_fell_back_reports_no_active_performance(self):
        """The rf_signal case from the specification, as a general rule."""
        for entry in _summary()["results"]:
            if entry["active_days"] == 0:
                block = entry["performance"]["excluding_fallback"]
                assert block["status"] == "not_estimable"
                assert block["net_sharpe"] is None


class TestDaysAndRebalancesAreBothReported:
    def test_day_counts_are_present_and_consistent(self):
        """A rebalance governs a month, so the day rate is the economic one."""
        for entry in _summary()["results"]:
            assert entry["fallback_days"] + entry["active_days"] == entry["oos_days"]
            assert 0.0 <= entry["fallback_rate_days"] <= 1.0

    def test_a_zero_rebalance_rate_implies_a_zero_day_rate(self):
        for entry in _summary()["results"]:
            if entry["fallback_rebalances"] == 0:
                assert entry["fallback_days"] == 0


class TestCostSensitivity:
    """A net Sharpe is a claim about an assumed cost model as much as a strategy."""

    def test_every_strategy_reports_all_four_multipliers(self):
        for entry in _summary()["results"]:
            scenarios = entry["cost_sensitivity"]["scenarios"]
            assert set(scenarios) == {"0.5x", "1x", "1.5x", "2x"}

    def test_the_one_x_scenario_equals_the_published_full_period_figure(self):
        """The scenarios are derived from the run, not a separate simulation."""
        for entry in _summary()["results"]:
            base = entry["cost_sensitivity"]["scenarios"]["1x"]["net_sharpe"]
            published = entry["performance"]["full_period_hybrid"]["net_sharpe"]
            assert base == pytest.approx(published, abs=5e-4), (
                f"{entry['universe']}/{entry['strategy_requested']}: the 1x cost "
                f"scenario must reproduce the headline figure exactly."
            )

    def test_higher_costs_never_improve_net_performance(self):
        """Monotonicity is the sanity check that the derivation is right."""
        for entry in _summary()["results"]:
            scenarios = entry["cost_sensitivity"]["scenarios"]
            sharpes = [scenarios[k]["net_sharpe"] for k in ("0.5x", "1x", "1.5x", "2x")]
            assert sharpes == sorted(sharpes, reverse=True), (
                f"{entry['universe']}/{entry['strategy_requested']}: net Sharpe rose "
                f"with costs — {sharpes}"
            )

    def test_the_fallback_rate_is_reported_per_scenario_not_hidden(self):
        """It must be visible per scenario, and it must not move.

        Invariance here is BY CONSTRUCTION — costs never reach the optimizer —
        so a scenario whose rate differed would mean the derivation had
        silently changed the weights.
        """
        for entry in _summary()["results"]:
            rates = {
                s["fallback_rate_rebalances"]
                for s in entry["cost_sensitivity"]["scenarios"].values()
            }
            assert len(rates) == 1, (
                f"{entry['universe']}/{entry['strategy_requested']}: fallback rate "
                f"varied across cost scenarios ({rates}) — costs must not reach "
                f"the optimizer for these strategies."
            )
            assert rates.pop() == entry["fallback_rate_rebalances"]

    def test_the_invariance_is_explained_not_merely_shown(self):
        for entry in _summary()["results"]:
            note = entry["cost_sensitivity"]["fallback_invariance_note"]
            assert "BY CONSTRUCTION" in note and "turnover-PENALIZED" in note


class TestExecutionProfile:
    def test_realism_levers_are_reported(self):
        for entry in _summary()["results"]:
            profile = entry["execution_profile"]
            for field in ("avg_turnover", "max_allocation_observed",
                          "max_allocation_permitted", "avg_holding_days",
                          "cap_binding_position_rate", "rebalance_frequency"):
                assert field in profile, f"{field} missing from execution_profile"

    def test_no_observed_allocation_exceeds_the_permitted_cap(self):
        for entry in _summary()["results"]:
            profile = entry["execution_profile"]
            assert profile["max_allocation_observed"] <= profile["max_allocation_permitted"] + 1e-6

    def test_the_liquidity_caveat_admits_the_missing_market_impact_term(self):
        for entry in _summary()["results"]:
            caveat = entry["execution_profile"]["liquidity_caveat"]
            assert "market-impact" in caveat and "optimistic" in caveat
