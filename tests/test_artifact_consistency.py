"""
test_artifact_consistency.py — Committed artifacts must agree with each other.

WHY THIS FILE EXISTS. Every serious incident in this project has been the same
shape: two artifacts that should describe the same world describing different
ones, with nothing detecting it.

  * 2026-07-25 — the BVC dividend correction regenerated Gold, but
    `phase5_results.json` was left three days stale. For three days the
    dashboard rendered PRE-correction confidence intervals directly beneath
    POST-correction Sharpe ratios, under a sentence claiming the latter had
    been validated on the former.
  * 2026-07-30 — a `phase4c` run was TORN: the Dagster nightly rebuilt Gold
    mid-run, so `etf_2017` was computed on the 07-24 snapshot and `full_2021`
    on a newer one. Every `full_2021` baseline disagreed with the dashboard by
    exactly the amount five extra days explains.
  * ongoing — `phase2_hurdle.json` and `dashboard_showcase.json` can name
    DIFFERENT winning strategies for `etf_2017` at an identical Sharpe, because
    the 25% cap makes several optimizers degenerate (CLAUDE.md §10.1).

The first two were caught by a human reading mtimes at the right moment. The
existing test suite could not have caught either: `test_run_dashboard_data.py`
verifies the dashboard runner copies its inputs FAITHFULLY, which stays true
when those inputs are stale. Fidelity and consistency are different properties.

This file tests consistency. It is deliberately an INTEGRATION check against
the real committed artifacts in `data/gold/`, not against synthetic fixtures —
a fixture cannot go stale, which is precisely the bug class being guarded.

SKIP POLICY. Every test skips when its artifacts are absent, so a fresh clone
(where `data/` is DVC-managed and not yet pulled) does not fail. That is a
deliberate trade: the suite must stay hermetic and runnable without data. When
the artifacts ARE present — which is the state any result is published from —
these become hard assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"

# Strategies every comparison artifact re-computes from the same Gold inputs.
# If two artifacts disagree about one of these, they were not computed on the
# same data — which is exactly the torn-artifact signature.
SHARED_BASELINES = ("equal_weight", "min_variance_lw", "max_sharpe", "regime_conditional")

# Net Sharpe is rounded to 4dp in the artifacts; anything above this is a real
# disagreement, not a formatting artifact.
TOLERANCE = 5e-4


def _load(name: str) -> dict | None:
    path = GOLD / name
    return json.loads(path.read_text()) if path.exists() else None


def _require(name: str) -> dict:
    data = _load(name)
    if data is None:
        pytest.skip(f"{name} not present — run the pipeline or `dvc pull` first.")
    return data


@pytest.fixture(scope="module")
def showcase() -> dict:
    return _require("dashboard_showcase.json")


class TestOosWindowsAgree:
    """The strongest torn-artifact detector: two artifacts covering different
    date ranges were computed on different data, full stop."""

    @pytest.mark.parametrize("artifact", ["phase2_hurdle.json", "phase4_results.json",
                                          "phase4b_results.json", "phase4c_results.json"])
    def test_oos_window_matches_the_dashboard(self, artifact, showcase):
        data = _load(artifact)
        if data is None:
            pytest.skip(f"{artifact} not present.")
        for universe, entry in data.items():
            if universe not in showcase["universes"]:
                continue
            expected = showcase["universes"][universe]
            for field in ("oos_start", "oos_end"):
                if field not in entry or field not in expected:
                    continue
                assert entry[field] == expected[field], (
                    f"{artifact} [{universe}] {field}={entry[field]} but the dashboard "
                    f"has {expected[field]}. These were computed on DIFFERENT data — "
                    f"the 2026-07-30 torn-artifact signature (CLAUDE.md §12C). Re-run "
                    f"the stale one against the committed Gold snapshot."
                )


class TestSharedStrategiesAgree:
    """`phase4c_results.json` re-computes every baseline the dashboard shows.
    Same strategy + same data => same number, to rounding."""

    def test_phase4c_baselines_match_the_dashboard(self, showcase):
        p4c = _load("phase4c_results.json")
        if p4c is None:
            pytest.skip("phase4c_results.json not present.")
        mismatches = []
        for universe, entry in p4c.items():
            per = entry.get("per_strategy", {})
            shown = showcase["universes"].get(universe, {}).get("strategies", {})
            for name in SHARED_BASELINES:
                if name not in per or name not in shown:
                    continue
                a, b = per[name]["sharpe_net"], shown[name]["sharpe_net"]
                if abs(a - b) > TOLERANCE:
                    mismatches.append(f"{universe}/{name}: phase4c={a} dashboard={b}")
        assert not mismatches, (
            "Artifacts disagree about strategies computed from identical inputs:\n  "
            + "\n  ".join(mismatches)
            + "\nThis means one of them is stale. Do NOT publish until resolved."
        )


class TestHurdlesAreConsistent:
    def test_phase4_hurdle_is_the_argmax_of_what_phase4c_measured(self):
        """A hurdle that isn't the best number in its own comparison is stale."""
        p4, p4c = _load("phase4_results.json"), _load("phase4c_results.json")
        if p4 is None or p4c is None:
            pytest.skip("phase4/phase4c results not present.")
        for universe, entry in p4.items():
            per = p4c.get(universe, {}).get("per_strategy", {})
            if not per:
                continue
            best = max(per.values(), key=lambda v: v["sharpe_net"])["sharpe_net"]
            assert entry["sharpe_net"] <= best + TOLERANCE, (
                f"{universe}: the Phase 4 hurdle ({entry['sharpe_net']}) EXCEEDS the best "
                f"strategy phase4c measured ({best}). One artifact predates the other."
            )

    def test_downstream_phases_cite_the_current_hurdle(self):
        """4B and 4C both record the hurdle they were judged against."""
        p4 = _load("phase4_results.json")
        if p4 is None:
            pytest.skip("phase4_results.json not present.")
        p4c = _load("phase4c_results.json")
        if not p4c:
            pytest.skip("phase4c_results.json not present.")
        for universe, entry in p4c.items():
            cited = entry.get("phase4_hurdle")
            if not cited:
                continue
            actual = p4[universe]["sharpe_net"]
            assert abs(cited["sharpe_net"] - actual) <= TOLERANCE, (
                f"{universe}: phase4c was judged against hurdle {cited['sharpe_net']}, but "
                f"phase4_results.json now says {actual}. The hurdle moved after 4C ran — "
                f"4C's beats/does-not-beat verdict is no longer meaningful."
            )


class TestDashboardCisComeFromTheCurrentPhase5:
    """The 2026-07-25 incident, as an assertion."""

    def test_phase5_cis_are_copied_from_the_committed_phase5_run(self, showcase):
        p5 = _load("phase5_results.json")
        if p5 is None:
            pytest.skip("phase5_results.json not present.")
        for universe, shown in showcase["universes"].items():
            window = shown.get("phase5_test_window") or {}
            source = p5.get(universe, {}).get("baselines", {})
            for name in ("regime_conditional", "equal_weight"):
                if name not in window or name not in source:
                    continue
                a = window[name].get("test_sharpe_net")
                b = source[name].get("test_sharpe_net")
                assert a == b, (
                    f"{universe}/{name}: the dashboard shows a held-out Sharpe of {a} but "
                    f"phase5_results.json says {b}. The dashboard is rendering CIs from a "
                    f"superseded Phase 5 run — the exact 2026-07-25 failure. Re-run "
                    f"`python src/run_dashboard_data.py`."
                )

    def test_phase5_test_window_dates_are_carried_verbatim(self, showcase):
        p5 = _load("phase5_results.json")
        if p5 is None:
            pytest.skip("phase5_results.json not present.")
        for universe, shown in showcase["universes"].items():
            window = shown.get("phase5_test_window") or {}
            src = p5.get(universe, {})
            for field in ("test_start", "test_end"):
                if window.get(field) and src.get(field):
                    assert window[field] == src[field], (
                        f"{universe}: dashboard {field}={window[field]} vs phase5 "
                        f"{src[field]} — different evaluation windows."
                    )


class TestGoldInputsAreOneSnapshot:
    """Both universes must end on the same date. They are refreshed by separate
    pipeline stages and have drifted apart twice (CLAUDE.md §8.2, §17.9)."""

    def test_both_universes_share_an_end_date(self):
        paths = [GOLD / "log_returns.parquet", GOLD / "log_returns_etf.parquet"]
        if not all(p.exists() for p in paths):
            pytest.skip("Gold return matrices not present.")
        ends = {p.name: pd.read_parquet(p).index.max() for p in paths}
        assert len(set(ends.values())) == 1, (
            f"The two universes end on different dates: {ends}. They are produced by "
            f"separate stages and have drifted twice before; every dual-universe result "
            f"computed across this boundary is incomparable."
        )

    def test_feature_matrices_do_not_lead_their_returns(self):
        """A feature matrix ending AFTER its returns would mean the engine could
        be handed rows it has no returns for."""
        pairs = [("log_returns.parquet", "ml_features_full.parquet"),
                 ("log_returns_etf.parquet", "ml_features_etf.parquet")]
        for rets, feats in pairs:
            rp, fp = GOLD / rets, GOLD / feats
            if not (rp.exists() and fp.exists()):
                pytest.skip(f"{rets}/{feats} not present.")
            r_end = pd.read_parquet(rp).index.max()
            f_end = pd.read_parquet(fp).index.max()
            assert f_end <= r_end, (
                f"{feats} ends {f_end}, after {rets} at {r_end}."
            )


class TestReportTablesAreGeneratedNotTyped:
    """The report must not restate a result the artifact can move.

    The paired-comparison table was hand-typed into `Chapter5.tex` when it was
    first written. Every figure in the report is built from Gold precisely so
    the prose cannot outlive the run it describes, and eight rows of LaTeX
    literals were the one place that convention had been broken: the next
    Phase 5 run would move `paired_comparison_results.json` and leave the
    report asserting the old numbers, with nothing able to tell.
    """

    TABLE = ROOT / "docs" / "rapport" / "assets" / "tables" / "paires.tex"
    CHAPTER = ROOT / "docs" / "rapport" / "chapters" / "Chapter5.tex"

    def _table(self) -> str:
        if not self.TABLE.is_file():
            pytest.skip("paires.tex not built — run scripts/build_figures_chap5.py.")
        return self.TABLE.read_text(encoding="utf-8")

    def test_chapter5_inputs_the_generated_table_instead_of_defining_one(self):
        if not self.CHAPTER.is_file():
            pytest.skip("Chapter5.tex not present.")
        source = self.CHAPTER.read_text(encoding="utf-8")
        assert r"\input{assets/tables/paires}" in source
        assert r"\label{tab:paires}" not in source, (
            "Chapter5.tex defines the paired table inline again. It must consume "
            "the generated assets/tables/paires.tex so the numbers cannot drift."
        )

    def test_every_row_matches_the_paired_comparison_artifact(self):
        paired = _require("paired_comparison_results.json")
        table = self._table()
        for comparison in paired["comparisons"]:
            # Rendered exactly as the generator writes it: French decimal comma
            # braced for math mode, three decimals.
            delta = f"{comparison['sharpe_diff']:+.3f}".replace(".", "{,}")
            p_value = f"{comparison['p_value_no_outperformance']:.3f}".replace(".", ",")
            assert delta in table, (
                f"{comparison['universe']} {comparison['candidate']} vs "
                f"{comparison['benchmark']}: Δ={delta} is not in the generated table. "
                f"Re-run scripts/build_figures_chap5.py."
            )
            assert p_value in table, f"p={p_value} is not in the generated table."

    def test_the_table_reports_as_many_rows_as_the_artifact_holds(self):
        paired = _require("paired_comparison_results.json")
        rows = [line for line in self._table().splitlines()
                if line.strip().startswith("& ") and line.rstrip().endswith(r"\\")]
        assert len(rows) == len(paired["comparisons"]), (
            f"The table shows {len(rows)} comparisons but the artifact holds "
            f"{len(paired['comparisons'])}. A silently dropped row reads as a "
            f"smaller search than was actually run."
        )
