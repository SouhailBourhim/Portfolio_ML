"""
test_provenance.py — an artifact must say what produced it, and a consumer must
refuse it when that no longer holds.

The incident, stated once so the tests below read as the answer to it:
`data/gold/nested_walkforward_results.json` was produced on 2026-07-28 from a
MIXED-CURRENCY `full_2021`. The base-currency correction rebuilt every pipeline
artifact; this one survived untouched because it was neither a DVC output nor
carried any record of its inputs. Chapter 5's figure builder would have consumed
it and rendered "the nested protocol puts the regime system back in front" —
true on the old data, sign-flipped on the new — beside current numbers.

Nothing in the suite could have caught it. `test_artifact_consistency.py`
compares artifacts against EACH OTHER; nothing compared an artifact against the
data underneath it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from provenance import (
    StaleArtifactError,
    build_provenance,
    read_numeraire,
    require_current_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def tree(tmp_path):
    """A miniature repo: a currency manifest and one source artifact."""
    (tmp_path / "data" / "gold").mkdir(parents=True)
    (tmp_path / "data" / "gold" / "currency_manifest.json").write_text(json.dumps({
        "universes": {
            "full_2021": {"converted": True, "base_currency": "MAD",
                          "hedge_status": "unhedged", "fx_series": "USDMAD"},
            "etf_2017": {"converted": False, "base_currency": "USD"},
        }
    }))
    (tmp_path / "params.yaml").write_text("backtest:\n  max_weight: 0.25\n")
    return tmp_path


@pytest.fixture()
def returns():
    idx = pd.bdate_range("2021-07-30", periods=400)
    return pd.DataFrame(1.0, index=idx, columns=["SPY", "IAM.CS"])


class TestProvenanceIsRecorded:
    def test_it_states_the_numeraire_and_ranges(self, tree, returns):
        prov = build_provenance("full_2021", returns, oos_index=returns.index[200:],
                                source_artifacts=["params.yaml"], root=tree)
        assert prov["base_currency"] == "MAD"
        assert prov["hedge_status"] == "unhedged"
        assert prov["data_range"]["start"] == "2021-07-30"
        assert prov["data_range"]["n_rows"] == 400
        assert prov["oos_range"]["n_rows"] == 200
        assert prov["git_revision"]
        assert prov["source_artifacts"]["params.yaml"]

    def test_the_numeraire_is_copied_from_the_manifest_not_restated(self, tree, returns):
        """An artifact must not be able to claim a currency the data layer
        disagrees with, so it reads the canonical manifest rather than asserting."""
        prov = build_provenance("etf_2017", returns, root=tree)
        assert prov["base_currency"] == "USD"
        assert prov["currency_converted"] is False

    def test_a_tree_without_a_manifest_cannot_produce_provenance(self, tmp_path, returns):
        """That state IS the pre-correction tree. Failing here is the point."""
        with pytest.raises(StaleArtifactError, match="predates the base-currency"):
            build_provenance("full_2021", returns, root=tmp_path)


class TestChapterFiveRefusesAStaleArtifact:
    """The four ways the 2026-07-28 artifact could have reached a figure."""

    def _artifact(self, tree, returns, **over):
        prov = build_provenance("full_2021", returns,
                                source_artifacts=["params.yaml"], root=tree)
        prov.update(over)
        return {"universe": "full_2021", "provenance": prov, "strategies": {}}

    def test_a_missing_artifact_is_refused_with_the_command_to_produce_it(self, tree):
        with pytest.raises(StaleArtifactError, match="dvc repro nested_walkforward"):
            require_current_artifact(tree / "data" / "gold" / "absent.json", root=tree)

    def test_an_artifact_with_no_provenance_block_is_refused(self, tree):
        """Exactly the shape of every pre-correction artifact: well-formed,
        plausible, and silent about what produced it."""
        path = tree / "data" / "gold" / "nested_walkforward_results.json"
        path.write_text(json.dumps({"universe": "full_2021", "strategies": {"a": 1}}))
        with pytest.raises(StaleArtifactError, match="carries no `provenance` block"):
            require_current_artifact(path, expect_universe="full_2021", root=tree)

    def test_a_mixed_currency_artifact_cannot_be_consumed(self, tree, returns):
        """The pre-correction result was denominated in neither one currency nor
        the other. Whatever it says, it is not MAD, and it must not appear beside
        MAD-valued numbers."""
        art = self._artifact(tree, returns, base_currency=None)
        with pytest.raises(StaleArtifactError, match="not 'MAD'"):
            require_current_artifact(art, expect_base_currency="MAD", root=tree)

    def test_a_usd_artifact_is_refused_where_mad_is_required(self, tree, returns):
        art = self._artifact(tree, returns, base_currency="USD")
        with pytest.raises(StaleArtifactError, match="denominated in 'USD'"):
            require_current_artifact(art, expect_base_currency="MAD", root=tree)

    def test_an_artifact_whose_inputs_changed_underneath_it_is_refused(self, tree, returns):
        """The strongest check, and the one a date cannot make. The artifact is
        internally perfect; the DATA moved after it was written."""
        art = self._artifact(tree, returns)
        (tree / "params.yaml").write_text("backtest:\n  max_weight: 0.40\n")   # data moved
        with pytest.raises(StaleArtifactError, match="have since changed"):
            require_current_artifact(art, expect_base_currency="MAD", root=tree)

    def test_a_current_artifact_passes(self, tree, returns):
        """Non-vacuity: the gate must accept the good case, or the tests above
        prove only that it always raises."""
        art = self._artifact(tree, returns)
        assert require_current_artifact(art, expect_universe="full_2021",
                                        expect_base_currency="MAD", root=tree) is art

    def test_the_wrong_universe_is_refused(self, tree, returns):
        art = self._artifact(tree, returns)
        with pytest.raises(StaleArtifactError, match="not 'etf_2017'"):
            require_current_artifact(art, expect_universe="etf_2017", root=tree)


class TestTheChapterFiveBuilderIsActuallyGated:
    """A gate nothing calls is decoration. This asserts the wiring, by source
    inspection, in the style of the existing no-hardcoded-Sharpe test."""

    def test_the_builder_loads_the_nested_artifact_through_the_gate(self):
        src = (ROOT / "scripts" / "build_figures_chap5.py").read_text(encoding="utf-8")
        assert "require_current_artifact" in src, (
            "build_figures_chap5.py must not read nested_walkforward_results.json "
            "with a bare json.loads — that is how the stale artifact reached the "
            "figure builder in the first place"
        )
        assert 'L("nested_walkforward_results.json")' not in src, (
            "the bare loader for the nested artifact must be gone, not merely "
            "supplemented by a gate that runs beside it"
        )

    def test_the_nested_experiment_records_provenance(self):
        src = (ROOT / "experiments" / "nested_walkforward.py").read_text(encoding="utf-8")
        assert "build_provenance" in src
        assert '"provenance"' in src

    def test_the_nested_experiment_is_a_tracked_dvc_stage(self):
        import yaml
        stages = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]
        assert "nested_walkforward" in stages, (
            "the experiment must be in the graph; being outside it is why the "
            "artifact went stale unnoticed (AGENTS.md §17.8)"
        )
        stage = stages["nested_walkforward"]
        assert "data/gold/nested_walkforward_results.json" in stage["outs"]
        for dep in ("data/gold/log_returns.parquet",
                    "data/gold/ml_features_full.parquet",
                    "data/gold/currency_manifest.json",
                    "src/provenance.py"):
            assert dep in stage["deps"], f"{dep} must be a declared dependency"


class TestTheCommittedArtifactIsCurrent:
    """Run against the real tree when the artifact is present. Skipped on a
    fresh clone, where it is DVC-managed and absent."""

    def test_the_nested_artifact_in_this_tree_passes_the_gate(self):
        path = ROOT / "data" / "gold" / "nested_walkforward_results.json"
        if not path.is_file():
            pytest.skip("nested_walkforward_results.json absent — run `dvc repro`.")
        require_current_artifact(path, expect_universe="full_2021",
                                 expect_base_currency="MAD", root=ROOT)

    def test_the_gold_manifest_says_what_each_universe_is(self):
        if not (ROOT / "data" / "gold" / "currency_manifest.json").is_file():
            pytest.skip("currency_manifest.json absent.")
        assert read_numeraire("full_2021", root=ROOT)["base_currency"] == "MAD"
        assert read_numeraire("etf_2017", root=ROOT)["base_currency"] == "USD"
