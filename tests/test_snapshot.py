"""Tests for the Phase 1 reproducibility snapshot manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import snapshot


@pytest.fixture()
def snapshot_root(tmp_path, monkeypatch) -> Path:
    """A complete, tiny local snapshot without real market data or Git state."""
    monkeypatch.setattr(snapshot, "ROOT", tmp_path)
    for relative in snapshot.SNAPSHOT_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            frame = pd.DataFrame(
                {"SPY": [0.01, -0.02]}, index=pd.date_range("2024-01-01", periods=2)
            )
            frame.to_parquet(path)
        elif path.suffix == ".json":
            path.write_text(json.dumps({"artifact": relative.name}), encoding="utf-8")
        else:
            path.write_text(f"fixture for {relative}\n", encoding="utf-8")
    return tmp_path


def test_write_then_verify_accepts_an_unchanged_snapshot(snapshot_root):
    path = snapshot.write_snapshot()

    assert path == snapshot_root / snapshot.MANIFEST_RELATIVE_PATH
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {str(path) for path in snapshot.SNAPSHOT_FILES}
    assert snapshot.verify_snapshot() == []


def test_verify_detects_a_changed_input(snapshot_root):
    snapshot.write_snapshot()
    (snapshot_root / "params.yaml").write_text("changed: true\n", encoding="utf-8")

    assert snapshot.verify_snapshot() == ["Snapshot mismatch: params.yaml"]


def test_verify_explains_a_missing_manifest(snapshot_root):
    failures = snapshot.verify_snapshot()

    assert len(failures) == 1
    assert "dvc repro snapshot_manifest" in failures[0]


# ── Provenance rules added after the first Phase 1 manifest named a revision ──
# that did not contain the code which produced its artifacts. Both rules exist
# so that failure is caught by the verifier rather than by a human reading a
# commit hash.
def test_verify_rejects_a_manifest_written_from_a_dirty_tree(snapshot_root, monkeypatch):
    """The exact defect that made the first Phase 1 snapshot untruthful."""
    monkeypatch.setattr(snapshot, "_git_dirty_paths", lambda: ["src/regime.py", "dvc.yaml"])
    snapshot.write_snapshot()

    failures = snapshot.verify_snapshot()
    assert failures, "a manifest built from uncommitted code must not verify"
    assert any("DIRTY" in f for f in failures)
    assert any("src/regime.py" in f for f in failures), "name what was uncommitted"


def test_verify_accepts_a_clean_tree_snapshot(snapshot_root, monkeypatch):
    monkeypatch.setattr(snapshot, "_git_dirty_paths", lambda: [])
    snapshot.write_snapshot()
    assert snapshot.verify_snapshot() == []


def test_manifest_records_whether_the_tree_was_dirty(snapshot_root, monkeypatch):
    monkeypatch.setattr(snapshot, "_git_dirty_paths", lambda: ["params.yaml"])
    manifest = snapshot.build_snapshot_manifest()
    assert manifest["git_dirty"] is True
    assert manifest["git_dirty_paths"] == ["params.yaml"]


def test_a_later_commit_of_the_same_history_still_verifies(snapshot_root, monkeypatch):
    """The manifest is itself a pipeline output, so the commit recording it is
    necessarily a CHILD of the commit that produced the artifacts. Requiring
    exact equality made a release tag unsatisfiable by construction."""
    monkeypatch.setattr(snapshot, "_git_dirty_paths", lambda: [])
    monkeypatch.setattr(snapshot, "_git_commit", lambda: "aaaaaaa_producing_commit")
    snapshot.write_snapshot()

    monkeypatch.setattr(snapshot, "_git_commit", lambda: "bbbbbbb_child_commit")
    monkeypatch.setattr(snapshot, "_commit_is_ancestor_of_head", lambda commit: True)
    assert snapshot.verify_snapshot() == []


def test_an_unrelated_revision_is_still_rejected(snapshot_root, monkeypatch):
    """Ancestry is looser than equality, but must still reject a different history."""
    monkeypatch.setattr(snapshot, "_git_dirty_paths", lambda: [])
    monkeypatch.setattr(snapshot, "_git_commit", lambda: "aaaaaaa_producing_commit")
    snapshot.write_snapshot()

    monkeypatch.setattr(snapshot, "_git_commit", lambda: "zzzzzzz_other_branch")
    monkeypatch.setattr(snapshot, "_commit_is_ancestor_of_head", lambda commit: False)
    failures = snapshot.verify_snapshot()
    assert any("not an ancestor" in f for f in failures)


def test_the_pinned_environment_is_a_snapshot_input(snapshot_root, monkeypatch):
    """A `>=` range does not identify the environment a result came from."""
    assert Path("requirements.lock.txt") in snapshot.SNAPSHOT_FILES

    monkeypatch.setattr(snapshot, "_git_dirty_paths", lambda: [])
    snapshot.write_snapshot()
    (snapshot_root / "requirements.lock.txt").write_text("xgboost==99.0.0\n", encoding="utf-8")
    assert any("requirements.lock.txt" in f for f in snapshot.verify_snapshot())
