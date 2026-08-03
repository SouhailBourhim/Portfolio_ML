"""
snapshot.py — create and verify a release manifest for the research snapshot.

Addresses: P4 — a result is reproducible only when its inputs, outputs, code
revision, and runtime are identified together. DVC records the stage graph; this
module makes the concrete published snapshot inspectable and rejects a changed
input or artifact before the dashboard/report is handed to a reviewer.

The manifest intentionally hashes files rather than embedding their contents.
Market data remains DVC-managed and out of Git, while the manifest can be shared
with a report, release archive, or future DVC remote without exposing data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RELATIVE_PATH = Path("data/gold/snapshot_manifest.json")

# Inputs and publication artifacts needed to reproduce every public number in
# the README, report, dashboard, and API. The manifest itself is excluded to
# avoid a self-referential checksum.
SNAPSHOT_FILES = (
    Path("params.yaml"),
    Path("dvc.yaml"),
    Path("requirements.txt"),
    # The pinned environment, not just the intended one: `requirements.txt`
    # uses `>=` ranges, which name a family of environments rather than the
    # one these numbers came from. xgboost in particular is version-sensitive
    # here (see the single-worker policy in `ml_signals`/`model_selection`).
    Path("requirements.lock.txt"),
    Path("data/bronze/raw_prices.parquet"),
    Path("data/bronze/bvc_prices.parquet"),
    Path("data/bronze/raw_macro.parquet"),
    Path("data/bronze/raw_bam_macro.parquet"),
    Path("data/gold/log_returns.parquet"),
    Path("data/gold/log_returns_etf.parquet"),
    Path("data/gold/ml_features_full.parquet"),
    Path("data/gold/ml_features_etf.parquet"),
    Path("data/gold/phase2_hurdle.json"),
    Path("data/gold/phase4_results.json"),
    Path("data/gold/phase4b_results.json"),
    Path("data/gold/phase4c_results.json"),
    Path("data/gold/phase5_results.json"),
    Path("data/gold/dsr_trial_ledger.json"),
    # Phase 2: the validation protocol actually applied, and the paired
    # comparisons. A results file without these cannot be audited for how its
    # hyperparameters were chosen or whether its differences were tested.
    Path("data/gold/phase5_validation_protocol.json"),
    Path("data/gold/paired_comparison_results.json"),
    # Phase 3: the explanation of the published decision. Hashed alongside the
    # results so a reviewer cannot be handed weights and an explanation of a
    # different run.
    Path("data/gold/model_explanations.json"),
    Path("data/gold/dashboard_showcase.json"),
    Path("data/gold/dashboard_equity.parquet"),
    Path("data/gold/dashboard_weights.parquet"),
    Path("data/gold/dashboard_regime.parquet"),
    Path("data/gold/crisis_windows.json"),
)


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a dataset into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    """Return the checked-out source revision, or ``unknown`` outside Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty_paths() -> list[str]:
    """Return tracked paths with uncommitted modifications, sorted.

    Addresses: P4 — recording only ``git_commit`` cannot distinguish a
    manifest built from that commit from one built from that commit plus
    uncommitted edits. That is not hypothetical: the first Phase 1 manifest
    named a revision whose tree did NOT contain the code that produced the
    artifacts, so a reviewer checking it out would have rebuilt something
    else. Untracked files are ignored deliberately — scratch files and tool
    output do not change what the pipeline computes.
    """
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line[3:].strip() for line in output.splitlines() if line.strip())


def _commit_is_ancestor_of_head(commit: str) -> bool:
    """Is ``commit`` reachable from HEAD?

    Addresses: P4 — the manifest is itself a pipeline output, so the commit
    that records it is necessarily a child of the commit that produced the
    artifacts. Demanding exact equality made a release tag impossible to
    satisfy by construction, which is what pushed the previous attempt into
    leaving the workspace dirty instead. Ancestry is the check that can
    actually hold at a tag while still rejecting a different branch, a
    rewritten history, or an unrelated revision.
    """
    if commit in ("", "unknown"):
        return False
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _parquet_metadata(path: Path) -> dict[str, object]:
    """Return small, reviewer-useful metadata for a Parquet artifact."""
    frame = pd.read_parquet(path)
    info: dict[str, object] = {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }
    if isinstance(frame.index, pd.DatetimeIndex) and not frame.empty:
        info["index_start"] = str(frame.index.min().date())
        info["index_end"] = str(frame.index.max().date())
    return info


def _file_record(relative_path: Path) -> dict[str, object]:
    """Build the checksum and lightweight structural metadata for one file."""
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Required snapshot file is missing: {relative_path}. "
            "Rebuild the pipeline before publishing results."
        )

    record: dict[str, object] = {
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".parquet":
        record["parquet"] = _parquet_metadata(path)
    elif path.suffix == ".json":
        # Reject a truncated/corrupted JSON artifact while keeping the manifest
        # compact; the checksum supplies the exact content identity.
        json.loads(path.read_text(encoding="utf-8"))
    return record


def build_snapshot_manifest() -> dict[str, object]:
    """Return the full manifest for the current required research snapshot."""
    dirty = _git_dirty_paths()
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        # A manifest written from a dirty tree does not identify a source
        # revision a reviewer can check out. Recorded so `verify` can reject
        # it rather than leaving the discrepancy for a human to notice.
        "git_dirty": bool(dirty),
        "git_dirty_paths": dirty,
        "python": platform.python_version(),
        "files": {str(path): _file_record(path) for path in SNAPSHOT_FILES},
        "notes": [
            "This manifest identifies a local/DVC-managed data snapshot; it does not publish data.",
            "A reviewer must obtain the matching data through the configured DVC remote or a release archive.",
            "Marginal confidence intervals in the results do not constitute a paired test of strategy superiority.",
            "git_commit names the revision that PRODUCED these artifacts; the commit recording "
            "this manifest is its child, so verification requires ancestry rather than equality.",
        ],
    }


def write_snapshot() -> Path:
    """Write the current snapshot manifest atomically and return its path."""
    manifest = build_snapshot_manifest()
    path = ROOT / MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def verify_snapshot() -> list[str]:
    """Return integrity mismatches between the stored manifest and local files."""
    path = ROOT / MANIFEST_RELATIVE_PATH
    if not path.is_file():
        return [
            f"Missing {MANIFEST_RELATIVE_PATH}. Run `dvc repro snapshot_manifest` "
            "after rebuilding the required stages."
        ]

    expected = json.loads(path.read_text(encoding="utf-8"))
    current = build_snapshot_manifest()
    failures: list[str] = []
    for relative_path in SNAPSHOT_FILES:
        key = str(relative_path)
        if expected.get("files", {}).get(key) != current["files"].get(key):
            failures.append(f"Snapshot mismatch: {key}")
    # A manifest produced from uncommitted code names a revision that does not
    # contain that code. Nothing downstream can detect this, so it is rejected
    # here rather than trusted.
    if expected.get("git_dirty"):
        paths = expected.get("git_dirty_paths") or []
        failures.append(
            "Snapshot was generated from a DIRTY working tree, so its "
            f"git_commit ({expected.get('git_commit')}) does not identify the code that "
            f"produced these artifacts ({len(paths)} uncommitted tracked path(s), e.g. "
            f"{', '.join(paths[:3]) or 'n/a'}). Commit the changes, then regenerate with "
            "`dvc repro --force snapshot_manifest`."
        )

    recorded = expected.get("git_commit", "unknown")
    if recorded != current["git_commit"] and not _commit_is_ancestor_of_head(recorded):
        failures.append(
            f"Snapshot mismatch: recorded revision {recorded} is not an ancestor of the "
            f"checked-out revision {current['git_commit']} — this is a different history, "
            "not a later commit of the same one."
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    """CLI: ``write`` a manifest or ``verify`` the stored snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify"))
    args = parser.parse_args(argv)

    if args.command == "write":
        print(write_snapshot().relative_to(ROOT))
        return 0

    failures = verify_snapshot()
    if failures:
        print("Snapshot verification failed:", *failures, sep="\n- ", file=sys.stderr)
        return 1
    print("Snapshot verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
