"""
check_artifacts.py — refuse to serve an incomplete artifact bundle.

Addresses: P4 — the API serves committed results, so starting it against a
missing or partial `data/gold/` is not a degraded mode worth entering: every
response would be a 503 or, worse, a subset a caller might mistake for the
whole. Failing at start with the list of what is missing turns that into one
loud error instead of a stream of quiet ones.

Runs as the container's preflight. Also usable directly:

    python scripts/check_artifacts.py            # bundle present?
    python scripts/check_artifacts.py --verify   # bundle present AND verified
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"

# What the API and dashboard actually read. Kept narrower than
# snapshot.SNAPSHOT_FILES, which also covers Bronze inputs the serving path
# never touches — an API container should not need the raw price history to
# start.
REQUIRED = (
    "dashboard_showcase.json",
    "dashboard_equity.parquet",
    "dashboard_weights.parquet",
    "dashboard_regime.parquet",
    "phase5_results.json",
    "paired_comparison_results.json",
    "model_explanations.json",
    "monitoring_baseline.json",
)

# Served when present, omitted when absent. Listed so "optional" is a decision
# on the record rather than an accident of which endpoint someone tested.
OPTIONAL = ("crisis_windows.json", "snapshot_manifest.json")


def missing_artifacts() -> list[str]:
    return [name for name in REQUIRED if not (GOLD / name).is_file()]


def check(verify: bool = False) -> int:
    missing = missing_artifacts()
    if missing:
        print(
            "Artifact bundle incomplete — refusing to start.\n"
            + "\n".join(f"  missing: data/gold/{name}" for name in missing)
            + "\n\nRebuild with `dvc repro`, or fetch a published snapshot with "
            "`dvc pull`.",
            file=sys.stderr,
        )
        return 1

    absent_optional = [name for name in OPTIONAL if not (GOLD / name).is_file()]
    for name in absent_optional:
        print(f"note: optional artifact data/gold/{name} is absent.", file=sys.stderr)

    manifest_path = GOLD / "snapshot_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("git_dirty"):
            # Not fatal: a research container may legitimately serve a
            # work-in-progress bundle. But it must be said out loud, because
            # the manifest's commit does not identify the code that produced
            # these artifacts, and `snapshot.py verify` will reject it.
            print(
                "WARNING: snapshot manifest was generated from a DIRTY working "
                "tree. producer_commit does not identify the code that produced "
                "these artifacts.",
                file=sys.stderr,
            )
        print(f"Artifact bundle complete ({len(REQUIRED)} required files); "
              f"manifest commit {str(manifest.get('git_commit'))[:12]}.")
    else:
        print(f"Artifact bundle complete ({len(REQUIRED)} required files); "
              f"no snapshot manifest present.")

    if verify:
        sys.path.insert(0, str(ROOT / "src"))
        from snapshot import verify_snapshot

        failures = verify_snapshot()
        if failures:
            print("Snapshot verification failed:", *failures, sep="\n- ", file=sys.stderr)
            return 1
        print("Snapshot verification passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true",
        help="Also run full snapshot checksum verification (release gate).",
    )
    return check(parser.parse_args(argv).verify)


if __name__ == "__main__":
    raise SystemExit(main())
