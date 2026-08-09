"""
provenance.py — make an artifact state what it was computed from, and refuse it
when that no longer holds.

Addresses: P4 — a result that cannot say which data produced it cannot be
audited, and a consumer that cannot check will eventually render a stale number
beside a current one.

THE INCIDENT THIS EXISTS FOR. `data/gold/nested_walkforward_results.json` was
produced on 2026-07-28 from a MIXED-CURRENCY `full_2021` universe. The
base-currency correction then rebuilt every pipeline artifact, and this one
survived untouched — it was neither a DVC output nor carried any statement of
its inputs. Chapter 5's figure builder would have consumed it happily and
rendered "the nested protocol puts the regime system back in front" (true on the
old data, sign-flipped on the new) beside numbers from the rebuilt tree. Nothing
in the test suite could have told: `test_artifact_consistency.py` compares
numbers ACROSS artifacts, not an artifact against the data underneath it.

TWO MECHANISMS, deliberately different in kind:

  1. `build_provenance` records the numéraire, the exact data and OOS ranges,
     the Git revision, and a SHA-256 of every source artifact.
  2. `require_current_artifact` refuses to consume the result if the numéraire
     is absent or wrong, or if any recorded source hash no longer matches the
     file on disk.

The second is the one that matters. A date or an mtime says when a file was
written; a source hash says whether the data it was computed from is still the
data in the tree. Only the latter survives a `dvc checkout`, a fresh clone, or a
file copied between worktrees — all of which happen routinely here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class StaleArtifactError(RuntimeError):
    """
    An artifact cannot be consumed: it predates a correction, was computed in a
    different numéraire, or its inputs have changed underneath it.

    Addresses: P4 — the alternative is a presentation surface mixing results
    from two different datasets, which is undetectable by eye and produced
    exactly one sign-flipped claim in this project's history.
    """


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def git_revision(root: Path = ROOT) -> str:
    """
    Addresses: P4 — pins the code that produced a result, so a rerun can be
    reproduced rather than approximated.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}" if rev else "unknown"
    except Exception:
        return "unknown"


def read_numeraire(universe: str, root: Path = ROOT) -> dict:
    """
    Read a universe's currency policy from the Gold manifest.

    Addresses: P1, P4 — the numéraire is copied from the canonical manifest
    rather than restated, so an artifact cannot claim a currency the data layer
    disagrees with.

    Raises:
        StaleArtifactError: if the manifest is missing or has no entry, which
            means the tree predates the base-currency correction.
    """
    path = root / "data" / "gold" / "currency_manifest.json"
    if not path.is_file():
        raise StaleArtifactError(
            f"{path} is absent, so no artifact in this tree can state its "
            f"numéraire. The tree predates the base-currency correction; run "
            f"`dvc repro features`."
        )
    manifest = json.loads(path.read_text())
    entry = manifest.get("universes", {}).get(universe)
    if not entry:
        raise StaleArtifactError(
            f"currency_manifest.json has no entry for {universe!r} "
            f"(present: {sorted(manifest.get('universes', {}))})."
        )
    return entry


def build_provenance(
    universe: str,
    returns: pd.DataFrame,
    oos_index=None,
    source_artifacts: list[str] | None = None,
    root: Path = ROOT,
) -> dict:
    """
    Assemble the provenance block persisted inside a result artifact.

    Addresses: P4 — see module docstring.

    Args:
        universe: Universe key, e.g. "full_2021".
        returns: The returns matrix the result was computed from.
        oos_index: Optional out-of-sample index, recorded separately from the
            full data range because they answer different questions.
        source_artifacts: Repo-relative paths whose contents this result depends
            on. Each is hashed; a later mismatch marks the result stale.
        root: Repository root.

    Returns:
        Dict embedded under the artifact's `provenance` key.
    """
    numeraire = read_numeraire(universe, root=root)
    record = {
        "universe": universe,
        "base_currency": numeraire.get("base_currency"),
        "hedge_status": numeraire.get("hedge_status", "unhedged"),
        "currency_converted": numeraire.get("converted"),
        "fx_series": numeraire.get("fx_series"),
        "data_range": {
            "start": str(pd.Timestamp(returns.index.min()).date()),
            "end": str(pd.Timestamp(returns.index.max()).date()),
            "n_rows": int(len(returns)),
            "n_assets": int(returns.shape[1]),
        },
        "git_revision": git_revision(root),
        "generated_at": pd.Timestamp.now().isoformat(),
        "source_artifacts": {},
    }
    if oos_index is not None and len(oos_index):
        record["oos_range"] = {
            "start": str(pd.Timestamp(min(oos_index)).date()),
            "end": str(pd.Timestamp(max(oos_index)).date()),
            "n_rows": int(len(oos_index)),
        }
    for rel in source_artifacts or []:
        record["source_artifacts"][rel] = _sha256(root / rel)
    return record


def require_current_artifact(
    artifact: dict | str | Path,
    expect_universe: str | None = None,
    expect_base_currency: str = "MAD",
    check_source_hashes: bool = True,
    root: Path = ROOT,
) -> dict:
    """
    Refuse an artifact that is stale, mis-denominated, or missing provenance.

    Addresses: P1, P4 — this is the gate that would have caught the 2026-07-28
    nested walk-forward artifact surviving the base-currency correction. It
    fails on CONTENT: an artifact with no `provenance` block predates the
    correction by construction, and one whose recorded source hashes no longer
    match was computed from data that is no longer in the tree.

    Args:
        artifact: The loaded dict, or a path to the JSON.
        expect_universe: Universe the consumer expects, if it cares.
        expect_base_currency: Required numéraire. "MAD" for full_2021; pass
            "USD" for etf_2017, which is single-currency and unconverted.
        check_source_hashes: Compare recorded hashes against the files on disk.
        root: Repository root.

    Returns:
        The artifact dict, unchanged, once every check passes.

    Raises:
        StaleArtifactError: naming what failed and how to refresh it.
    """
    if isinstance(artifact, (str, Path)):
        path = Path(artifact)
        if not path.is_file():
            raise StaleArtifactError(
                f"{path} is absent. It is a declared DVC output — run "
                f"`dvc repro nested_walkforward` (or `dvc pull`) before building "
                f"any surface that consumes it."
            )
        label = str(path)
        artifact = json.loads(path.read_text())
    else:
        label = artifact.get("universe", "<artifact>")

    prov = artifact.get("provenance")
    if not prov:
        raise StaleArtifactError(
            f"{label} carries no `provenance` block, so it cannot state which "
            f"data produced it. Every artifact written before the base-currency "
            f"correction is in this state, and consuming one would render "
            f"pre-correction numbers beside current ones. Regenerate it."
        )

    if expect_universe and prov.get("universe") != expect_universe:
        raise StaleArtifactError(
            f"{label} is for universe {prov.get('universe')!r}, not "
            f"{expect_universe!r}."
        )

    actual = prov.get("base_currency")
    if actual != expect_base_currency:
        raise StaleArtifactError(
            f"{label} is denominated in {actual!r}, not {expect_base_currency!r}. "
            f"A mixed-currency or pre-correction result must not be presented "
            f"beside MAD-valued ones — that is precisely the defect the "
            f"base-currency correction removed."
        )

    if check_source_hashes:
        drifted = []
        for rel, recorded in (prov.get("source_artifacts") or {}).items():
            current = _sha256(root / rel)
            if recorded is not None and current is not None and current != recorded:
                drifted.append(rel)
        if drifted:
            raise StaleArtifactError(
                f"{label} was computed from {len(drifted)} source artifact(s) that "
                f"have since changed: {drifted}. The result no longer describes the "
                f"data in this tree. Regenerate it rather than presenting it."
            )
    return artifact
