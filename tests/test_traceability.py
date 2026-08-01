"""
test_traceability.py — The supervisor's first requirement, made executable.

CLAUDE.md §18 states it as a hard gate:

    "Traceability first. Every component states which of P1–P4 it addresses;
     untraceable code fails review."

and §15.9 fixes the format: every function docstring carries
`Addresses: P1, P2 — ...`. Until now that was a convention — followed well
(107 of 110 public functions satisfy it) but enforced by nobody, so the 108th
could slip in unnoticed and only surface in review.

This file turns it into a build failure, which is this project's established
pattern for a rule that matters: the engine enforces the weight cap rather than
trusting strategies, `conftest._no_network` enforces hermeticity rather than
documenting it, and `test_run_dashboard_data` forbids hardcoded Sharpe literals
rather than asking nicely. Traceability was the last load-bearing rule still
resting on discipline.

WHAT COUNTS AS TRACED. A public function is traced if `Addresses:` appears in
its own docstring, its enclosing class's docstring, or its module's docstring.
Class- and module-level attribution is deliberate and not a loophole:
`strategies.py` documents each strategy's P-mapping on the CLASS, because that
is where the design decision lives — the `fit()` method just implements it.
Requiring per-method repetition would produce copy-paste noise, which is how
docstrings start lying.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# Explicitly grandfathered, with a reason each. This list should only ever
# SHRINK — adding to it is a decision someone has to defend in review, which is
# the point of writing it down rather than making the rule advisory.
GRANDFATHERED = {
    "pipeline.py::run_phase1":
        "orchestration entry point — sequences the traced Bronze/Silver/Gold "
        "functions and addresses no problem of its own.",
    "utils.py::query_gold":
        "thin DuckDB convenience wrapper over Gold parquet; infrastructure.",
    "utils.py::setup_logging":
        "logging configuration; infrastructure.",
}

VALID_PROBLEMS = {"P1", "P2", "P3", "P4"}


def _public_functions() -> list[tuple[str, str, bool]]:
    """(qualified_name, addresses_text, is_traced) for every public function."""
    out: list[tuple[str, str, bool]] = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_doc = ast.get_docstring(tree) or ""

        def visit(node, inherited: str) -> None:
            for child in node.body:
                if isinstance(child, ast.ClassDef):
                    visit(child, inherited + "\n" + (ast.get_docstring(child) or ""))
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_"):
                        continue
                    own = ast.get_docstring(child) or ""
                    combined = inherited + "\n" + own
                    out.append((f"{path.name}::{child.name}", combined,
                                "Addresses:" in combined))

        visit(tree, module_doc)
    return out


class TestEveryComponentIsTraceable:
    def test_all_public_functions_map_to_a_problem(self):
        untraced = [
            name for name, _, traced in _public_functions()
            if not traced and name not in GRANDFATHERED
        ]
        assert not untraced, (
            "CLAUDE.md §18: 'Every component states which of P1–P4 it addresses; "
            "untraceable code fails review.'\n\nMissing an `Addresses:` line (on the "
            "function, its class, or its module):\n  "
            + "\n  ".join(untraced)
            + "\n\nAdd `Addresses: P<n> — why` to the docstring, or — if it is genuinely "
              "infrastructure that solves none of P1–P4 — add it to GRANDFATHERED with a "
              "written reason."
        )

    def test_grandfather_list_has_not_grown_silently(self):
        """The exception list is allowed to shrink, never to grow unnoticed."""
        assert len(GRANDFATHERED) <= 3, (
            f"GRANDFATHERED now has {len(GRANDFATHERED)} entries. It is meant to shrink "
            f"as infrastructure gets traced, not to absorb new untraced code."
        )

    def test_grandfathered_entries_still_exist(self):
        """A stale exemption silently weakens the rule for a function that is
        gone — and would mask a genuinely untraced replacement."""
        known = {name for name, _, _ in _public_functions()}
        stale = [n for n in GRANDFATHERED if n not in known]
        assert not stale, (
            f"GRANDFATHERED names functions that no longer exist: {stale}. "
            f"Remove them so the exemption list stays honest."
        )


class TestProblemReferencesAreValid:
    def test_only_p1_to_p4_are_cited(self):
        """`Addresses: P5` would be a typo pointing at a problem that does not
        exist — the docstring would look traced while tracing nothing."""
        bad: list[str] = []
        for name, doc, traced in _public_functions():
            if not traced:
                continue
            for line in doc.splitlines():
                if "Addresses:" not in line:
                    continue
                cited = set(re.findall(r"\bP(\d+)\b", line))
                invalid = {f"P{c}" for c in cited} - VALID_PROBLEMS
                if invalid:
                    bad.append(f"{name}: {sorted(invalid)}")
        assert not bad, (
            "Docstrings cite problems outside P1–P4 (see CLAUDE.md §2):\n  "
            + "\n  ".join(bad)
        )

    def test_every_addresses_line_names_at_least_one_problem(self):
        """`Addresses:` with no P-number is decoration, not traceability."""
        empty: list[str] = []
        for name, doc, traced in _public_functions():
            if not traced:
                continue
            lines = [ln for ln in doc.splitlines() if "Addresses:" in ln]
            # The P-number may wrap onto the following line, so check the whole
            # docstring rather than the matched line in isolation.
            if lines and not re.search(r"\bP[1-4]\b", doc):
                empty.append(name)
        assert not empty, (
            "These have an `Addresses:` line that names no P1–P4 problem:\n  "
            + "\n  ".join(empty)
        )


def test_coverage_is_reported_for_visibility():
    """Not a gate — prints the ratio so a drop is visible in test output."""
    funcs = _public_functions()
    traced = sum(1 for _, _, t in funcs if t)
    pct = 100.0 * traced / len(funcs)
    print(f"\nP1–P4 traceability: {traced}/{len(funcs)} public functions ({pct:.0f}%), "
          f"{len(GRANDFATHERED)} grandfathered.")
    assert pct > 90.0, f"Traceability fell to {pct:.0f}% — below the 90% floor."
