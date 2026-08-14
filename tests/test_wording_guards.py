"""
test_wording_guards.py — Retracted claims, made un-reintroducible.

This project has now been bitten twice by the same class of defect: a claim is
an artifact too. The 2026-08-02 reframing (AGENTS.md §5.2) had to be applied to
six surfaces plus a binary deliverable plus a JSON field the API re-serves, and
§17.11 records that nothing in the suite could tell they had drifted apart —
`test_artifact_consistency.py` compares NUMBERS, not WORDING.

This file compares wording. It guards claims that were found to be WRONG, not
claims that are merely awkward, so a failure here is always a correctness
regression rather than a style opinion.

── GUARD 1: the cap arithmetic ────────────────────────────────────────────────

The retracted claim:

    "With five assets and a 25% cap, every feasible portfolio must hold four
     assets AT THE CAP."

This is false. With w_i >= 0, sum(w) = 1 and w_i <= 0.25, equal weight
(0.20 each) is feasible with NOTHING at the cap, as is (0.25, 0.25, 0.25, 0.15,
0.10). What the arithmetic actually gives is:

  * at least ceil(1 / 0.25) = 4 assets must carry POSITIVE WEIGHT, since
    1 = sum(w) <= 0.25 * |{i : w_i > 0}|;
  * no asset can exceed equal weight (0.20) by more than 5 percentage points.

The feasible region stays non-empty and multidimensional. That all three
objectives nonetheless landed on the SAME corner is an EMPIRICAL finding —
`min_variance_lw` emitting one allocation across 248 rebalances at a 0.25 cap
versus 171 at 0.30 — not a theorem. The numeric results never changed; only the
causal explanation was wrong, and it had propagated to eleven surfaces in two
languages including both report trees and two Gold-artifact generators.

Say "empirically cap-dominated", never "mathematically cap-determined".

── WHY THERE IS NO SIGNIFICANCE GUARD HERE ────────────────────────────────────

A second guard against the §5.2 retractions ("statistically significant" /
"statistically indistinguishable") was written and then DELIBERATELY REMOVED.
Every occurrence of those phrases still in the tree turned out to be correct:
they are prohibitions ("must not be used to claim that the project has
established a statistically significant detection result"), search strings in
the .docx rewriters that REPLACE the retracted wording, or a different subject
entirely (significant autocorrelation in the Phase 1 ACF discussion).

A lexical guard cannot separate "X is statistically significant" from "nothing
here is statistically significant" without negation parsing, and a guard that
cries wolf is worse than no guard — the suite's authority depends on a failure
always meaning a defect. The cap claim is guardable precisely because its false
form ("four assets AT THE CAP") and its true form ("four assets with POSITIVE
WEIGHT") are lexically distinct.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Surfaces a reader or supervisor actually sees. Deliberately includes the
# LaTeX sources and the dashboard, because both carry claims to an audience.
SEARCH_DIRS = ("src", "scripts", "experiments", "dashboard", "docs", "notebooks")
SEARCH_SUFFIXES = (".py", ".md", ".tex", ".ipynb", ".yaml")

# Paths that legitimately QUOTE a retracted claim in order to retract it.
# Every entry must be a file whose purpose is to record the correction.
ALLOWED = {
    "tests/test_wording_guards.py",
    "docs/GLOBAL_UNIVERSE_PREREGISTRATION.md",
}


def _iter_files():
    for d in SEARCH_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in SEARCH_SUFFIXES or not path.is_file():
                continue
            if any(part in {".ipynb_checkpoints", "__pycache__"} for part in path.parts):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in ALLOWED:
                continue
            yield path, rel


# Each pattern is the FALSE causal form. The corrected form ("four assets with
# positive weight", "quatre actifs a poids strictement positif") must not match.
CAP_PATTERNS = [
    # English: "...four assets at the cap", "...4 assets at the cap"
    re.compile(r"(?:four|4)\s+assets?\s+(?:\w+\s+){0,2}?at\s+the\s+cap", re.I),
    # English: "must hold most assets at the cap"
    re.compile(r"most\s+assets?\s+at\s+the\s+cap", re.I),
    # English: "forces >=4 assets to the cap" / "forced to the cap". \S+ rather
    # than \w+ so that "4", ">=4" and "≥4" all count.
    re.compile(r"forc(?:es|ed|ing)\s+(?:\S+\s+){0,4}?to\s+the\s+cap", re.I),
    # French: "quatre actifs au plafond" / "4 actifs au plafond"
    re.compile(r"(?:quatre|4)\s+actifs?\s+(?:\w+\s+){0,2}?au\s+plafond", re.I),
    # French: "forcés au plafond"
    re.compile(r"forc[ée]s?\s+au\s+plafond", re.I),
]


@pytest.mark.parametrize("pattern", CAP_PATTERNS, ids=lambda p: p.pattern[:40])
def test_false_cap_arithmetic_claim_cannot_return(pattern):
    """The retracted claim is that the cap DETERMINES the corner. It does not.

    See this module's docstring for the proof. Equal weight is feasible with
    nothing at the cap, so no arithmetic forces four assets to the cap.
    """
    offenders = []
    for path, rel in _iter_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{rel}:{line}: {match.group(0)!r}")

    assert not offenders, (
        "Retracted cap claim reintroduced. The arithmetic requires at least four "
        "assets with POSITIVE WEIGHT, not four assets AT THE CAP — equal weight "
        "(20% each) is feasible with nothing at the cap. Say 'empirically "
        "cap-dominated', never 'mathematically cap-determined'.\n  "
        + "\n  ".join(offenders)
    )


# The RENDERED surfaces a reader actually receives. Source checks are not
# enough: the 2026-08 MAD defect survived them because live prose had been
# fixed while a static table and a screenshot still showed old values
# (`test_final_report.py` documents that failure). A .tex file nobody
# recompiles is not a corrected report.
RENDERED_PDFS = (
    "output/pdf/Rapport_PFA_Final_2026.pdf",   # the DISTRIBUTED file, README-linked
    "docs/rapport/main.pdf",
    "docs/rapport_final/main.pdf",             # gitignored build output
)


@pytest.mark.parametrize("rel", RENDERED_PDFS)
def test_rendered_reports_do_not_carry_the_false_cap_claim(rel):
    """Check the PDF the reader receives, not only the source it came from.

    Skips when the PDF or `pdftotext` is absent, matching the established
    pattern in `test_final_report.py` — a fresh clone has neither.
    """
    pdf = ROOT / rel
    if not pdf.is_file() or not shutil.which("pdftotext"):
        pytest.skip(f"{rel} or pdftotext absent — rebuild the report to check it.")

    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True, capture_output=True, text=True, timeout=300,
    )
    # Collapse the hard line wrapping LaTeX introduces, so a claim split
    # across two lines is still caught.
    text = re.sub(r"\s+", " ", result.stdout)

    offenders = [
        match.group(0)
        for pattern in CAP_PATTERNS
        for match in pattern.finditer(text)
    ]
    assert not offenders, (
        f"{rel} still renders the retracted cap claim: {offenders}. "
        "Recompile the report after correcting its .tex source — and remember "
        "output/pdf/ is a COPY, so rebuilding docs/rapport_final alone does not "
        "update the distributed file."
    )


def test_the_guard_actually_matches_the_retracted_forms():
    """A guard that matches nothing is a guard that proves nothing.

    Locks in that each pattern fires on the exact retracted sentences, so a
    future refactor cannot quietly defang the regex while leaving it green.
    """
    retracted = [
        "every feasible long-only portfolio must hold at least four assets at the cap",
        "must hold most assets at the cap",
        "5 x 0.25 = 1.25 forces >=4 assets to the cap",
        "tout portefeuille admissible doit placer au moins quatre actifs au plafond",
        "au moins 4 actifs sont forces au plafond",
    ]
    for sentence in retracted:
        assert any(p.search(sentence) for p in CAP_PATTERNS), (
            f"No CAP_PATTERN matches a known-retracted sentence: {sentence!r}"
        )

    corrected = [
        "must hold at least four assets with positive weight",
        "doit detenir au moins quatre actifs a poids strictement positif",
        "the constraint empirically dominates the objective",
    ]
    for sentence in corrected:
        assert not any(p.search(sentence) for p in CAP_PATTERNS), (
            f"A CAP_PATTERN falsely flags corrected wording: {sentence!r}"
        )
