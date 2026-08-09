"""
test_rendered_pdf.py — check the PDF a reader actually receives.

Why this exists, precisely. Every other guard in this suite reads SOURCE:
release_facts strings, TeX files, Python surfaces. All of them passed while the
built report carried the retired `+6,2 %` headline on four pages, because LaTeX
writes it `$+6{,}2$~\\%` and the source scanners matched literal forms. The
rendered text is the only representation where every encoding — `6,2~\\%`,
`$+6{,}2$~\\%`, `\\textbf{$+6{,}2$~\\%}` — collapses to the same characters.

So this reads the PDF, normalises whitespace and separators, and forbids the
retired positive claims outside passages explicitly marked historical.

Skipped when `main.pdf` or `pdftotext` is absent, so a fresh clone without a
TeX toolchain is not blocked; the release gate runs it after building.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "docs" / "rapport" / "main.pdf"

# Marks a page as an explicitly historical passage. Anything else is a live claim.
HISTORY_MARKERS = ("superseded", "estimation initiale", "après correction des dividendes")

# Retired POSITIVE claims about the regime strategy, in normalised form. The
# numbers alone are not enough: "1,6" appears legitimately as the etf_2017
# difference, so the patterns pin the sign and the context.
RETIRED_PATTERNS = (
    r"\+\s*6[.,]2\s*%",          # +6,2 %  in any spacing
    r"\+\s*14[.,]3\s*%",         # the pre-dividend estimate
    r"gain de\s*\+?\s*6[.,]2",   # "soit un gain de +6,2"
    r"1[.,]2363",
    r"1[.,]1644",
)


def _pdf_pages() -> list[str]:
    if not PDF.is_file():
        pytest.skip("main.pdf not built — run `tectonic -X compile main.tex`.")
    if not shutil.which("pdftotext"):
        pytest.skip("pdftotext unavailable (poppler).")
    out = subprocess.run(
        ["pdftotext", "-layout", str(PDF), "-"],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        pytest.skip(f"pdftotext failed: {out.stderr[:200]}")
    return out.stdout.split("\f")


def _normalise(text: str) -> str:
    """Collapse the variation LaTeX introduces, so one pattern matches them all.

    Non-breaking and thin spaces become ordinary ones, runs of whitespace
    collapse, and the French decimal comma is left intact (the patterns accept
    either separator).
    """
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    text = text.replace("−", "-")          # unicode minus -> hyphen
    return re.sub(r"\s+", " ", text)


class TestTheRenderedReportMakesNoRetiredPositiveClaim:
    def test_no_page_states_a_retired_gain_outside_a_historical_passage(self):
        offenders = []
        for number, raw in enumerate(_pdf_pages(), 1):
            page = _normalise(raw)
            if any(m in page.lower() for m in HISTORY_MARKERS):
                continue                      # explicitly marked as history
            for pattern in RETIRED_PATTERNS:
                for hit in re.finditer(pattern, page):
                    lo = max(0, hit.start() - 90)
                    offenders.append(
                        f"p{number}: …{page[lo:hit.end() + 60].strip()}…"
                    )
        assert not offenders, (
            "the rendered PDF states retired positive claims outside a passage "
            "marked historical:\n  " + "\n  ".join(offenders[:12])
        )

    def test_the_current_figure_is_actually_present(self):
        """Non-vacuity: a PDF that simply lost the results section would pass the
        test above. The live figure must be there."""
        pages = [_normalise(p) for p in _pdf_pages()]
        assert any(re.search(r"-\s*10[.,]47\s*%", p) for p in pages), (
            "the rendered report does not state the current -10,47 % difference"
        )

    def test_the_paired_test_caveat_travels_with_it(self):
        pages = [_normalise(p) for p in _pdf_pages()]
        assert any("aucun test pairé" in p.lower() for p in pages)

    def test_the_numeraire_of_each_universe_is_stated(self):
        joined = " ".join(_normalise(p) for p in _pdf_pages()).lower()
        assert "bank al-maghrib" in joined
        assert "non couvert" in joined
