"""Integrity checks for the standalone final PFA report.

The report is a release surface: a stale sentence or screenshot is a data defect,
not merely a documentation typo. These tests protect the facts introduced by
the final edition without changing the canonical report under ``docs/rapport``.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "rapport_final"
GOLD = ROOT / "data" / "gold"


def _sources() -> str:
    paths = [REPORT / "main.tex", *sorted((REPORT / "chapters").glob("*.tex")), *sorted((REPORT / "frontmatter").glob("*.tex"))]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths if path.name != "resume_ar.tex")


def test_final_report_omits_the_arabic_summary() -> None:
    main = (REPORT / "main.tex").read_text(encoding="utf-8")
    assert "ARABICRESUME" not in main
    assert "resume_ar" not in main
    assert not (REPORT / "frontmatter" / "resume_ar.tex").exists()


def test_final_report_contains_the_new_chapters_once() -> None:
    main = (REPORT / "main.tex").read_text(encoding="utf-8")
    assert main.count(r"\input{chapters/Chapter6}") == 1
    assert main.count(r"\input{chapters/Chapter7}") == 1


def test_final_report_uses_current_canonical_result() -> None:
    showcase = json.loads((GOLD / "dashboard_showcase.json").read_text())
    full = showcase["universes"]["full_2021"]
    regime = full["strategies"]["regime_conditional"]["sharpe_net"]
    classical = full["strategies"]["max_sharpe"]["sharpe_net"]
    delta_pct = full["headline_lift_pct"]
    chapter = (REPORT / "chapters" / "Chapter6.tex").read_text(encoding="utf-8")
    assert f"{regime:.4f}".replace(".", ",") in chapter
    assert f"{classical:.4f}".replace(".", ",") in chapter
    assert f"{delta_pct:.2f}".replace(".", "{,}") in chapter


def test_final_report_states_numeraire_per_universe() -> None:
    manifest = json.loads((GOLD / "currency_manifest.json").read_text())
    chapter = (REPORT / "chapters" / "Chapter6.tex").read_text(encoding="utf-8")
    assert manifest["universes"]["full_2021"]["base_currency"] == "MAD"
    assert manifest["universes"]["etf_2017"]["base_currency"] == "USD"
    assert "entièrement exprimé en \\textbf{MAD}" in chapter
    assert "univers \\textbf{USD mono-devise}" in chapter


def test_final_report_figures_are_present_and_referenced() -> None:
    source = _sources()
    required = {
        "architecture_globale.pdf",
        "revisions_resultat.pdf",
        "numeraire.pdf",
        "chaine_preuve.pdf",
        "protocoles.pdf",
        "multiple_testing.pdf",
        "explicabilite.pdf",
        "strategie_cout.pdf",
        "industrialisation.pdf",
        "api_swagger.png",
    }
    for name in required:
        assert (REPORT / "assets" / "figures" / name).is_file(), name
        assert name in source, name


def test_final_report_has_no_retired_test_count() -> None:
    source = _sources()
    assert "plus de 580" not in source
    assert "more than 580" not in source
    assert source.count("788") >= 4
    assert "781 lignes" in source


def test_final_report_does_not_promote_the_prototype() -> None:
    source = _sources().lower()
    assert "monitoring prêt, mais non actif" in source
    assert "aucune route d'ordre" in source
    assert "pas « validé pour la\nproduction »" in source
    assert "ne prétend pas battre" in source
