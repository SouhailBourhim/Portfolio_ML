"""Integrity checks for the standalone final PFA report.

The report is a release surface: a stale sentence or screenshot is a data defect,
not merely a documentation typo. These tests protect the facts introduced by
the final edition without changing the canonical report under ``docs/rapport``.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "rapport_final"
GOLD = ROOT / "data" / "gold"
PDF = ROOT / "output" / "pdf" / "Rapport_PFA_Final_2026.pdf"


def _require_gold(*names: str) -> None:
    missing = [name for name in names if not (GOLD / name).is_file()]
    if missing:
        pytest.skip(f"Gold artifacts absent — run `dvc pull`: {', '.join(missing)}")


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
    _require_gold("dashboard_showcase.json")
    showcase = json.loads((GOLD / "dashboard_showcase.json").read_text())
    full = showcase["universes"]["full_2021"]
    regime = full["strategies"]["regime_conditional"]["sharpe_net"]
    classical = full["strategies"]["max_sharpe"]["sharpe_net"]
    delta_pct = full["headline_lift_pct"]
    chapter = (REPORT / "chapters" / "Chapter6.tex").read_text(encoding="utf-8")
    assert f"{regime:.4f}".replace(".", ",") in chapter
    assert f"{classical:.4f}".replace(".", ",") in chapter
    assert f"{delta_pct:.2f}".replace(".", "{,}") in chapter


def test_final_report_headline_table_matches_the_current_mad_release() -> None:
    """The headline table is a release surface, not a decorative summary.

    This specifically prevents the pre-MAD 1.236/1.164 table from coexisting
    with current prose that correctly states 0.9571/1.0690.
    """
    _require_gold("dashboard_showcase.json")
    showcase = json.loads((GOLD / "dashboard_showcase.json").read_text())
    full = showcase["universes"]["full_2021"]["strategies"]
    chapter = (REPORT / "chapters" / "Chapter5.tex").read_text(encoding="utf-8")
    for strategy in ("regime_conditional", "max_sharpe", "equal_weight", "min_variance_lw"):
        sharpe = f"{full[strategy]['sharpe_net']:.3f}".replace(".", ",")
        assert sharpe in chapter
    assert "1,236" not in chapter
    assert "1,164" not in chapter


def test_final_report_describes_the_deep_morocco_experiment_precisely() -> None:
    """Pre-2021 BVC data was studied; it is not absent, nor release-grade yet."""
    source = " ".join(_sources().split())
    assert "2005--2024" in source
    assert "56\\,184" in source
    assert "sans établir un avantage de portefeuille" in source
    assert "contractualisés" in source
    assert "aucune source gratuite ne fournit" not in source


def test_final_report_states_the_established_scope_of_search_correction() -> None:
    conclusion = (REPORT / "chapters" / "Conclusion.tex").read_text(encoding="utf-8")
    assert "correction pour le nombre d'essais est établie" in conclusion
    assert "240" in conclusion
    assert "Huit comparaisons externes" in conclusion


def test_rendered_final_report_cannot_mix_mad_prose_with_precorrection_metrics() -> None:
    """The reader receives the PDF, not the TeX source.

    The previous defect survived source checks because live prose had been fixed
    while a static headline table and a screenshot still showed pre-MAD values.
    """
    if not PDF.is_file() or not shutil.which("pdftotext"):
        return
    result = subprocess.run(
        ["pdftotext", "-layout", str(PDF), "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    text = re.sub(r"\s+", " ", result.stdout.replace("−", "-"))
    assert "0,9571 contre 1,0690" in text
    assert "1,236" not in text
    assert "1,164" not in text
    assert "56 184 lignes" in text


def test_final_report_states_numeraire_per_universe() -> None:
    _require_gold("currency_manifest.json")
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


def test_final_report_reports_a_non_stale_test_suite_scale() -> None:
    source = _sources()
    assert "plus de 580" not in source
    assert "more than 580" not in source
    assert "plus de 780" in source
    assert "more than 780" in source
    assert "788" not in source
    assert "781 lignes" in source


def test_final_report_does_not_promote_the_prototype() -> None:
    source = _sources().lower()
    assert "monitoring prêt, mais non actif" in source
    assert "aucune route d'ordre" in source
    assert "pas « validé pour la\nproduction »" in source
    assert "ne prétend pas battre" in source
