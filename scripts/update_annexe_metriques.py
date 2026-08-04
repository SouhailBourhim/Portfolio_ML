"""Remove the retracted equivalence claim from the evaluation-metrics annex.

Addresses: P4 — this annex described the block bootstrap as the tool that
"autorise la conclusion « statistiquement indiscernables »". It does not. The
bootstrap produces MARGINAL intervals, one per strategy; overlapping marginal
intervals are not a test of the difference between two strategies, and failing
to reject is not accepting the null. The annex was therefore teaching the
wrong inference in the very document that explains the project's evaluation
metrics — the worst possible place for it.

What the bootstrap genuinely does is prevent a point ranking from being
presented as a fact. What tests a difference is the PAIRED bootstrap on the
return differences, which is a different instrument and is named as such.

Targeted edits: this file has no generator, and the replacement text belongs
in reviewable source rather than only inside a binary. Idempotent — each
replacement is skipped once the new wording is present.

NOTE for whoever next revises this annex: it documents the block bootstrap
(§4.2) but has no section on the PAIRED bootstrap, which is the instrument
that actually tests a difference and the basis of every comparative claim
the project now makes. The text below therefore points at the code rather
than at a section number. Adding that section is worthwhile and is left as
a deliberate follow-up, not done here.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "Annexe_Metriques_Evaluation.docx"

REPLACEMENTS = (
    (
        "et c'est lui qui autorise la conclusion « statistiquement "
        "indiscernables » au lieu d'un classement ponctuel présenté comme un fait",
        "et c'est lui qui empêche de présenter un classement ponctuel comme un "
        "fait. Il n'autorise pas pour autant la conclusion inverse : des "
        "intervalles MARGINAUX qui se chevauchent ne testent pas la différence "
        "entre deux stratégies. Tester cette différence exige un bootstrap "
        "PAIRÉ, mené sur les écarts de rendement eux-mêmes — instrument absent "
        "de cette annexe et implémenté dans metrics.paired_block_bootstrap, "
        "résultats dans data/gold/paired_comparison_results.json"
    ),
    (
        "et ce sont elles qui permettent d'affirmer « statistiquement "
        "indiscernable » avec des preuves, au lieu de publier une estimation "
        "ponctuelle en espérant qu'elle tienne",
        "et ce sont elles qui permettent de dire ce qui est établi et ce qui ne "
        "l'est pas, au lieu de publier une estimation ponctuelle en espérant "
        "qu'elle tienne. La formulation licite est « aucune surperformance "
        "n'est établie », jamais « les stratégies sont équivalentes » : une "
        "équivalence demanderait un test dédié contre une marge fixée à "
        "l'avance, qui n'a pas été mené"
    ),
)


def set_paragraph(paragraph, text: str) -> None:
    """Rewrite a paragraph, keeping run[0] as the style carrier."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    keeper = runs[0]
    for run in runs[1:]:
        run._element.getparent().remove(run._element)
    keeper.text = text


def main() -> None:
    document = Document(str(DOCX))
    changed = 0
    for paragraph in document.paragraphs:
        text = paragraph.text
        updated = text
        for old, new in REPLACEMENTS:
            if old in updated:
                updated = updated.replace(old, new)
        if updated != text:
            set_paragraph(paragraph, updated)
            changed += 1
    document.save(str(DOCX))
    print(f"{DOCX.relative_to(ROOT)}: {changed} paragraph(s) rewritten")


if __name__ == "__main__":
    main()
