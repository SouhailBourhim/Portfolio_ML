"""Generate the model-integrity section from `fit_report_summary.json`.

Addresses: P4 — states, in one place and from the artifact, whether any
published result was produced by something other than the model its label
names.

TWO RENDERING MODES, chosen by the data rather than by an author:

* **All-clear** (no fallback anywhere): a compact statement plus a small
  `0 / N` audit table. Rendering fallback-period and excluding-fallback
  performance tables here would be four columns of the same number and would
  imply a problem that the measurement says is absent.
* **Any fallback present**: the full tables appear automatically — per-strategy
  rates in rebalances AND days, the reasons, and the three-way performance
  split. Nobody has to remember to switch them on.

WORDING RULES, both deliberate:

1. "On the released snapshot", never "the model never falls back". The
   measurement covers one dataset at one revision. The fallback paths are live
   code and a different snapshot can exercise them.
2. The 2026-07-20 DCC fallback is preserved as historical context WITH the
   note that it no longer reproduces. Deleting a superseded observation is how
   a project quietly loses the ability to explain its own history; the dividend
   correction and the deep-history window changed the data underneath it.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
SUMMARY_PATH = GOLD / "fit_report_summary.json"
OUT_MD = ROOT / "docs" / "MODEL_INTEGRITY.md"
OUT_TEX = ROOT / "docs" / "rapport" / "assets" / "tables" / "integrite_modele.tex"
OUT_COSTS = ROOT / "docs" / "rapport" / "assets" / "tables" / "sensibilite_couts.tex"

# Kept because it happened, flagged because it no longer does.
HISTORICAL_NOTE = (
    "**Historical context — 2026-07-20.** An earlier run recorded the DCC-GARCH "
    "Ledoit-Wolf fallback firing exactly once, on `IAM.CS` in `full_2021`, and that "
    "observation is cited in the project record as evidence the safety net worked. "
    "It does **not** reproduce on the current snapshot: the BVC dividend correction "
    "and the adoption of the deep ETF history both changed the return series the "
    "GARCH fits see. The original observation was true when made; it is retained "
    "here rather than deleted, because a superseded measurement is part of how a "
    "result came to be trusted."
)


def _load() -> dict:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"{SUMMARY_PATH.relative_to(ROOT)} is missing. Run `dvc repro fit_reports`."
        )
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _distinct_rebalance_dates(results: list[dict]) -> int:
    """Rebalance DATES, not fits: four strategies share one universe's calendar."""
    per_universe = {r["universe"]: r["rebalances"] for r in results}
    return sum(per_universe.values())


def _audit_table_md(results: list[dict]) -> str:
    rows = ["| Universe | Strategy | Fallback rebalances | Fallback days | Effective model |",
            "|---|---|---:|---:|---|"]
    for entry in sorted(results, key=lambda r: (r["universe"], r["strategy_requested"])):
        effective = ", ".join(f"`{k}`" for k in sorted(entry["models_effective"]))
        rows.append(
            f"| `{entry['universe']}` | `{entry['strategy_requested']}` "
            f"| {entry['fallback_rebalances']} / {entry['rebalances']} "
            f"| {entry['fallback_days']} / {entry['oos_days']} | {effective} |"
        )
    return "\n".join(rows)


def _degraded_tables_md(results: list[dict]) -> str:
    """Only reached when something actually fell back."""
    blocks = ["\n## Degraded-period performance\n",
              "A fallback occurred, so the split below is meaningful and is shown.\n",
              "| Universe | Strategy | Full period (hybrid) | Fallback periods | Excluding fallback |",
              "|---|---|---:|---:|---:|"]
    for entry in sorted(results, key=lambda r: (r["universe"], r["strategy_requested"])):
        if entry["fallback_rebalances"] == 0:
            continue
        perf = entry["performance"]

        def cell(block: dict) -> str:
            if block["status"] != "estimated":
                return f"*not estimable — {block['reason']}*"
            return f"{block['net_sharpe']} ({block['n_days']}d)"

        blocks.append(
            f"| `{entry['universe']}` | `{entry['strategy_requested']}` "
            f"| {cell(perf['full_period_hybrid'])} | {cell(perf['fallback_periods'])} "
            f"| {cell(perf['excluding_fallback'])} |"
        )
    blocks.append(
        "\n**Full-period figures are the HYBRID** of the requested model and its "
        "fallback, never the requested model alone. An excluding-fallback figure is "
        "withheld rather than approximated when there are too few active days."
    )
    reasons = ["\n### Reasons\n"]
    for entry in sorted(results, key=lambda r: (r["universe"], r["strategy_requested"])):
        for reason, count in entry.get("fallback_reasons", {}).items():
            reasons.append(
                f"- `{entry['universe']}` / `{entry['strategy_requested']}` "
                f"— {count}x: {reason}"
            )
    return "\n".join(blocks + reasons)


def build_markdown(summary: dict) -> str:
    results = summary["results"]
    total_fits = sum(r["rebalances"] for r in results)
    total_fallbacks = sum(r["fallback_rebalances"] for r in results)
    dates = _distinct_rebalance_dates(results)
    clean = total_fallbacks == 0

    strategies = len({r["strategy_requested"] for r in results})
    headline = (
        f"On the released snapshot, **{total_fallbacks} of {total_fits:,} fits** across "
        f"**{strategies} evaluated strategies** and **{dates:,} rebalance dates** used a "
        f"fallback. Every result reported here was produced by the model named in its "
        f"label."
        if clean else
        f"On the released data snapshot, **{total_fallbacks} of {total_fits:,} strategy "
        f"fits** used a fallback. Results for the affected strategies are HYBRIDS of "
        f"the requested model and its substitute — see the tables below."
    )

    parts = [
        "<!-- GENERATED by scripts/build_integrity_section.py — do not edit by hand.",
        "     Source: data/gold/fit_report_summary.json -->",
        "",
        "# Model integrity — did the label match the model?",
        "",
        "Several estimators in this project degrade rather than crash: DCC-GARCH falls "
        "back to Ledoit-Wolf shrinkage on non-convergence, the ML signals fall back to "
        "the naive sample mean on a thin panel or a failed fit, and the regime strategy "
        "resolves an uncertain posterior to its defensive branch. Each is deliberate — "
        "a walk-forward loop must not die on one bad window — but each also means a "
        "result can carry a label that is not the whole truth.",
        "",
        headline,
        "",
        "## Audit",
        "",
        _audit_table_md(results),
        "",
        "Source artifacts, both versioned and hashed into the snapshot manifest:",
        "`data/gold/fit_reports.parquet` (one row per rebalance) and "
        "`data/gold/fit_report_summary.json` (aggregates). The table above is generated "
        "from the second; neither is typed by hand.",
        "",
        "## What this does and does not say",
        "",
        "- **Does:** on this snapshot, at this revision, across the strategies and "
        "dates counted above, no fallback path was taken. The value is a reproducible, "
        "versioned measurement rather than an assumption.",
        "- **Does NOT:** claim that no model ever falls back. The fallback paths are "
        "live, tested code and a different snapshot can exercise them. The scope of the "
        "claim is exactly the fits counted above — the telemetry exists to make the "
        "wider claim TESTABLE, not to assert it.",
        "- **The control that makes this credible** is not the zero itself but "
        "`test_full_period_sharpe_matches_the_published_dashboard_figure`: the runner "
        "must reproduce every published Sharpe through the instrumented engine, so the "
        "telemetry cannot be auditing a differently-configured lookalike under the same "
        "name.",
        "- Degraded-period and excluding-fallback performance tables are omitted here "
        "**because there is nothing to show** — with zero fallback days they would "
        "repeat the full-period column. They render automatically if a future run "
        "records any fallback."
        if clean else
        "- Degraded-period tables are shown below because a fallback occurred.",
        "",
        HISTORICAL_NOTE,
        "",
    ]
    if not clean:
        parts.append(_degraded_tables_md(results))
    return "\n".join(parts) + "\n"


def build_latex(summary: dict) -> str:
    """Compact French table for the report."""
    results = summary["results"]
    total_fits = sum(r["rebalances"] for r in results)
    total_fallbacks = sum(r["fallback_rebalances"] for r in results)

    rows = []
    for entry in sorted(results, key=lambda r: (r["universe"], r["strategy_requested"])):
        universe = entry["universe"].replace("_", r"\_")
        strategy = entry["strategy_requested"].replace("_", r"\_")
        rows.append(
            f"    \\texttt{{{universe}}} & \\texttt{{{strategy}}} & "
            f"{entry['fallback_rebalances']} / {entry['rebalances']} & "
            f"{entry['fallback_days']} / {entry['oos_days']} \\\\"
        )

    strategies = len({r["strategy_requested"] for r in results})
    # Four strategies share each universe's rebalance calendar. Counting every
    # fit here would call 1,188 fits "rebalance dates"; the audit statement
    # must distinguish the two units exactly as its Markdown counterpart does.
    dates = _distinct_rebalance_dates(results)
    verdict = (
        f"Sur l'instantané publié, {total_fallbacks} ajustement sur {total_fits} "
        f"— {strategies} stratégies évaluées, {dates} dates de rééquilibrage — "
        f"n'a utilisé de repli : chaque résultat rapporté ici a été produit par le "
        f"modèle que son étiquette désigne. La portée de cette affirmation est "
        f"exactement l'ensemble compté ci-dessus."
        if total_fallbacks == 0 else
        f"Sur l'instantané publié, {total_fallbacks} ajustements sur {total_fits} "
        f"ont utilisé un repli ; les résultats concernés sont des HYBRIDES."
    )
    return "\n".join([
        "% GÉNÉRÉ par scripts/build_integrity_section.py — ne pas éditer à la main.",
        "% Source : data/gold/fit_report_summary.json",
        f"\\newcommand{{\\integriteVerdict}}{{{verdict}}}",
        f"\\newcommand{{\\integriteAjustements}}{{{total_fits}}}",
        f"\\newcommand{{\\integriteReplis}}{{{total_fallbacks}}}",
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\caption[Intégrité des modèles]{Repli des estimateurs sur l'instantané "
        "publié. Un repli signifie que le résultat a été produit par un estimateur "
        "de substitution, non par le modèle nommé.}",
        "  \\label{tab:integrite}",
        "  \\begin{tabular}{@{}llrr@{}}",
        "    \\toprule",
        "    Univers & Stratégie & Replis / rééquilibrages & Replis / jours \\\\",
        "    \\midrule",
        *rows,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])


def build_cost_latex(summary: dict) -> str:
    """Cost re-pricing table. Deliberately labelled as a re-pricing.

    The caption says what the numbers are — the same allocation path priced
    under different linear cost assumptions — so a reader cannot mistake it for
    a re-optimization study in which the strategy would have traded differently.
    """
    rows = []
    for entry in sorted(summary["results"], key=lambda r: (r["universe"], r["strategy_requested"])):
        scenarios = entry["cost_sensitivity"]["scenarios"]
        break_even = entry["cost_sensitivity"]["break_even_cost_multiplier"]
        universe = entry["universe"].replace("_", r"\_")
        strategy = entry["strategy_requested"].replace("_", r"\_")

        def fr(value: float) -> str:
            return f"{value:.3f}".replace(".", ",").replace("-", "$-$")

        rows.append(
            f"    \\texttt{{{universe}}} & \\texttt{{{strategy}}} & "
            + " & ".join(fr(scenarios[k]["net_sharpe"]) for k in ("0.5x", "1x", "1.5x", "2x"))
            + " & "
            + (f"{break_even:.1f}$\\times$".replace(".", ",") if break_even else "$>20\\times$")
            + " & "
            + f"{entry['execution_profile']['avg_turnover']:.3f}".replace(".", ",")
            + " \\\\"
        )

    return "\n".join([
        "% GÉNÉRÉ par scripts/build_integrity_section.py — ne pas éditer à la main.",
        "% Source : data/gold/fit_report_summary.json",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\caption[Sensibilité aux coûts]{Sharpe net du \\emph{même} chemin "
        "d'allocation, repricé sous des coûts de transaction multipliés. Il ne "
        "s'agit pas d'une ré-optimisation : les pondérations sont identiques dans "
        "chaque colonne. « Seuil » est le multiplicateur annulant le rendement net.}",
        "  \\label{tab:couts}",
        "  \\begin{tabular}{@{}llrrrrrr@{}}",
        "    \\toprule",
        "    Univers & Stratégie & 0,5$\\times$ & 1$\\times$ & 1,5$\\times$ & "
        "2$\\times$ & Seuil & Rotation \\\\",
        "    \\midrule",
        *rows,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])


def main() -> None:
    summary = _load()
    OUT_MD.write_text(build_markdown(summary), encoding="utf-8")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(build_latex(summary), encoding="utf-8")
    OUT_COSTS.write_text(build_cost_latex(summary), encoding="utf-8")
    total = sum(r["rebalances"] for r in summary["results"])
    fallbacks = sum(r["fallback_rebalances"] for r in summary["results"])
    print(f"  docs/MODEL_INTEGRITY.md")
    print(f"  docs/rapport/assets/tables/integrite_modele.tex")
    print(f"  docs/rapport/assets/tables/sensibilite_couts.tex")
    print(f"  ({fallbacks}/{total} fits used a fallback; "
          f"degraded tables {'omitted' if fallbacks == 0 else 'RENDERED'})")


if __name__ == "__main__":
    main()
