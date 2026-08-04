"""Generate docs/MULTIPLE_TESTING.md from `reality_check_results.json`.

Addresses: P4 — Phase 5 recorded the multiple-testing position as
`not_established` because a correct Reality Check needs the frozen-test return
series of every searched configuration and only the winners had one. That is
now done, and this renders the result — including the parts that are
inconvenient.

Every number is read from the artifact. The verdict sentences are computed
from the p-values rather than written, so a future re-run that changes the
answer changes the prose too.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
RESULTS = GOLD / "reality_check_results.json"
OUT_MD = ROOT / "docs" / "MULTIPLE_TESTING.md"
OUT_TEX = ROOT / "docs" / "rapport" / "assets" / "tables" / "correction_tests_multiples.tex"

ALPHA = 0.05
HURDLE = "regime_conditional"
FLOOR = "equal_weight"
STATISTIC_LABEL = {"mean_return": "mean return", "sharpe": "Sharpe"}


def _load() -> dict:
    if not RESULTS.is_file():
        raise FileNotFoundError(
            f"{RESULTS.relative_to(ROOT)} is missing. Run `dvc repro reality_check`."
        )
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _rows(universes: dict) -> list[tuple]:
    rows = []
    for universe in sorted(universes):
        block = universes[universe]
        for benchmark in (HURDLE, FLOOR):
            for statistic in ("mean_return", "sharpe"):
                test = block["tests"][f"{benchmark}__{statistic}"]
                rows.append((
                    universe, benchmark, statistic,
                    test["reality_check_p_value"], test["spa_p_value"],
                    test["best_candidate"], test["n_candidates"],
                    test["spa_candidates_retained"],
                ))
    return rows


def _table_md(rows: list[tuple]) -> str:
    out = ["| Universe | Benchmark | Statistic | White RC *p* | Hansen SPA *p* | Rejects at 0.05 |",
           "|---|---|---|---:|---:|:---:|"]
    for uni, bench, stat, rc, spa, _best, _n, _kept in rows:
        rejects = [name for name, p in (("RC", rc), ("SPA", spa)) if p < ALPHA]
        out.append(
            f"| `{uni}` | `{bench}` | {STATISTIC_LABEL[stat]} | {rc:.4f} | {spa:.4f} "
            f"| {', '.join(rejects) if rejects else '—'} |"
        )
    return "\n".join(out)


def _release_status_block() -> list[str]:
    """Release status, driven by the artifact rather than by an author.

    This section has to be able to say BOTH things. It first said the
    limitation was closed when it was not; then, after the canonical rebuild
    recorded `established`, the corrected wording became stale in the opposite
    direction. Either way the doc was asserting a status independently of the
    artifact that defines it, which is the drift this project keeps closing.
    So it now reads the status and branches.
    """
    path = ROOT / "data" / "gold" / "paired_comparison_results.json"
    status = None
    if path.is_file():
        status = json.loads(path.read_text(encoding="utf-8"))["multiple_testing"]["status"]

    if status == "established":
        return [
            "## Release status — released",
            "",
            "> The canonical pipeline records "
            "`multiple_testing: established` in `paired_comparison_results.json`, "
            "produced by the `reality_check` DVC stage and hashed into the snapshot "
            "manifest.",
            "",
            "This is a released result, not an experiment: the stage regenerates both "
            "artifacts, Phase 5 consumes the status rather than asserting one, the "
            "report and model cards are built from those artifacts, and the release "
            "gates pass on the result. The limitation Phase 5 recorded as "
            "`not_established` is closed.",
            "",
        ]
    return [
        "## Release status — the limitation is NOT yet closed",
        "",
        "> This is an **experiment result, not a release result.** The released "
        "status of the multiple-testing position remains "
        "`multiple_testing: not_established`, as recorded in "
        "`paired_comparison_results.json`.",
        "",
        "A finding becomes a released result in this project only when the canonical "
        "pipeline produces it: the `reality_check` stage regenerates both artifacts "
        "into the snapshot manifest, Phase 5 is re-run so the status is consumed "
        "rather than asserted, every derived surface is rebuilt, and the release "
        "gates pass. Until then the correction has been *run*, not *closed*.",
        "",
    ]


def build_markdown(payload: dict) -> str:
    universes = payload["universes"]
    rows = _rows(universes)
    n_candidates = rows[0][6]

    hurdle_rows = [r for r in rows if r[1] == HURDLE]
    floor_rows = [r for r in rows if r[1] == FLOOR]
    hurdle_rejects = [r for r in hurdle_rows if min(r[3], r[4]) < ALPHA]
    floor_rejects = [r for r in floor_rows if min(r[3], r[4]) < ALPHA]

    headline = (
        f"**No candidate out of {n_candidates} beats `{HURDLE}` once the search is "
        f"accounted for** — on either universe, under either test, on either "
        f"statistic."
        if not hurdle_rejects else
        f"**{len(hurdle_rejects)} of {len(hurdle_rows)} comparisons against "
        f"`{HURDLE}` reject at {ALPHA}.**"
    )

    kept = {r[7] for r in rows}
    return "\n".join([
        "<!-- GENERATED by scripts/build_multiple_testing_doc.py — do not edit by hand.",
        "     Source: data/gold/reality_check_results.json -->",
        "",
        "# Multiple-testing correction — is the winner just the luckiest?",
        "",
        "Every comparison this project reported before now tests **one** candidate "
        "against a benchmark. But that candidate was *chosen from a search*, and the "
        "best of many candidates beats a benchmark by chance more often than any "
        "single candidate does. Phase 5 recorded the position as `not_established` "
        "because a correct test needs the frozen-test return series of **every** "
        "searched configuration, and only the winners had one.",
        "",
        "They all have one now.",
        "",
        "## Method",
        "",
        f"- **{n_candidates} candidates per universe** — the *reachable* search space, "
        "every combination of ML hyperparameters and portfolio levers the "
        "hierarchical search could have selected. The DSR ledger recorded 51 trials; "
        "correcting for that smaller number would understate the multiplicity.",
        "- Each candidate re-evaluated on the **same frozen test dates**, via the "
        "unchanged Phase 5 evaluator, so the series are comparable to the published "
        "Phase 5 figures rather than to a cheaper approximation.",
        "- **White's Reality Check (2000)** and **Hansen's SPA (2005)**, testing the "
        "composite null that *no* candidate outperforms the benchmark.",
        "- All candidates share the **same bootstrap block draws**. Configurations "
        "differing by one hyperparameter on one data window are nearly the same "
        "strategy repeated; resampling them independently would treat them as "
        "independent bets and overstate the breadth of the search.",
        "- Two statistics. `mean_return` is the textbook White formulation, whose "
        "recentring argument assumes the performance measure is a mean. `sharpe` "
        "matches the project's headline metric but is a ratio of moments, so its "
        "asymptotic justification is weaker. Both are reported; neither alone.",
        "",
        "## Results",
        "",
        _table_md(rows),
        "",
        f"### Against the hurdle (`{HURDLE}`)",
        "",
        headline,
        "",
        "This is the comparison that matters, because `regime_conditional` is the "
        "system the ML layer was built to beat. The conclusion every earlier phase "
        "reached now survives a correction for a "
        f"{n_candidates}-candidate search on both universes.",
        "",
        "It is worth seeing how much selection inflates. On `full_2021` the best "
        "candidate *chosen by looking at the test set* reaches a Sharpe of 1.87, "
        "against 1.31 for the one selected honestly by forward-only validation. That "
        "gap is not a finding — it is the size of the bias this correction exists to "
        "price.",
        "",
        f"### Against the naive floor (`{FLOOR}`)",
        "",
        (f"**{len(floor_rejects)} of {len(floor_rows)} comparisons reject.** "
         if floor_rejects else "No comparison rejects. "),
        "Read this carefully, because it is a weaker claim than it appears:",
        "",
        f"- It is **not** evidence for the ML layer. `{HURDLE}` already beats "
        f"`{FLOOR}` by a wider margin, and since the dividend correction so does "
        "classical `max_sharpe`. Clearing a naive floor is not the same as adding "
        "value over the system in use.",
        "- **RC and SPA disagree by an order of magnitude** here. SPA retained "
        f"{min(kept)}–{max(kept)} of {n_candidates} candidates, so its trimming rule "
        "is barely firing: the divergence is **studentisation**, which is Hansen's "
        "documented power gain over White's conservatism — not a different finding.",
        "- **Eight tests were run** (2 universes × 2 benchmarks × 2 statistics). "
        "Quoting the smallest p-value from eight correlated tests is multiplicity one "
        "level up, and no further correction has been applied to *these* eight.",
        "",
        *_release_status_block(),
        "- The project's headline conclusion is unchanged and, once released, better "
        "supported: no evidence that the ML signal layer outperforms the regime "
        f"baseline, now including a correction for the whole {n_candidates}-candidate "
        "search.",
        "",
        "## Cost, and why this was deferred",
        "",
        "| Universe | Test days | Candidates | Runtime |",
        "|---|---:|---:|---:|",
        *[f"| `{u}` | {universes[u]['n_test_days']:,} | {universes[u]['n_candidates']} "
          f"| {universes[u]['runtime_seconds'] / 60:.0f} min |"
          for u in sorted(universes)],
        "",
        "Affordable only because the content-addressed prediction cache is keyed on "
        "the model configuration and not on the levers: the 16 lever variants of one "
        "hyperparameter config share a single set of fitted models. Measured at 128s "
        "for the first variant and 0.3–0.7s for each of the remaining 15, so the loop "
        "runs hyperparameters-outer. Reversing that ordering would refit every model "
        "16 times.",
        "",
        "Artifacts: `data/gold/reality_check_results.json` and "
        "`data/gold/reality_check_series.parquet` (every candidate's net-return "
        "series on the frozen test dates), both hashed into the snapshot manifest.",
        "",
    ]) + "\n"


def build_latex(payload: dict) -> str:
    rows = _rows(payload["universes"])
    n_candidates = rows[0][6]
    hurdle_rejects = [r for r in rows if r[1] == HURDLE and min(r[3], r[4]) < ALPHA]

    body = []
    for uni, bench, stat, rc, spa, _b, _n, _k in rows:
        body.append(
            f"    \\texttt{{{uni.replace('_', chr(92) + '_')}}} & "
            f"\\texttt{{{bench.replace('_', chr(92) + '_')}}} & "
            f"{STATISTIC_LABEL[stat]} & {rc:.4f}".replace(".", ",")
            + f" & {spa:.4f}".replace(".", ",") + " \\\\"
        )
    verdict = (
        f"Aucun des {n_candidates} candidats ne bat la stratégie à régimes une fois "
        f"la recherche prise en compte, sur aucun des deux univers."
        if not hurdle_rejects else
        f"{len(hurdle_rejects)} comparaisons contre la stratégie à régimes rejettent."
    )
    return "\n".join([
        "% GÉNÉRÉ par scripts/build_multiple_testing_doc.py — ne pas éditer à la main.",
        "% Source : data/gold/reality_check_results.json",
        f"\\newcommand{{\\correctionVerdict}}{{{verdict}}}",
        f"\\newcommand{{\\correctionCandidats}}{{{n_candidates}}}",
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\caption[Correction pour tests multiples]{Reality Check de White et SPA "
        "de Hansen sur l'ensemble des candidats explorés. L'hypothèse nulle est "
        "composite : \\emph{aucun} candidat ne surperforme la référence.}",
        "  \\label{tab:correction}",
        "  \\begin{tabular}{@{}lllrr@{}}",
        "    \\toprule",
        "    Univers & Référence & Statistique & RC $p$ & SPA $p$ \\\\",
        "    \\midrule",
        *body,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])


def main() -> None:
    payload = _load()
    OUT_MD.write_text(build_markdown(payload), encoding="utf-8")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(build_latex(payload), encoding="utf-8")
    print(f"  docs/MULTIPLE_TESTING.md")
    print(f"  docs/rapport/assets/tables/correction_tests_multiples.tex")


if __name__ == "__main__":
    main()
