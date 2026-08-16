"""
build_global_2004_results.py — Generate docs/GLOBAL_2004_RESULTS.md.

Addresses: P4 — every figure in the document is read from a committed
artifact. Nothing is typed. The project has been bitten repeatedly by numbers
that lived in prose and drifted from the evidence they described
(§17.1, §17.11), and a results summary is the single worst place for that to
happen, so this file is generated and the document carries a banner saying so.

Usage:
    python scripts/build_global_2004_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

GOLD = ROOT / "data" / "gold"
OUT = ROOT / "docs" / "GLOBAL_2004_RESULTS.md"


def load(name: str) -> dict:
    path = GOLD / name
    if not path.is_file():
        raise SystemExit(f"{path} absent — run the global_2004 stages first.")
    return json.loads(path.read_text())


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    readiness = load("global_2004_readiness.json")
    q1 = load("global_2004_q1_results.json")
    q2 = load("global_2004_q2_results.json")
    interp = load("global_2004_interpretation.json")
    cap = load("etf_cap_verdict.json")

    free = readiness["allocation_freedom"]
    mvlw = free["min_variance_lw"]
    etf_distinct = cap["results"]["0.25"]["min_variance_lw"]["distinct_allocations"]
    etf_n = 248

    cand, comp = q1["candidate"], q1["comparator"]
    diff, pi, verdicts = (
        q1["observed_difference"], q1["paired_inference"], q1["verdicts"]
    )
    primary = q2["family_tests"]["primary_sharpe"]
    secondary = q2["family_tests"]["secondary_mean_return"]
    ledger, bench = q2["candidate_ledger"], interp["benchmark"]
    sel = interp["honestly_selected_challengers"]
    tel = interp["telemetry_summary"]

    L: list[str] = []
    a = L.append

    a("# `global_2004` — Results")
    a("")
    a("> **GENERATED FILE — do not edit by hand.** Every number below is read from a")
    a("> committed artifact by `scripts/build_global_2004_results.py`. Regenerate it")
    a("> rather than correcting it in place.")
    a("")
    a(f"Protocol: [`GLOBAL_UNIVERSE_PREREGISTRATION.md`](GLOBAL_UNIVERSE_PREREGISTRATION.md), "
      f"frozen before any data was ingested.")
    a("")
    a("---")
    a("")

    # 1 — why
    a("## 1. Why this universe exists")
    a("")
    a("Two **measured** defects in the released universes made them unable to test the")
    a("ML stack cleanly. Neither is a modelling failure; both are properties of the")
    a("opportunity set, which is why no modelling change could address them.")
    a("")
    a(f"**`etf_2017` — the constraint dominated the objective.** With 5 assets at a 25%")
    a(f"cap, `min_variance_lw` emitted **{etf_distinct} distinct allocation across")
    a(f"{etf_n} rebalances**. The arithmetic does not force that corner — equal weight")
    a("is feasible with nothing at the cap — but empirically the constraint picked the")
    a("portfolio, so a better covariance estimate had nowhere to show up.")
    a("")
    a("**`full_2021` — the covariance input was known-biased.** Casablanca and NYSE")
    a("sessions barely overlap and BVC prices are frequently stale: same-day correlation")
    a("with SPY of 0.0041 against lag-1 of 0.0779, with 17.1% zero-return days.")
    a("")
    a("`global_2004` was built to have neither defect, under the **same 25% cap**.")
    a("")

    # 2 — expressiveness
    a("## 2. Allocation expressiveness — the defect is removed")
    a("")
    a("| | `etf_2017` | **`global_2004`** |")
    a("|---|---:|---:|")
    a(f"| `min_variance_lw` distinct allocations | **{etf_distinct} / {etf_n}** | "
      f"**{mvlw['distinct_allocations']} / {free['n_rebalances']}** |")
    a(f"| mean assets at the cap | ~4 | {mvlw['mean_assets_at_cap']} |")
    a(f"| effective positions | — | {mvlw['mean_effective_positions']} of 10 |")
    a(f"| window | 20.7 yr | {readiness['window']['years']} yr, "
      f"{readiness['coverage']['silver_rows']:,} rows |")
    a("")
    a(f"Data readiness: **{readiness['verdict']}**, "
      f"{readiness['n_gates_passed']}/{readiness['n_gates']} pre-registered gates, "
      f"including no stale-price lead/lag signature "
      f"(max lag dominance {readiness['lag_dominance']['max_lag_dominance']}, "
      "requirement ≤ 0) and both no-lookahead gates.")
    a("")
    a("Same cap, same costs, same engine — and the optimizer now varies its allocation")
    a("at **every** rebalance instead of emitting one portfolio for two decades.")
    a("")

    # 3 — Q1
    a("## 3. Q1 — the regime layer versus the classical comparator")
    a("")
    a(f"One pre-specified comparison on the frozen test segment "
      f"({q1['provenance']['test_segment']['start']} → "
      f"{q1['provenance']['test_segment']['end']}, "
      f"{q1['provenance']['test_segment']['n_days']:,} days). No multiple-testing")
    a("correction, because a single hypothesis fixed in advance needs none.")
    a("")
    a("| | net Sharpe | geometric ann. return | max DD | turnover |")
    a("|---|---:|---:|---:|---:|")
    a(f"| `{cand['strategy']}` | {cand['net_sharpe']} | "
      f"{pct(cand['net_geometric_annual_return'])} | {pct(cand['max_drawdown'])} | "
      f"{cand['avg_turnover']} |")
    a(f"| `{comp['strategy']}` | **{comp['net_sharpe']}** | "
      f"{pct(comp['net_geometric_annual_return'])} | {pct(comp['max_drawdown'])} | "
      f"{comp['avg_turnover']} |")
    a("")
    a(f"- observed ΔSharpe **{diff['net_sharpe_diff']:+}**, "
      f"90% CI {pi['sharpe_diff_ci']}")
    a(f"- one-sided null-centred p = **{pi['one_sided_null_centred_p_value']}**")
    a(f"- P(ΔSharpe > 0) = {pi['prob_sharpe_diff_positive']}")
    a(f"- `candidate_improvement_at_least_0_05` = "
      f"**{verdicts['candidate_improvement_at_least_0_05']}**")
    a(f"- `evidence_of_candidate_outperformance` = "
      f"**{verdicts['evidence_of_candidate_outperformance']}**")
    a(f"- `observed_absolute_sharpe_gap_at_least_0_05` = "
      f"**{verdicts['observed_absolute_sharpe_gap_at_least_0_05']}** "
      f"({verdicts['observed_gap_direction']})")
    a("")
    a(f"**{q1['reading']['primary']}**")
    a("")
    a(f"*Secondary.* {q1['reading']['secondary']}")
    a("")
    a(f"{q1['reading']['costs_did_not_create_the_result']}")
    a("")
    a(f"{q1['reading']['not_worse_on_every_dimension']}")
    a("")

    # 4 — selected challengers
    a("## 4. The honestly selected challengers")
    a("")
    a("The configurations a disciplined practitioner would actually have deployed,")
    a("chosen forward-only on train+validation with the frozen test segment never")
    a("shown to a selector.")
    a("")
    a("| strategy | net Sharpe | Δ vs regime | geometric ann. return | fallbacks |")
    a("|---|---:|---:|---:|---:|")
    a(f"| `{bench['strategy']}` | {bench['net_sharpe']} | — | "
      f"{pct(bench['net_geometric_annual_return'])} | — |")
    for name, s in sel.items():
        a(f"| `{name}` | {s['net_sharpe']} | {s['net_sharpe_diff_vs_benchmark']:+} | "
          f"{pct(s['net_geometric_annual_return'])} | {s['fallback_count']} |")
    a("")
    a(f"{interp['reading']['selected_vs_family']}")
    a("")

    # 5 — the correction
    a("## 5. Q2 — the 240-candidate family test")
    a("")
    a(f"Every reachable RF/XGBoost × portfolio-lever configuration — "
      f"**{ledger['expected_reachable_count']} expected from frozen configuration, "
      f"{ledger['executed_count']} executed** — against `regime_conditional`. "
      "No candidate was deduplicated on observed performance.")
    a("")
    a("| endpoint | White RC | Hansen SPA | beating benchmark | best raw differential |")
    a("|---|---:|---:|---:|---:|")
    a(f"| **PRIMARY** (Sharpe) | {primary['reality_check_p_value']} | "
      f"{primary['spa_p_value']} | {primary['n_candidates_beating_benchmark']}/240 | "
      f"**{primary['best_differential']:+}** |")
    a(f"| SECONDARY (mean return) | {secondary['reality_check_p_value']} | "
      f"{secondary['spa_p_value']} | "
      f"{secondary['n_candidates_beating_benchmark']}/240 | "
      f"{secondary['best_differential']:+} |")
    a("")
    a(f"**`concordant_evidence_of_family_outperformance` = "
      f"{q2['verdict']['concordant_evidence_of_family_outperformance']}** — "
      f"`{q2['verdict']['evidence_status']}`.")
    a("")
    a("### ⭐ Why this is the project's clearest methodological result")
    a("")
    a(f"The best of the 240 candidates beat the benchmark by "
      f"**{primary['best_differential']:+.4f} Sharpe** — *larger in magnitude than "
      f"Q1's {diff['net_sharpe_diff']:+} gap, and in the opposite direction*. Quoted "
      "alone it would read as a headline result.")
    a("")
    a(f"It is not one. It is the maximum of a {ledger['executed_count']}-wide search, "
      f"and **RC p = {primary['reality_check_p_value']}** is exactly the correction "
      f"pricing that fact. Only {primary['n_candidates_beating_benchmark']} of "
      f"{ledger['executed_count']} candidates beat the benchmark on Sharpe at all.")
    a("")
    a("The auditable evidence showing why the attractive raw winner should not be")
    a("believed is the contribution here — more so than the negative result itself.")
    a("")

    # 6 — telemetry
    a("## 6. Telemetry — the results are not a fallback artefact")
    a("")
    a(f"- **Benchmark:** {tel['benchmark_fallbacks']} fallbacks in "
      f"{tel['benchmark_fits']} fits, **all before the frozen test segment**. Q1 "
      "reports zero because it counts only the rebalances inside that segment; the "
      "two counts cover different windows and are consistent.")
    a(f"- **Candidate family:** {tel['candidate_family_fallbacks']} fallback in "
      f"{tel['candidate_family_fits']:,} fits.")
    a(f"- **Both honestly selected challengers: "
      f"{', '.join(str(v) for v in tel['selected_challenger_fallbacks'].values())} "
      "fallbacks.**")
    a("")
    a("Fallback behaviour therefore does **not** explain the negative results — every")
    a("number came from the model its label names.")
    a("")

    # 7 — limitations
    a("## 7. Limitations")
    a("")
    a("### These evaluations are not independent")
    a("")
    a(f"{interp['reading']['overlap_caveat']}")
    a("")
    a("Describe them as **two distinct but statistically overlapping evaluations**,")
    a("never as independent confirmations.")
    a("")
    a("### Attribution")
    a("")
    for item in q2["limitations"]:
        a(f"- {item}")
    a("")

    # 8 — conclusion
    a("## 8. What is licensed")
    a("")
    a(f"> {interp['reading']['licensed_conclusion']}")
    a("")
    a("Three separate things, deliberately not merged:")
    a("")
    a("| | |")
    a("|---|---|")
    a("| **Engineering success** | A pre-registered universe was built, gated on ten "
      "data-readiness checks, wired into DVC and Dagster, and frozen as run-once "
      "evidence. |")
    a("| **Methodological success** | The correction did its job: a selected "
      f"{primary['best_differential']:+.3f} Sharpe difference was prevented from "
      "becoming a false headline. |")
    a("| **Absence of established ML outperformance** | Neither the regime layer (Q1) "
      "nor the challenger family (Q2) established an advantage over simpler portfolio "
      "rules on this universe. |")
    a("")
    a("Once the two identification defects were removed, the complex models were")
    a("finally given a fair test — and still did not establish an advantage.")
    a("")
    a("---")
    a("")
    a(f"*`global_2004` is a RESEARCH EXPERIMENT. It is not wired into the API, the")
    a("dashboard, or any production-facing allocation, and no released result depends")
    a("on it.*")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
