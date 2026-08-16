"""
run_global_2004_interpretation.py — Additive reading of the frozen Q2 evidence.

Addresses: P4 — Q2's artifact records WHICH RF/XGBoost configurations the
forward-only selectors chose, but not how those configurations actually
performed on the frozen test segment. That gap matters: the honestly selected
challenger is the only one a practitioner could have deployed, so its result is
the one that answers "what would this have done for me", while the RC/SPA
p-values answer the different question "can the family's best be believed".

STRICTLY ADDITIVE. This script READS the frozen Q2 artifacts and writes a new
one. It re-runs nothing, modifies nothing, and cannot: `global_2004_q2` is a
frozen DVC stage, and every number below is recovered from the persisted
per-candidate return series rather than recomputed from the models. A 6.1-hour
re-run to recover numbers already on disk would be waste, and would risk
replacing frozen evidence to obtain something the evidence already contains.

WHAT IT ADDS
  * Frozen-test performance of the two honestly selected challengers.
  * Their differentials against the benchmark, on the same dates.
  * An accurate telemetry summary, because the headline "1 fallback" and the
    benchmark's "3 fallbacks" are counted over different denominators and
    different windows, and stating them side by side without that context
    would misdescribe both.

Usage:
    python src/run_global_2004_interpretation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from metrics import annualized_return, annualized_sharpe, max_drawdown  # noqa: E402
from provenance import _sha256, git_revision  # noqa: E402
from utils import load_params  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("global_2004_interpretation")

GOLD = ROOT / "data" / "gold"
Q2_RESULTS = GOLD / "global_2004_q2_results.json"
Q2_SERIES = GOLD / "global_2004_q2_series.parquet"
Q1_RESULTS = GOLD / "global_2004_q1_results.json"
OUT = GOLD / "global_2004_interpretation.json"

BENCHMARK = "regime_conditional"


def candidate_label(spec: dict) -> str:
    """Reproduce `run_reality_check._label` for a selected configuration.

    Duplicated deliberately rather than imported: this script must be able to
    read the frozen artifacts without importing the search machinery, so that
    a later change to that machinery cannot silently alter how a FROZEN
    artifact is interpreted.
    """
    ml = "_".join(f"{k}={v}" for k, v in sorted(spec["ml_params"].items()))
    return (f"{spec['model_type']}__{ml}__shrink={spec['shrinkage_weight']}"
            f"__pen={spec['turnover_penalty']}")


def describe(series: pd.Series, rf: float) -> dict:
    return {
        "net_sharpe": round(float(annualized_sharpe(series, risk_free_annual=rf)), 4),
        "net_geometric_annual_return": round(float(annualized_return(series)), 6),
        "max_drawdown": round(float(max_drawdown(series)), 6),
        "n_test_days": int(len(series)),
    }


def run() -> dict:
    params = load_params()
    rf = params["backtest"]["risk_free_annual"]

    for path in (Q2_RESULTS, Q2_SERIES, Q1_RESULTS):
        if not path.is_file():
            raise RuntimeError(f"{path} absent; the frozen evidence is incomplete.")

    q2 = json.loads(Q2_RESULTS.read_text())
    q1 = json.loads(Q1_RESULTS.read_text())
    wide = (
        pd.read_parquet(Q2_SERIES)
        .pivot(index="Date", columns="candidate", values="net_return")
    )
    if BENCHMARK not in wide.columns:
        raise RuntimeError("Benchmark series absent from the persisted Q2 series.")

    bench = wide[BENCHMARK]
    bench_desc = describe(bench, rf)

    # The benchmark recovered from the series must agree with BOTH frozen
    # artifacts. If it does not, the series file does not describe the run
    # those artifacts report, and nothing below is safe to read.
    for source, name in ((q1["candidate"], "Q1"), (q2["benchmark"], "Q2")):
        if bench_desc["net_sharpe"] != source["net_sharpe"]:
            raise RuntimeError(
                f"Benchmark recovered from the series ({bench_desc['net_sharpe']}) "
                f"disagrees with {name} ({source['net_sharpe']})."
            )
    log.info("Benchmark reconciles with Q1 and Q2: net Sharpe %.4f", bench_desc["net_sharpe"])

    telemetry = q2["fallback_telemetry"]["candidates"]
    selected = {}
    for display, spec in q2["deployable_challengers"].items():
        label = candidate_label(spec)
        if label not in wide.columns:
            raise RuntimeError(f"Selected challenger {label!r} absent from the series.")
        desc = describe(wide[label], rf)
        selected[display] = {
            **desc,
            "candidate_label": label,
            "model_type": spec["model_type"],
            "ml_params": spec["ml_params"],
            "shrinkage_weight": spec["shrinkage_weight"],
            "turnover_penalty": spec["turnover_penalty"],
            "net_sharpe_diff_vs_benchmark": round(
                desc["net_sharpe"] - bench_desc["net_sharpe"], 4
            ),
            "geometric_annual_return_diff_vs_benchmark": round(
                desc["net_geometric_annual_return"]
                - bench_desc["net_geometric_annual_return"], 6
            ),
            "fallback_count": int(telemetry[label]["fallback_count"]),
            "n_fits": int(telemetry[label]["n_fits"]),
            "selection": spec["selection"],
        }

    total_fits = sum(int(t["n_fits"]) for t in telemetry.values())
    total_fallbacks = sum(int(t["fallback_count"]) for t in telemetry.values())
    bench_tel = q2["fallback_telemetry"]["benchmark"]

    artifact = {
        "provenance": {
            "universe": "global_2004",
            "base_currency": "USD",
            "kind": "ADDITIVE INTERPRETATION — reads frozen evidence, re-runs nothing",
            "git_revision": git_revision(ROOT),
            "generated_at": pd.Timestamp.now().isoformat(),
            "source_artifacts": {
                rel: _sha256(ROOT / rel) for rel in (
                    "data/gold/global_2004_q1_results.json",
                    "data/gold/global_2004_q2_results.json",
                    "data/gold/global_2004_q2_series.parquet",
                    "data/gold/global_2004_readiness.json",
                    "docs/GLOBAL_UNIVERSE_PREREGISTRATION.md",
                )
            },
        },
        "benchmark": {"strategy": BENCHMARK, **bench_desc},
        "honestly_selected_challengers": selected,
        "what_this_adds": (
            "Q2's artifact records WHICH configurations the forward-only "
            "selectors chose but not how they performed on the frozen test "
            "segment. The selected challenger is the only one a practitioner "
            "could have deployed, so its result answers 'what would this have "
            "done', while the RC/SPA p-values answer 'can the family's best be "
            "believed'. Both are needed; neither substitutes for the other."
        ),
        "telemetry_summary": {
            "benchmark_fallbacks": int(bench_tel["fallback_count"]),
            "benchmark_fits": int(bench_tel["n_fits"]),
            "benchmark_note": (
                "3 of 249 fits fell back (regime_conditional via "
                "min_variance_lw). ALL occur BEFORE the frozen test segment: "
                "Q1 reports zero fallbacks because it counts only the 91 "
                "rebalances inside that segment. The two counts are consistent "
                "— they cover different windows."
            ),
            "candidate_family_fallbacks": total_fallbacks,
            "candidate_family_fits": total_fits,
            "selected_challenger_fallbacks": {
                k: v["fallback_count"] for k, v in selected.items()
            },
            "conclusion": (
                f"{total_fallbacks} fallback across {total_fits:,} candidate "
                "fits, and ZERO for both honestly selected challengers. "
                "Fallback behaviour therefore does NOT explain their negative "
                "test results — every number they produced came from the model "
                "its label names."
            ),
        },
        "reading": {
            "selected_vs_family": (
                "The honestly selected RF and XGBoost challengers had LOWER "
                "OBSERVED net Sharpe on the frozen test segment: -0.0657 and "
                "-0.0898 relative to regime_conditional. This is an "
                "operationally relevant DESCRIPTIVE result, not statistical "
                "evidence of underperformance — no individual paired test of "
                "these differences was pre-specified. It is a complementary "
                "operational diagnostic alongside the family verdict, not a "
                "stronger version of it: the family test asks whether the best "
                "of 240 can be believed, this describes what a disciplined "
                "practitioner would have held."
            ),
            "selected_vs_family_wording_note": (
                "An earlier version of this field said the selected challengers "
                "'UNDERPERFORM' and 'did worse', and called it a 'stronger "
                "statement'. That over-claimed: lower observed Sharpe is not "
                "established underperformance without a paired test, and none "
                "was pre-specified for these two comparisons. Corrected "
                "2026-08-16."
            ),
            "overlap_caveat": (
                "etf_2017 and global_2004 are TWO DISTINCT BUT STATISTICALLY "
                "OVERLAPPING evaluations, not independent ones. They share five "
                "instruments (SPY, QQQ, EEM, GLD, TLT) and largely overlapping "
                "evaluation periods, so they share the same market shocks. "
                "Agreement between them is weaker evidence than agreement "
                "between independent samples would be."
            ),
            "licensed_conclusion": (
                "The 25% cap made etf_2017 weak for model attribution. After "
                "restoring allocation expressiveness in global_2004, no regime "
                "or challenger advantage was established either. Cap dominance "
                "was therefore not the sole explanation for the earlier "
                "negative result."
            ),
        },
    }

    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log.info("")
    log.info("  %-22s %8s %12s %10s", "strategy", "netSharpe", "vs regime", "geo ann")
    log.info("  %-22s %8.4f %12s %9.2f%%", BENCHMARK, bench_desc["net_sharpe"], "—",
             bench_desc["net_geometric_annual_return"] * 100)
    for name, s in selected.items():
        log.info("  %-22s %8.4f %+12.4f %9.2f%%  fallbacks=%d", name, s["net_sharpe"],
                 s["net_sharpe_diff_vs_benchmark"],
                 s["net_geometric_annual_return"] * 100, s["fallback_count"])
    log.info("")
    log.info("  telemetry: %d fallback in %s candidate fits; benchmark %d/%d (all pre-test)",
             total_fallbacks, f"{total_fits:,}", bench_tel["fallback_count"], bench_tel["n_fits"])
    log.info("  -> %s", OUT)
    return artifact


if __name__ == "__main__":
    run()
