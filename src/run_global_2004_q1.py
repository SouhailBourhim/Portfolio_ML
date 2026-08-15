"""
run_global_2004_q1.py — Q1 of the frozen protocol, and nothing else.

    regime_conditional  versus  max_sharpe,  net of costs, on global_2004.

Addresses: P1, P4 — the first test of the regime layer on an opportunity set
that is both free of the stale-price lead/lag signature and empirically
allocation-expressive under the project's own 25% cap. Q1 asks whether
conditioning the optimizer on a detected regime beats the pre-specified
classical comparator; §12's `etf_2017` finding could not answer that, because
there the constraint dominated the objective and no optimizer expressed a view.

SCOPE, deliberately narrow (pre-registration §5, Q1):

  * TWO strategies. `max_sharpe` is the comparator, named in the frozen
    protocol before any result existed. No other strategy is evaluated here.
  * NO DSR, NO White Reality Check, NO Hansen SPA. Q1 is a SINGLE hypothesis
    fixed in advance; correcting it over a challenger grid it was never part
    of would be over-conservative and could bury a real effect. Q2 is the
    search and carries the full correction.
  * TWO INDEPENDENT VERDICTS. `economically_material` (an absolute
    ΔSharpe ≥ 0.05) and `statistical_evidence` (the paired test) are reported
    as separate booleans and never collapsed into a `wins` field. Clearing one
    licenses nothing about the other, and a single flag would hide exactly
    which of the two was actually met.

RUN-ONCE DISCIPLINE. This is executed once, from a clean committed revision.
A technical defect permits a documented full rerun; an unfavourable result
does not. That asymmetry is the whole point of having frozen the protocol.

Usage:
    python src/run_global_2004_q1.py
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

from backtest import build_cost_vector, run_backtest  # noqa: E402
from global_universe import load_global_config  # noqa: E402
from metrics import (  # noqa: E402
    annualized_sharpe,
    annualized_return,
    max_drawdown,
    paired_block_bootstrap,
)
from provenance import _sha256, git_revision  # noqa: E402
from run_phase4 import _BASELINE_FACTORY, _validate_regime_strategy_names  # noqa: E402
from run_phase5 import split_train_test  # noqa: E402
from strategies import MaxSharpe, RegimeConditionalStrategy  # noqa: E402
from utils import load_params  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_global_2004_q1")

# The pre-specified comparator and candidate. Constants, not configuration:
# the protocol named these two before any result existed, and a config knob
# here would be a place for the comparison to drift.
CANDIDATE = "regime_conditional"
COMPARATOR = "max_sharpe"

# Absolute improvement in net Sharpe that counts as economically material.
# The project's existing constant, with its existing meaning
# (experiments/regime_conditional_cap.py:107). NOT a percentage — a relative
# form was proposed and overruled in review because it is undefined when the
# comparator is at or below zero and unstable near it.
MATERIAL_MARGIN = 0.05


def _build_pair(params: dict):
    """Instantiate exactly the two strategies Q1 compares.

    Freshly constructed, never reused: `RegimeConditionalStrategy` carries a
    per-instance `regime_log`, and sharing one across runs would blend
    unrelated histories into a single diagnostic.
    """
    bp = params["backtest"]
    regime_params = params["regime"]
    _validate_regime_strategy_names(regime_params)
    max_weight, rf = bp["max_weight"], bp["risk_free_annual"]

    candidate = RegimeConditionalStrategy(
        bull_strategy=_BASELINE_FACTORY[regime_params["bull_strategy"]](max_weight, rf),
        bear_strategy=_BASELINE_FACTORY[regime_params["bear_strategy"]](max_weight, rf),
        n_states=regime_params["n_states"],
        n_restarts=regime_params["n_restarts"],
        random_state_base=regime_params["random_state_base"],
        covariance_type=regime_params["covariance_type"],
        min_regime_train_days=regime_params["min_regime_train_days"],
    )
    comparator = MaxSharpe(max_weight=max_weight, risk_free_annual=rf)
    return candidate, comparator


def _describe(result, test_start: pd.Timestamp, rf: float) -> dict:
    """Per-strategy economics on the frozen test segment only."""
    net = result.net_returns.loc[result.net_returns.index >= test_start]
    gross = result.gross_returns.loc[result.gross_returns.index >= test_start]
    reb = result.turnover.index >= test_start

    reports = result.fit_reports
    in_test = reports.loc[reports.index >= test_start] if not reports.empty else reports
    effective = (
        in_test["model_effective"].value_counts().to_dict() if not in_test.empty else {}
    )
    fallbacks = (
        int((in_test["fit_status"] == "fallback").sum()) if not in_test.empty else 0
    )

    return {
        "strategy": result.strategy_name,
        "net_sharpe": round(float(annualized_sharpe(net, risk_free_annual=rf)), 4),
        "gross_sharpe": round(float(annualized_sharpe(gross, risk_free_annual=rf)), 4),
        "net_annual_return": round(float(annualized_return(net)), 6),
        "gross_annual_return": round(float(annualized_return(gross)), 6),
        "max_drawdown": round(float(max_drawdown(net)), 6),
        "avg_turnover": round(float(result.turnover.loc[reb].mean()), 6),
        "total_cost_fraction": round(float(result.costs.loc[reb].sum()), 6),
        "avg_cost_per_rebalance": round(float(result.costs.loc[reb].mean()), 8),
        "n_rebalances_in_test": int(reb.sum()),
        "n_test_days": int(len(net)),
        # "A strategy labelled dcc_garch that fell back to Ledoit-Wolf is, on
        # those dates, a Ledoit-Wolf result under a DCC-GARCH label."
        "fallback_count": fallbacks,
        "fallback_rate_in_test": (
            round(fallbacks / len(in_test), 4) if len(in_test) else 0.0
        ),
        "effective_models": {str(k): int(v) for k, v in effective.items()},
    }


def run() -> dict:
    """Execute Q1 once and write its artifact."""
    cfg = load_global_config()
    params = load_params()
    bp = params["backtest"]
    boot = params["phase5"]["bootstrap"]
    rf = bp["risk_free_annual"]

    readiness_path = ROOT / cfg["paths"]["readiness"]
    readiness = json.loads(readiness_path.read_text())
    if readiness.get("verdict") != "READY":
        raise RuntimeError(
            f"Readiness artifact reports {readiness.get('verdict')!r}, not READY. "
            "Checkpoint 1 gates the evaluation; do not compute a portfolio "
            "return on a universe that has not passed them."
        )
    if readiness["provenance"]["git_revision"].endswith("-dirty"):
        raise RuntimeError(
            "The readiness artifact was produced from a dirty tree, so it does "
            "not identify the code that made it. Regenerate it from a clean "
            "committed revision before evaluating."
        )

    log.info("=== global_2004 Q1: %s vs %s ===", CANDIDATE, COMPARATOR)

    returns = pd.read_parquet(ROOT / cfg["paths"]["gold_returns"])
    features = pd.read_parquet(ROOT / cfg["paths"]["gold_features"])

    # The FROZEN split helper, with the frozen fraction. No custom dates.
    _, test_start = split_train_test(returns, params["phase5"]["test_frac"])
    log.info(
        "Frozen test segment: %s -> %s (test_frac=%.2f)",
        test_start.date(), returns.index.max().date(), params["phase5"]["test_frac"],
    )

    cost_vector = build_cost_vector(
        returns.columns,
        etf_cost_bps=bp["costs_bps"]["etf"],
        bvc_cost_bps=bp["costs_bps"]["bvc"],
    )
    candidate_strategy, comparator_strategy = _build_pair(params)

    kwargs = dict(
        rebalance_freq=bp["rebalance_freq"],
        min_train_days=bp["min_train_days"],
        cost_bps=cost_vector,
        max_weight=bp["max_weight"],
        risk_free_annual=rf,
        extras={"features": features},
        universe_name="global_2004",
    )

    log.info("Running %s...", CANDIDATE)
    cand_result = run_backtest(returns, candidate_strategy, **kwargs)
    log.info("Running %s...", COMPARATOR)
    comp_result = run_backtest(returns, comparator_strategy, **kwargs)

    cand_net = cand_result.net_returns.loc[cand_result.net_returns.index >= test_start]
    comp_net = comp_result.net_returns.loc[comp_result.net_returns.index >= test_start]

    # Identical indexes are REQUIRED, not repaired. Aligning silently would
    # change which days are compared, which is a different experiment.
    if not cand_net.index.equals(comp_net.index):
        raise ValueError(
            "Q1 requires identical test-segment return indexes; refusing to "
            f"align silently. candidate={len(cand_net)} rows, "
            f"comparator={len(comp_net)} rows."
        )

    cand_reb = cand_result.turnover.index >= test_start
    comp_reb = comp_result.turnover.index >= test_start

    paired = paired_block_bootstrap(
        candidate=cand_net,
        benchmark=comp_net,
        candidate_turnover=cand_result.turnover.loc[cand_reb],
        benchmark_turnover=comp_result.turnover.loc[comp_reb],
        candidate_cost=cand_result.costs.loc[cand_reb],
        benchmark_cost=comp_result.costs.loc[comp_reb],
        block_len=boot["block_len"],
        n_boot=boot["n_boot"],
        alpha=boot["alpha"],
        risk_free_annual=rf,
        seed=boot["seed"],
    )

    cand_desc = _describe(cand_result, test_start, rf)
    comp_desc = _describe(comp_result, test_start, rf)

    sharpe_diff = cand_desc["net_sharpe"] - comp_desc["net_sharpe"]
    alpha = boot["alpha"]

    # TWO INDEPENDENT VERDICTS. Never collapsed: one asks whether the
    # difference is big enough to matter to an allocator, the other whether
    # the data can distinguish it from zero. Either can hold without the other.
    economically_material = bool(sharpe_diff >= MATERIAL_MARGIN)
    statistical_evidence = bool(paired["p_value_no_outperformance"] < alpha)

    artifact = {
        "provenance": {
            "universe": "global_2004",
            "base_currency": "USD",
            "base_currency_source": (
                "Eligibility rule E1 — every instrument is US-listed and "
                "USD-denominated. Single-currency, so no FX conversion is "
                "applied and no currency exposure is embedded in these returns."
            ),
            "hedge_status": "not applicable — single-currency universe",
            "currency_converted": False,
            "data_range": {
                "start": str(returns.index.min().date()),
                "end": str(returns.index.max().date()),
                "n_rows": int(len(returns)),
                "n_assets": int(returns.shape[1]),
            },
            "test_segment": {
                "start": str(test_start.date()),
                "end": str(returns.index.max().date()),
                "n_days": int(len(cand_net)),
                "test_frac": float(params["phase5"]["test_frac"]),
                "split_helper": "run_phase5.split_train_test (frozen, no custom dates)",
            },
            "git_revision": git_revision(ROOT),
            "generated_at": pd.Timestamp.now().isoformat(),
            "source_artifacts": {
                rel: _sha256(ROOT / rel)
                for rel in (
                    "src/run_global_2004_q1.py",
                    "src/global_universe.py",
                    "params_global_2004.yaml",
                    "docs/GLOBAL_UNIVERSE_PREREGISTRATION.md",
                    cfg["paths"]["gold_returns"],
                    cfg["paths"]["gold_features"],
                    cfg["paths"]["readiness"],
                )
            },
        },
        "question": "Q1",
        "hypothesis": (
            f"H1: {CANDIDATE} outperforms the pre-specified classical "
            f"comparator {COMPARATOR}, net of transaction costs, on the frozen "
            "test segment of global_2004."
        ),
        "candidate": cand_desc,
        "comparator": comp_desc,
        "observed_difference": {
            "net_sharpe_diff": round(float(sharpe_diff), 4),
            "annual_return_diff": round(float(paired["ann_return_diff"]), 6),
            "avg_turnover_diff": paired.get("avg_turnover_diff"),
            "avg_cost_diff": paired.get("avg_cost_diff"),
            "max_drawdown_diff": round(
                float(cand_desc["max_drawdown"] - comp_desc["max_drawdown"]), 6
            ),
        },
        "paired_inference": {
            "method": (
                "Paired moving-block bootstrap on identical test dates. Both "
                "series resampled with the SAME block indices, preserving "
                "serial dependence within each strategy and same-day "
                "correlation between them."
            ),
            "one_sided_null_centred_p_value": round(
                float(paired["p_value_no_outperformance"]), 6
            ),
            "p_value_definition": (
                "H0: no outperformance. Resampled differences are recentred on "
                "zero to simulate the null; p is the share of that null "
                "distribution at or beyond the OBSERVED difference. It is NOT "
                "the fraction of draws above zero — that is reported "
                "separately as prob_sharpe_diff_positive."
            ),
            "prob_sharpe_diff_positive": round(
                float(paired["prob_sharpe_diff_positive"]), 4
            ),
            "sharpe_diff_ci": [round(float(x), 4) for x in paired["sharpe_diff_ci"]],
            "ann_return_diff_ci": [
                round(float(x), 6) for x in paired["ann_return_diff_ci"]
            ],
            "ci_alpha": float(alpha),
            "block_len": int(boot["block_len"]),
            "n_boot": int(boot["n_boot"]),
            "seed": int(boot["seed"]),
        },
        "verdicts": {
            "economically_material": economically_material,
            "economic_rule": (
                f"net_sharpe_diff >= {MATERIAL_MARGIN} ABSOLUTE Sharpe points "
                "(not a percentage)"
            ),
            "statistical_evidence": statistical_evidence,
            "statistical_rule": f"one-sided null-centred p < {alpha}",
            "note": (
                "These are INDEPENDENT and are deliberately not collapsed into "
                "a single 'wins' field. Economic materiality asks whether the "
                "difference is large enough to matter; statistical evidence "
                "asks whether the data can distinguish it from zero. Either "
                "can hold without the other, and reporting one flag would hide "
                "which."
            ),
        },
        "multiple_testing": {
            "applied": False,
            "reason": (
                "Q1 is a SINGLE hypothesis with a comparator named in the "
                "frozen protocol before any result existed. It was not part of "
                "the challenger search, so DSR, White Reality Check and Hansen "
                "SPA are deliberately NOT computed: correcting a pre-specified "
                "comparison over a grid it never entered would be "
                "over-conservative. Q2 is the search and carries the full "
                "correction."
            ),
        },
        "limitations": [
            "RESIDUAL OUTER SELECTION. The multiple-testing correction that "
            "does apply to Q2 corrects the STRATEGY search. Nothing corrects "
            "the outer selection involved in DESIGNING a new universe after "
            "measuring that the two released ones were each defective. Any "
            "positive result here must be reported with that residual "
            "multiplicity stated.",
            "The readiness diagnostics that shaped this universe were computed "
            "before the protocol was frozen and are disclosed in "
            "docs/GLOBAL_UNIVERSE_PREREGISTRATION.md §7. No performance "
            "quantity was among them.",
            "USD numéraire. Returns are single-currency USD and carry no FX "
            "exposure, so they are NOT directly comparable to full_2021, which "
            "is expressed in MAD.",
            "One test segment, one split. No nested walk-forward is run here; "
            "§12G showed point orderings on a single split can be unstable to "
            "evaluation design.",
        ],
    }

    out_path = ROOT / "data" / "gold" / "global_2004_q1_results.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log.info("")
    log.info("  %-22s net Sharpe %.4f  turnover %.4f  maxDD %.4f  fallbacks %d",
             CANDIDATE, cand_desc["net_sharpe"], cand_desc["avg_turnover"],
             cand_desc["max_drawdown"], cand_desc["fallback_count"])
    log.info("  %-22s net Sharpe %.4f  turnover %.4f  maxDD %.4f  fallbacks %d",
             COMPARATOR, comp_desc["net_sharpe"], comp_desc["avg_turnover"],
             comp_desc["max_drawdown"], comp_desc["fallback_count"])
    log.info("")
    log.info("  observed dSharpe      : %+.4f   CI %s", sharpe_diff,
             artifact["paired_inference"]["sharpe_diff_ci"])
    log.info("  one-sided p (null)    : %.4f", paired["p_value_no_outperformance"])
    log.info("  P(dSharpe > 0)        : %.4f", paired["prob_sharpe_diff_positive"])
    log.info("  d annual return       : %+.4f", paired["ann_return_diff"])
    log.info("")
    log.info("  economically_material : %s  (>= %.2f absolute)",
             economically_material, MATERIAL_MARGIN)
    log.info("  statistical_evidence  : %s  (p < %.2f)", statistical_evidence, alpha)
    log.info("")
    log.info("  Q1 complete -> %s", out_path)
    log.info("  Q2 is a SEPARATE step. Commit this artifact before starting it.")
    return artifact


if __name__ == "__main__":
    run()
