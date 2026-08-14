"""
etf_cap_verdict.py — Does regime/covariance ML help on etf_2017 when the
optimizer is actually allowed to act?

Since Phase 4 the project has concluded that regime + dynamic-covariance ML
"adds no value on etf_2017". `docs/ETF_DEEP_HISTORY_EXPERIMENT.md` showed that
claim is **not falsifiable as stated**: with 5 assets and a 25% cap,
5 x 0.25 = 1.25, so any feasible long-only portfolio must hold at least four
assets with POSITIVE WEIGHT, and no asset may exceed equal weight by more than
five percentage points. The arithmetic alone does not force a corner — equal
weight stays feasible with nothing at the cap — but EMPIRICALLY the constraint
dominates the optimizer rather than the covariance model: at 0.25
`min_variance_lw` produced ONE allocation across 248 rebalances (versus 171 at
0.30), and min-var / max-Sharpe / regime returned byte-identical weights
post-2018.

You cannot conclude "the model doesn't help" from a setup where the model
cannot express a view. This experiment removes that objection: it re-runs the
same comparison across caps, from the binding 0.25 up to effectively
unconstrained, and asks whether the ML-vs-classical verdict changes once the
optimizer has room.

Three outcomes, all publishable, decided in advance so the result cannot be
rationalised after the fact:

  A. ML wins at looser caps  -> the Phase 4 negative result was an artefact
     of the constraint, and the project's etf_2017 conclusion must be
     rewritten.
  B. ML still loses at every cap -> the Phase 4 conclusion survives a much
     stronger test, and is now defensible rather than merely unfalsifiable.
  C. Everything degrades as the cap loosens -> the cap is the dominant
     performance driver (Jagannathan & Ma 2003), which is itself the finding
     and reframes the constraint as a modelling choice, not a formality.

Runs on the DEEP 2004-11 window adopted in production, so the verdict rests on
20.7 years and 248 rebalances rather than 8.5 years.

Usage:
    .venv/bin/python experiments/etf_cap_verdict.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from backtest import build_cost_vector, run_backtest
from metrics import annualized_sharpe, block_bootstrap_sharpe_ci, max_drawdown
from strategies import (
    EqualWeight,
    MaxSharpe,
    MinVarianceLW,
    RegimeConditionalStrategy,
)
from provenance import build_provenance
from utils import load_params

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("etf_cap_verdict")

# 0.25 is the project default and is fully binding on 5 assets.
# 0.35/0.40 leave real freedom; 1.00 is the unconstrained reference.
CAPS = (0.25, 0.30, 0.35, 0.40, 1.00)

CLASSICAL = ("equal_weight", "min_variance_lw", "max_sharpe")
ML = "regime_conditional"


def build_strategies(cap: float, params: dict) -> dict:
    rf = params["backtest"]["risk_free_annual"]
    rp = params["regime"]
    return {
        "equal_weight": EqualWeight(),
        "min_variance_lw": MinVarianceLW(max_weight=cap),
        "max_sharpe": MaxSharpe(max_weight=cap, risk_free_annual=rf),
        ML: RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=cap, risk_free_annual=rf),
            bear_strategy=MinVarianceLW(max_weight=cap),
            n_states=rp["n_states"], n_restarts=rp["n_restarts"],
            random_state_base=rp["random_state_base"],
            covariance_type=rp["covariance_type"],
            min_regime_train_days=rp["min_regime_train_days"],
            features=rp.get("features"),
        ),
    }


def main() -> None:
    params = load_params()
    bt = params["backtest"]
    boot = params["phase5"]["bootstrap"]

    returns = pd.read_parquet(ROOT / "data/gold/log_returns_etf.parquet")
    features = pd.read_parquet(ROOT / "data/gold/ml_features_etf.parquet")
    features = features.reindex(returns.index).ffill()
    cost_vector = build_cost_vector(
        list(returns.columns),
        etf_cost_bps=bt["costs_bps"]["etf"], bvc_cost_bps=bt["costs_bps"]["bvc"],
    )
    log.info("etf_2017 (deep): %s -> %s, %d rows, %d assets",
             returns.index.min().date(), returns.index.max().date(),
             len(returns), returns.shape[1])
    log.info("A %d-asset universe needs >= %d assets at a %.0f%% cap to be feasible.",
             returns.shape[1], int(np.ceil(1 / 0.25)), 25)

    results = {}
    for cap in CAPS:
        log.info("")
        log.info("=== cap %.2f ===", cap)
        row = {}
        for name, strat in build_strategies(cap, params).items():
            result = run_backtest(
                returns, strat, rebalance_freq=bt["rebalance_freq"],
                min_train_days=bt["min_train_days"], cost_bps=cost_vector,
                extras={"features": features}, universe_name="etf_cap_sweep",
                max_weight=cap,
            )
            net = result.net_returns
            point, lo, hi = block_bootstrap_sharpe_ci(
                net, block_len=boot["block_len"], n_boot=boot["n_boot"],
                alpha=boot["alpha"], seed=boot["seed"])
            weights = result.target_weights
            row[name] = {
                "sharpe_net": round(float(annualized_sharpe(net, bt["risk_free_annual"])), 4),
                "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                "max_drawdown": round(float(max_drawdown(net)), 4),
                "avg_turnover": round(float(result.turnover.mean()), 4),
                # How much freedom did the optimizer actually have here?
                "distinct_allocations": len(set(map(tuple, np.round(weights.to_numpy(), 6)))),
                "n_rebalances": int(len(weights)),
            }
            log.info("  %-18s Sharpe %.4f  CI [%+.2f, %+.2f]  %4d/%d distinct allocs",
                     name, row[name]["sharpe_net"], lo, hi,
                     row[name]["distinct_allocations"], row[name]["n_rebalances"])
        results[f"{cap:.2f}"] = row

    # ── Verdict ────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 78)
    log.info("VERDICT — does the ML-vs-classical answer change once the cap is loose?")
    log.info("=" * 78)
    log.info("  %-6s %-18s %8s %8s %9s %10s", "cap", "best classical", "clsc", ML[:8],
             "lift", "optim.free")
    verdicts = {}
    for cap_key, row in results.items():
        best_classical = max(CLASSICAL, key=lambda s: row[s]["sharpe_net"])
        base = row[best_classical]["sharpe_net"]
        ml = row[ML]["sharpe_net"]
        lift = (ml - base) / abs(base) * 100
        # "free" = the min-variance optimizer produced more than a handful of
        # distinct allocations, i.e. the cap was not dictating the answer.
        free = row["min_variance_lw"]["distinct_allocations"] > 10
        verdicts[cap_key] = {"best_classical": best_classical, "classical_sharpe": base,
                             "ml_sharpe": ml, "lift_pct": round(lift, 2),
                             "optimizer_free": bool(free)}
        log.info("  %-6s %-18s %8.3f %8.3f %+8.1f%% %10s",
                 cap_key, best_classical, base, ml, lift, "yes" if free else "NO — capped")

    unconstrained = [v for k, v in verdicts.items() if v["optimizer_free"]]
    log.info("")
    if not unconstrained:
        log.info("  No cap in the sweep left the optimizer free — inconclusive.")
    elif all(v["lift_pct"] < 0 for v in unconstrained):
        log.info("  OUTCOME B: regime ML loses at EVERY cap where the optimizer is free.")
        log.info("  The Phase 4 etf_2017 conclusion survives a much stronger test — it is")
        log.info("  now defensible, not merely unfalsifiable.")
    elif all(v["lift_pct"] > 0 for v in unconstrained):
        log.info("  OUTCOME A: regime ML WINS wherever the optimizer is free.")
        log.info("  The Phase 4 negative result was an artefact of the 25%% cap and the")
        log.info("  project's etf_2017 conclusion must be rewritten.")
    else:
        log.info("  MIXED: the sign of the lift depends on the cap. The cap is a modelling")
        log.info("  choice with first-order effect, not a formality.")

    # Is the cap itself the dominant driver? (Jagannathan & Ma 2003)
    sharpes = {k: v["classical_sharpe"] for k, v in verdicts.items()}
    best_cap = max(sharpes, key=sharpes.get)
    log.info("")
    log.info("  Best classical Sharpe across caps: %.3f at cap %s (worst %.3f at cap %s).",
             sharpes[best_cap], best_cap, min(sharpes.values()),
             min(sharpes, key=sharpes.get))
    if best_cap == "0.25":
        log.info("  The TIGHTEST cap is the best-performing configuration — the constraint")
        log.info("  is doing the estimation-error control (Jagannathan & Ma 2003).")

    out = ROOT / "data" / "gold" / "etf_cap_verdict.json"
    # Provenance, for the same reason the nested walk-forward now carries it:
    # this artifact was also a hand-run, untracked experiment, and an untracked
    # result is one nobody can tell has gone stale. It happens to have SURVIVED
    # the base-currency correction intact -- it is scoped to etf_2017, which is
    # single-currency USD and did not move -- but that was luck, not a property
    # anyone could verify without rerunning it. Recording base_currency=USD here
    # is what makes "this one was unaffected" checkable rather than asserted.
    out.write_text(json.dumps({
        "provenance": build_provenance(
            universe="etf_2017",
            returns=returns,
            source_artifacts=[
                params["backtest"]["universes"]["etf_2017"],
                params["ml_features"]["outputs"]["etf_2017"],
                "data/gold/currency_manifest.json",
                "params.yaml",
                "experiments/etf_cap_verdict.py",
            ],
        ),
        "caps": CAPS, "results": results, "verdicts": verdicts,
    }, indent=2))
    log.info("")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
