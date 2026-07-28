"""
regime_conditional_cap.py — Make the CONSTRAINT the regime-conditional decision.

WHY THIS EXPERIMENT EXISTS. This project's own evidence says the weight cap
does more estimation-error control than any covariance model it has tried: on
`etf_2017`, sweeping `max_weight` alone moves the best classical net Sharpe
0.9525 → 0.8650, a 10.1% swing larger than the gap between any two models on
that universe (CLAUDE.md §10.1). Jagannathan & Ma (2003) explains why — a
binding long-only weight constraint is mathematically equivalent to shrinking
the covariance matrix.

Phase 4 conditions the COVARIANCE MODEL on the detected regime. If the cap is
the stronger regularizer, the obvious untested move is to condition the CAP on
the regime instead: tighten it in a detected bear (more shrinkage exactly when
correlations spike and estimation error hurts most — P1 and P3 together), relax
it in a bull (let the optimizer express a view when it is likelier to be right).

Notably this needs NO new production code. `RegimeConditionalStrategy` already
accepts sub-strategy instances, and each baseline already owns its `max_weight`,
so a regime-conditional cap is `MaxSharpe(max_weight=bull_cap)` +
`MinVarianceLW(max_weight=bear_cap)`. The engine is handed the LOOSER of the two
so its trust-boundary check still binds; the strategy self-imposes the tighter
one. Configuration, not machinery — the same "swap one thing into the unmodified
engine" pattern every Phase 4 addition used.

THE CONTROL THAT MAKES THIS HONEST. "Tighter bear cap wins" would be a weak
finding on its own: a tighter cap might simply be better everywhere, with the
regime label contributing nothing. So the grid includes
  * FIXED caps at both endpoints — isolates "the cap level helped", and
  * an INVERTED variant (loose in bear, tight in bull) — the deliberately WRONG
    direction. If inverted also beats the baseline, the effect is not about
    regimes at all, and the honest conclusion is that we found cap sensitivity,
    not a regime signal.

PRE-REGISTERED OUTCOMES, fixed before the run:
  (A) A regime-conditional cap beats the 0.25 baseline BY A MATERIAL MARGIN
      AND beats both the matching fixed caps AND beats the inverted control
      → conditioning the CONSTRAINT on the regime adds real value.
  (B) It beats the baseline materially, but a fixed cap or the inverted control
      does too → cap-level sensitivity, not a regime effect. Report as such.
  (C) No variant beats the baseline materially
      → the 25% cap is already at/near its optimum here; the lever is spent.

MATERIALITY, and a disclosure. The first draft of this script tested "beats the
baseline" with a bare `>`. A smoke run returned a candidate at +0.0016 Sharpe
against intervals ~1.16 wide, which that rule would have reported as outcome A —
a meaningless difference dressed as a finding. `MATERIAL_MARGIN` below was added
in response, BEFORE the real run. The change makes the test STRICTER (harder to
claim success), and the threshold is a judgement call for interpretability, not
a value tuned against results. Recording it here because adjusting a decision
rule after seeing any output is exactly what pre-registration exists to prevent,
so the adjustment has to be visible rather than quietly folded in.

No outcome here is a significance claim: every variant is evaluated on the same
window with heavily overlapping bootstrap intervals, and the script reports that
overlap alongside every verdict.

Addresses: P1 (constraint as shrinkage), P3 (defensive posture when
diversification breaks down), P4 (pre-registered outcomes, error bars, and a
control designed to kill the flattering interpretation).

Usage:
    python experiments/regime_conditional_cap.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import build_cost_vector, run_backtest        # noqa: E402
from metrics import (                                        # noqa: E402
    annualized_sharpe,
    block_bootstrap_sharpe_ci,
    max_drawdown,
)
from run_phase4 import load_features, load_universe          # noqa: E402
from strategies import (                                     # noqa: E402
    MaxSharpe,
    MinVarianceLW,
    RegimeConditionalStrategy,
)
from utils import load_params                                # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("backtest").setLevel(logging.WARNING)
logging.getLogger("regime").setLevel(logging.ERROR)   # thin-window fallbacks are expected
log = logging.getLogger("regime_cap")

OUT_PATH = ROOT / "data" / "gold" / "regime_conditional_cap.json"

# Minimum net-Sharpe improvement over the baseline that counts as a result.
# 0.05 is a judgement call, chosen for interpretability: it is roughly 4% of the
# baseline and still an order of magnitude INSIDE the bootstrap interval width
# (~1.2), so clearing it is necessary but nowhere near sufficient for
# significance. See the module docstring for why this threshold exists at all.
MATERIAL_MARGIN = 0.05

# (label, bull_cap, bear_cap). bear < bull = "defensive": tighten (shrink harder)
# when the HMM says bear. INVERTED is the control, not a candidate.
# Caps must satisfy n_assets x cap >= 1, so the feasible floor is 1/5 = 0.20 on
# etf_2017 and 1/9 = 0.112 on full_2021 — hence a per-universe grid.
GRIDS = {
    "full_2021": [
        ("baseline_25_25",      0.25, 0.25),   # == the shipped regime_conditional
        ("defensive_25_15",     0.25, 0.15),
        ("defensive_25_125",    0.25, 0.125),
        ("aggressive_40_25",    0.40, 0.25),
        ("both_40_15",          0.40, 0.15),
        ("INVERTED_15_40",      0.15, 0.40),   # control: the wrong direction
    ],
    "etf_2017": [
        ("baseline_25_25",      0.25, 0.25),
        ("defensive_25_20",     0.25, 0.20),   # 5 x 0.20 = 1.0 -> equal weight in bear
        ("aggressive_40_25",    0.40, 0.25),
        ("both_40_20",          0.40, 0.20),
        ("INVERTED_20_40",      0.20, 0.40),   # control
    ],
}

# Fixed-cap references: isolates "a tighter/looser cap helped" from "conditioning
# the cap on the regime helped". Same optimizer, no regime switching.
FIXED_CAPS = {"full_2021": [0.125, 0.15, 0.25, 0.40], "etf_2017": [0.20, 0.25, 0.40]}


def summarize(net: pd.Series, turnover: pd.Series, boot: dict, rf: float) -> dict:
    point, lo, hi = block_bootstrap_sharpe_ci(
        net, block_len=boot["block_len"], n_boot=boot["n_boot"],
        alpha=boot["alpha"], risk_free_annual=rf,
    )
    return {
        "sharpe_net": round(float(annualized_sharpe(net, rf)), 4),
        "ci": [round(float(lo), 4), round(float(hi), 4)],
        "ci_width": round(float(hi - lo), 4),
        "max_drawdown": round(float(max_drawdown(net)), 4),
        "avg_turnover": round(float(turnover.mean()), 4),
    }


def main() -> dict:
    params = load_params()
    bp = params["backtest"]
    rf, boot = bp["risk_free_annual"], params["phase5"]["bootstrap"]
    regime_kwargs = dict(
        n_states=params["regime"]["n_states"],
        n_restarts=params["regime"]["n_restarts"],
        random_state_base=params["regime"]["random_state_base"],
        covariance_type=params["regime"]["covariance_type"],
        min_regime_train_days=params["regime"]["min_regime_train_days"],
    )

    out: dict[str, dict] = {"grids": {k: [list(v) for v in g] for k, g in GRIDS.items()},
                            "universes": {}}

    for universe, grid in GRIDS.items():
        returns = load_universe(bp["universes"][universe])
        features = load_features(params["ml_features"]["outputs"][universe])
        cost_vector = build_cost_vector(
            returns.columns,
            etf_cost_bps=bp["costs_bps"]["etf"], bvc_cost_bps=bp["costs_bps"]["bvc"],
        )
        n_assets = returns.shape[1]
        log.info("=== %s: %d rows x %d assets (cap floor %.3f) ===",
                 universe, len(returns), n_assets, 1.0 / n_assets)

        results: dict[str, dict] = {}

        # ── Regime-conditional caps ──────────────────────────────────────────
        for label, bull_cap, bear_cap in grid:
            strategy = RegimeConditionalStrategy(
                bull_strategy=MaxSharpe(max_weight=bull_cap, risk_free_annual=rf),
                bear_strategy=MinVarianceLW(max_weight=bear_cap),
                **regime_kwargs,
            )
            # The engine polices the LOOSER cap; the strategy self-imposes the
            # tighter one per regime. Handing it the tighter value would reject
            # legitimate bull-regime weights at the trust boundary.
            engine_cap = max(bull_cap, bear_cap)
            result = run_backtest(
                returns, strategy,
                rebalance_freq=bp["rebalance_freq"], min_train_days=bp["min_train_days"],
                cost_bps=cost_vector, extras={"features": features},
                universe_name=universe, max_weight=engine_cap,
            )
            entry = summarize(result.net_returns, result.turnover, boot, rf)
            regimes = [r["regime"] for r in strategy.regime_log]
            entry.update({
                "bull_cap": bull_cap, "bear_cap": bear_cap,
                "kind": "control" if label.startswith("INVERTED") else "candidate",
                "n_bear_rebalances": sum(1 for r in regimes if r == "bear"),
                "n_rebalances": len(regimes),
            })
            results[label] = entry
            log.info("  %-20s bull %.3f / bear %.3f -> Sharpe %6.4f  [%.2f, %.2f]  bear %d/%d",
                     label, bull_cap, bear_cap, entry["sharpe_net"],
                     entry["ci"][0], entry["ci"][1],
                     entry["n_bear_rebalances"], entry["n_rebalances"])

        # ── Fixed-cap references (no regime switching) ───────────────────────
        for cap in FIXED_CAPS[universe]:
            for name, strategy in (
                ("fixed_minvarlw", MinVarianceLW(max_weight=cap)),
                ("fixed_maxsharpe", MaxSharpe(max_weight=cap, risk_free_annual=rf)),
            ):
                result = run_backtest(
                    returns, strategy,
                    rebalance_freq=bp["rebalance_freq"], min_train_days=bp["min_train_days"],
                    cost_bps=cost_vector, universe_name=universe, max_weight=cap,
                )
                label = f"{name}_{cap:.3f}".replace("0.", "")
                entry = summarize(result.net_returns, result.turnover, boot, rf)
                entry.update({"kind": "fixed_reference", "cap": cap})
                results[label] = entry
                log.info("  %-20s cap %.3f -> Sharpe %6.4f", label, cap, entry["sharpe_net"])

        # ── Verdict against the pre-registered outcomes ──────────────────────
        baseline = results["baseline_25_25"]["sharpe_net"]
        candidates = {k: v for k, v in results.items()
                      if v.get("kind") == "candidate" and k != "baseline_25_25"}
        controls = {k: v for k, v in results.items() if v.get("kind") == "control"}
        fixed = {k: v for k, v in results.items() if v.get("kind") == "fixed_reference"}

        best_candidate = max(candidates, key=lambda k: candidates[k]["sharpe_net"])
        best_control = max(controls, key=lambda k: controls[k]["sharpe_net"])
        best_fixed = max(fixed, key=lambda k: fixed[k]["sharpe_net"])

        def beats(entry) -> bool:
            return entry["sharpe_net"] - baseline >= MATERIAL_MARGIN

        cand_beats = beats(candidates[best_candidate])
        control_beats = beats(controls[best_control])
        fixed_beats = beats(fixed[best_fixed])

        if cand_beats and not control_beats and not fixed_beats:
            verdict = "A — regime-conditional cap adds value beyond level and direction"
        elif cand_beats:
            verdict = "B — cap-level sensitivity, NOT a regime effect (control/fixed also clear it)"
        else:
            verdict = "C — no cap variant beats the 0.25 baseline materially; the lever is spent"

        # Whatever the verdict, state the uncertainty next to it: these variants
        # share a window and their intervals overlap almost completely, so the
        # ranking is not a significance claim and must never be quoted as one.
        base_ci = results["baseline_25_25"]["ci"]
        cand_ci = candidates[best_candidate]["ci"]
        overlap = min(base_ci[1], cand_ci[1]) - max(base_ci[0], cand_ci[0])

        out["universes"][universe] = {
            "n_assets": n_assets,
            "cap_floor": round(1.0 / n_assets, 4),
            "baseline_sharpe": baseline,
            "results": results,
            "best_candidate": best_candidate,
            "best_control": best_control,
            "best_fixed_reference": best_fixed,
            "material_margin": MATERIAL_MARGIN,
            "best_candidate_lift": round(
                candidates[best_candidate]["sharpe_net"] - baseline, 4
            ),
            "baseline_vs_best_candidate_ci_overlap": round(float(overlap), 4),
            "statistically_significant": False,   # never, on overlapping CIs
            "verdict": verdict,
        }
        log.info("  VERDICT %s: %s", universe, verdict)
        log.info("     best candidate %s lift %+.4f (margin %.2f); CI overlap %.3f "
                 "of widths %.3f/%.3f — NOT significant either way",
                 best_candidate, candidates[best_candidate]["sharpe_net"] - baseline,
                 MATERIAL_MARGIN, overlap,
                 results["baseline_25_25"]["ci_width"],
                 candidates[best_candidate]["ci_width"])

    OUT_PATH.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", OUT_PATH)
    return out


if __name__ == "__main__":
    main()
