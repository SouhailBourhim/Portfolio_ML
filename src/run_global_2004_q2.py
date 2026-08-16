"""
run_global_2004_q2.py — Q2 of the frozen protocol.

    RF / XGBoost challenger FAMILY  versus  regime_conditional,
    with the full White (2000) Reality Check and Hansen (2005) SPA
    correction over the complete reachable candidate ledger.

Addresses: P4 — Q1 tested ONE pre-specified comparison and needed no
correction. Q2 is a SEARCH: 240 reachable configurations, any of which could
have been selected. Reporting the best of them as though it had been chosen in
advance is the exact backtest overfitting this project exists to avoid, so the
composite null — "no candidate in the searched family outperforms the
benchmark" — is what gets tested.

WHY THE LEDGER IS THE REACHABLE SPACE, not the configs a search happened to
visit. Correcting for the smaller number would understate the multiplicity and
would do so in the project's own favour. Every RF/XGB hyperparameter point
crossed with every portfolio-lever point is included, and NOTHING is
deduplicated on observed performance — dropping a poor candidate after seeing
it perform poorly is selection, and it is selection that the correction exists
to price.

INTERPRETATION IS FROZEN (pre-registration §10.3, amendment 2), because with
two tests and two endpoints there are four p-values in this artifact and
quoting the smallest is one sentence away from looking like reporting:

  * PRIMARY endpoint   — net-Sharpe differential.
  * SECONDARY endpoint — annualized mean-return differential, labelled as
                         such, never promoted.
  * alpha = 0.10, inherited.
  * RC and SPA reported SEPARATELY; no single family p-value is ever formed.
  * concordant_evidence_of_family_outperformance = RC rejects AND SPA rejects,
    on the PRIMARY endpoint only.
  * exactly one rejects -> "test-dependent evidence", NOT established
    outperformance.

RUN-ONCE, from a clean committed revision. A technical defect permits a
documented rerun; an unfavourable result does not.

Usage:
    python src/run_global_2004_q2.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backtest import build_cost_vector, run_backtest  # noqa: E402
from global_universe import load_global_config  # noqa: E402
from metrics import annualized_sharpe, reality_check  # noqa: E402
from model_selection import (  # noqa: E402
    select_ml_hyperparameters,
    select_portfolio_levers,
)
from provenance import _sha256, git_revision  # noqa: E402
from purged_kfold import PurgedWalkForwardSplit  # noqa: E402
from run_global_2004_q1 import _build_pair, _describe  # noqa: E402
from run_phase5 import split_train_test  # noqa: E402
from run_reality_check import MODEL_CLASSES, _label, candidate_configs  # noqa: E402
from utils import load_params  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_global_2004_q2")

BENCHMARK = "regime_conditional"

# Endpoint hierarchy, frozen by amendment 2. `sharpe` is PRIMARY.
PRIMARY_STATISTIC = "sharpe"
SECONDARY_STATISTIC = "mean_return"


def expected_candidate_count(params: dict) -> int:
    """The reachable candidate count, derived from FROZEN CONFIG ALONE.

    Addresses: P4 — computed before any return is calculated, so the executed
    ledger can be checked against a number that no result could have
    influenced. A ledger that silently differs from the reachable space is
    either an under-correction (fewer candidates than were truly reachable) or
    a bug; both must stop the run rather than be discovered afterwards in the
    p-value.
    """
    p5 = params["phase5"]
    n_models = 0
    for grid_key in ("rf_grid", "xgb_grid"):
        grid = p5[grid_key]
        size = 1
        for values in grid.values():
            size *= len(values)
        n_models += size
    n_levers = len(p5["shrink_grid"]) * len(p5["penalty_grid"])
    return n_models * n_levers


def _telemetry(result) -> dict:
    """Fallback counts and effective-model identities for one evaluation."""
    reports = result.fit_reports
    if reports.empty:
        return {"n_fits": 0, "fallback_count": 0, "effective_models": {}}
    return {
        "n_fits": int(len(reports)),
        "fallback_count": int((reports["fit_status"] == "fallback").sum()),
        "degraded_count": int((reports["fit_status"] == "degraded").sum()),
        "effective_models": {
            str(k): int(v) for k, v in reports["model_effective"].value_counts().items()
        },
    }


def run() -> dict:
    cfg = load_global_config()
    params = load_params()
    bp, sig, p5 = params["backtest"], params["ml_signals"], params["phase5"]
    wf, boot = params["walk_forward_cv"], p5["bootstrap"]
    rf_rate, alpha = bp["risk_free_annual"], float(boot["alpha"])

    # ── Gate on Q1 and on readiness ─────────────────────────────────────────
    q1_path = ROOT / "data" / "gold" / "global_2004_q1_results.json"
    if not q1_path.is_file():
        raise RuntimeError("Q1 artifact absent. Q2 must not run before Q1 is frozen.")
    q1 = json.loads(q1_path.read_text())
    if q1["provenance"]["git_revision"].endswith("-dirty"):
        raise RuntimeError("Q1 artifact came from a dirty tree; it is not canonical.")

    log.info("=== global_2004 Q2: RF/XGB family vs %s ===", BENCHMARK)

    # ── Control 1: expected count from frozen config, BEFORE any return ─────
    expected_n = expected_candidate_count(params)
    configs = candidate_configs(params)
    if len(configs) != expected_n:
        raise RuntimeError(
            f"Reachable ledger mismatch BEFORE evaluation: candidate_configs "
            f"produced {len(configs)}, frozen config implies {expected_n}."
        )
    labels = [_label(c) for c in configs]
    if len(set(labels)) != len(labels):
        raise RuntimeError("Candidate labels are not unique; metadata would collide.")
    log.info("Reachable candidate ledger: %d configurations (verified)", expected_n)

    returns = pd.read_parquet(ROOT / cfg["paths"]["gold_returns"])
    features = pd.read_parquet(ROOT / cfg["paths"]["gold_features"])
    train_val, test_start = split_train_test(returns, p5["test_frac"])
    tv_features = features.loc[features.index <= train_val.index[-1]]
    log.info("Frozen test segment: %s -> %s", test_start.date(), returns.index.max().date())

    etf_bps = float(bp["costs_bps"]["etf"])
    costs = build_cost_vector(
        returns.columns, etf_cost_bps=etf_bps, bvc_cost_bps=bp["costs_bps"]["bvc"],
        overrides={t: etf_bps for t in cfg["tickers"]},
    )
    bt_kwargs = dict(
        rebalance_freq=bp["rebalance_freq"], min_train_days=bp["min_train_days"],
        cost_bps=costs, max_weight=bp["max_weight"],
        extras={"features": features}, universe_name="global_2004",
    )
    common = dict(
        max_weight=bp["max_weight"], risk_free_annual=rf_rate,
        min_train_rows=sig["min_train_rows"], short_window=sig["short_window"],
        long_window=sig["long_window"], momentum_windows=sig["momentum_windows"],
        condition_on_regime=sig["condition_on_regime"],
    )

    # ── Control 5: the benchmark must reproduce Q1 EXACTLY ──────────────────
    log.info("Evaluating benchmark %s...", BENCHMARK)
    bench_strategy, _ = _build_pair(params)
    bench_result = run_backtest(returns, bench_strategy, **bt_kwargs)
    bench_net = bench_result.net_returns.loc[bench_result.net_returns.index >= test_start]
    bench_desc = _describe(bench_result, test_start, rf_rate)

    q1_bench = q1["candidate"]
    for field in ("net_sharpe", "net_geometric_annual_return", "max_drawdown",
                  "avg_turnover", "n_rebalances_in_test", "n_test_days"):
        if bench_desc[field] != q1_bench[field]:
            raise RuntimeError(
                f"Benchmark does not reproduce Q1 on {field!r}: Q2={bench_desc[field]} "
                f"vs Q1={q1_bench[field]}. The two questions must be answered "
                "against the identical benchmark realization, or the comparison "
                "across them is meaningless."
            )
    log.info("Benchmark reproduces Q1 exactly (net Sharpe %.4f).", bench_desc["net_sharpe"])

    # ── Control 4: deployable challenger, selected FORWARD-ONLY ─────────────
    # Uses train+validation ONLY. The frozen test segment is never shown to a
    # selector — it is where the verdict is measured, not where it is chosen.
    regime_kwargs = dict(
        n_states=params["regime"]["n_states"], n_restarts=params["regime"]["n_restarts"],
        random_state_base=params["regime"]["random_state_base"],
        covariance_type=params["regime"]["covariance_type"],
        min_regime_train_days=params["regime"]["min_regime_train_days"],
    )
    deployable = {}
    for model_type, grid_key, display in (
        ("random_forest", "rf_grid", "rf_signal_tuned"),
        ("xgboost", "xgb_grid", "xgb_signal_tuned"),
    ):
        log.info("Forward-only selection for %s...", display)
        splitter = PurgedWalkForwardSplit(
            min_train_dates=wf["min_train_dates"], val_dates=wf["val_dates"],
            n_splits=wf["n_splits"], embargo_dates=wf["embargo_dates"],
            label_horizon=wf["label_horizon"], mode=wf["mode"],
            step_dates=wf["step_dates"],
        )
        ml_params, _ = select_ml_hyperparameters(
            train_val, tv_features, p5[grid_key], model_type=model_type,
            short_window=sig["short_window"], long_window=sig["long_window"],
            momentum_windows=sig["momentum_windows"],
            condition_on_regime=sig["condition_on_regime"],
            regime_kwargs=regime_kwargs, splitter=splitter,
        )
        levers, _ = select_portfolio_levers(
            train_val, tv_features, model_type=model_type, ml_params=ml_params,
            shrink_grid=p5["shrink_grid"], penalty_grid=p5["penalty_grid"],
            backtest_kwargs={**bt_kwargs, "extras": {"features": tv_features}},
            ledger=None, universe="global_2004",
        )
        deployable[display] = {
            "model_type": model_type, "ml_params": ml_params,
            "shrinkage_weight": levers["shrinkage_weight"],
            "turnover_penalty": levers["turnover_penalty"],
            "selection": (
                "forward-only PurgedWalkForwardSplit on train+validation for the "
                "hyperparameters, validation-segment net Sharpe for the levers. "
                "The frozen test segment was never shown to either selector."
            ),
        }
        log.info("  %s -> %s levers=%s", display, ml_params, levers)

    # ── Control 2/3: evaluate EVERY reachable candidate ─────────────────────
    series: dict[str, pd.Series] = {}
    telemetry: dict[str, dict] = {}
    started = time.time()
    for i, (config, label) in enumerate(zip(configs, labels), start=1):
        strategy = MODEL_CLASSES[config["model_type"]](
            name="q2_candidate", model_params=config["ml_params"],
            mu_transform="shrink", shrinkage_weight=config["shrinkage_weight"],
            turnover_penalty=config["turnover_penalty"], **common,
        )
        result = run_backtest(returns, strategy, **bt_kwargs)
        net = result.net_returns.loc[result.net_returns.index >= test_start]
        if not net.index.equals(bench_net.index):
            raise ValueError(
                f"Candidate {label!r} has a different test index than the "
                "benchmark; refusing to align silently."
            )
        series[label] = net
        telemetry[label] = _telemetry(result)
        if i % 16 == 0 or i == len(configs):
            log.info("%d/%d candidates (%.0fs elapsed)", i, len(configs),
                     time.time() - started)

    # Control 1, second half: the EXECUTED ledger must match the expectation.
    if len(series) != expected_n:
        raise RuntimeError(
            f"Executed ledger ({len(series)}) differs from the expected "
            f"reachable count ({expected_n})."
        )

    # ── Family tests: 2 tests x 2 endpoints, all four persisted ─────────────
    tests: dict[str, dict] = {}
    for endpoint, statistic in (
        ("primary_sharpe", PRIMARY_STATISTIC),
        ("secondary_mean_return", SECONDARY_STATISTIC),
    ):
        log.info("Reality Check + SPA on the %s endpoint...", endpoint)
        res = reality_check(
            series, bench_net, statistic=statistic,
            block_len=boot["block_len"], n_boot=boot["n_boot"],
            seed=boot["seed"], risk_free_annual=rf_rate,
        )
        tests[endpoint] = {
            "statistic": statistic,
            "reality_check_p_value": round(float(res["reality_check_p_value"]), 6),
            "spa_p_value": round(float(res["spa_p_value"]), 6),
            "rc_rejects_at_alpha": bool(res["reality_check_p_value"] < alpha),
            "spa_rejects_at_alpha": bool(res["spa_p_value"] < alpha),
            "n_candidates": int(res.get("n_candidates", len(series))),
            "n_candidates_beating_benchmark": int(
                res.get("n_candidates_beating_benchmark", 0)
            ),
            # The best candidate is reported for transparency, NOT as a
            # result: it is the maximum of a 240-wide search, which is
            # precisely the quantity the correction exists to deflate.
            "best_candidate": res.get("best_candidate"),
            "best_differential": (
                round(float(res["best_differential"]), 6)
                if res.get("best_differential") is not None else None
            ),
            "best_candidate_note": (
                "The argmax of the search. Its raw differential is NOT "
                "evidence of outperformance — the RC/SPA p-values above are "
                "what price the fact that it was selected from 240."
            ),
            "spa_candidates_retained": res.get("spa_candidates_retained"),
        }

    primary = tests["primary_sharpe"]
    concordant = bool(primary["rc_rejects_at_alpha"] and primary["spa_rejects_at_alpha"])
    n_reject = int(primary["rc_rejects_at_alpha"]) + int(primary["spa_rejects_at_alpha"])
    if concordant:
        evidence_status = "concordant_evidence_of_family_outperformance"
    elif n_reject == 1:
        evidence_status = "test_dependent_evidence"
    else:
        evidence_status = "no_evidence_of_family_outperformance"

    # ── Persist every frozen-test return series (control 3) ────────────────
    frame = pd.DataFrame({**series, BENCHMARK: bench_net})
    frame.index.name = "Date"
    series_path = ROOT / "data" / "gold" / "global_2004_q2_series.parquet"
    frame.reset_index().melt(
        id_vars="Date", var_name="candidate", value_name="net_return"
    ).to_parquet(series_path, index=False)

    artifact = {
        "provenance": {
            "universe": "global_2004",
            "base_currency": "USD",
            "hedge_status": "not applicable — single-currency universe",
            "currency_converted": False,
            "test_segment": {
                "start": str(bench_net.index.min().date()),
                "end": str(bench_net.index.max().date()),
                "n_days": int(len(bench_net)),
                "test_frac": float(p5["test_frac"]),
                "split_helper": "run_phase5.split_train_test (frozen, no custom dates)",
            },
            "git_revision": git_revision(ROOT),
            "generated_at": pd.Timestamp.now().isoformat(),
            "runtime_seconds": round(time.time() - started, 1),
            "source_artifacts": {
                rel: _sha256(ROOT / rel) for rel in (
                    "src/run_global_2004_q2.py", "src/global_universe.py",
                    "params_global_2004.yaml",
                    "docs/GLOBAL_UNIVERSE_PREREGISTRATION.md",
                    cfg["paths"]["gold_returns"], cfg["paths"]["gold_features"],
                    "data/gold/global_2004_q1_results.json",
                )
            },
        },
        "question": "Q2",
        "hypothesis": (
            f"H1: at least one candidate in the RF/XGBoost family outperforms "
            f"{BENCHMARK}, net of costs, on the frozen test segment. The null "
            "is COMPOSITE: no candidate in the searched family outperforms."
        ),
        "candidate_ledger": {
            "expected_reachable_count": expected_n,
            "executed_count": len(series),
            "derivation": (
                "(|rf_grid| + |xgb_grid|) x |shrink_grid| x |penalty_grid|, "
                "computed from frozen configuration BEFORE any return was "
                "calculated, and re-checked against the executed ledger."
            ),
            "deduplication": (
                "NONE. No candidate was removed on the basis of observed "
                "performance; dropping a poor performer after seeing it is the "
                "selection the correction exists to price."
            ),
            "labels_unique": True,
            "series_artifact": "data/gold/global_2004_q2_series.parquet",
        },
        "benchmark": {
            **bench_desc,
            "reproduces_q1_exactly": True,
            "q1_artifact": "data/gold/global_2004_q1_results.json",
        },
        "deployable_challengers": deployable,
        "family_tests": tests,
        "verdict": {
            "concordant_evidence_of_family_outperformance": concordant,
            "evidence_status": evidence_status,
            "primary_endpoint": "net-Sharpe differential",
            "secondary_endpoint": "annualized mean-return differential",
            "alpha": alpha,
            "rule": (
                "concordant = RC rejects AND SPA rejects, on the PRIMARY "
                "Sharpe endpoint only. Exactly one rejecting is reported as "
                "TEST-DEPENDENT EVIDENCE, not established outperformance. The "
                "secondary endpoint never promotes to primary."
            ),
            "no_statistic_shopping": (
                "All four cells (2 tests x 2 endpoints) are computed and "
                "persisted above. The verdict reads ONLY the primary-endpoint "
                "pair. The conclusion may not be taken from whichever cell "
                "yields the smallest p-value."
            ),
        },
        "fallback_telemetry": {
            "benchmark": _telemetry(bench_result),
            "candidates": telemetry,
            "total_candidate_fallbacks": sum(
                t["fallback_count"] for t in telemetry.values()
            ),
        },
        "limitations": [
            "ATTRIBUTION. Q2 changes TWO things at once relative to the "
            "released universes: the asset cross-section AND the macro-feature "
            "policy (the Bank Al-Maghrib block is excluded, §3.4). The "
            "challengers consume macro features, so their results here may NOT "
            "be attributed to the wider universe alone. Q1's comparison is "
            "unaffected, because neither of its strategies reads a macro column.",
            "RESIDUAL OUTER SELECTION. White RC and Hansen SPA correct the "
            "STRATEGY search. Nothing corrects the outer decision to CONSTRUCT "
            "global_2004 after measuring that the two released universes were "
            "each defective. Any positive result must be reported with that "
            "residual multiplicity stated.",
            "USD numéraire, single-currency: not directly comparable to "
            "full_2021, which is expressed in MAD.",
            "One frozen split. No nested walk-forward is run here; §12G showed "
            "point orderings on a single split can be unstable to evaluation "
            "design.",
        ],
    }

    out_path = ROOT / "data" / "gold" / "global_2004_q2_results.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log.info("")
    log.info("  ledger            : %d reachable = %d executed", expected_n, len(series))
    log.info("  benchmark         : %s net Sharpe %.4f (reproduces Q1)",
             BENCHMARK, bench_desc["net_sharpe"])
    for name, t in tests.items():
        log.info("  %-22s RC p=%.4f (%s)  SPA p=%.4f (%s)  beating=%d",
                 name, t["reality_check_p_value"],
                 "reject" if t["rc_rejects_at_alpha"] else "no",
                 t["spa_p_value"], "reject" if t["spa_rejects_at_alpha"] else "no",
                 t["n_candidates_beating_benchmark"])
    log.info("")
    log.info("  concordant_evidence_of_family_outperformance : %s", concordant)
    log.info("  evidence_status                              : %s", evidence_status)
    log.info("")
    log.info("  Q2 complete -> %s", out_path)
    return artifact


if __name__ == "__main__":
    run()
