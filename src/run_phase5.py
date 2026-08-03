"""
run_phase5.py — Phase 5 entry point: out-of-sample evaluation with honest,
leak-free hyperparameter selection. MLflow-tracked.

WHAT THIS RESOLVES. Phase 4C ended on a deliberate cliffhanger: on
`full_2021`, `rf_signal_shrunk` reached net 1.117 vs. the 1.1215 hurdle — but
with `shrinkage_weight` and `turnover_penalty` CHOSEN, not tuned. Phase 4C
also proved (rf-vs-xgb asymmetry) that a single global penalty is wrong.
Selecting these knobs by eyeballing the test set would be the exact P4
backtest overfitting the project exists to avoid. Phase 5 makes the selection
legitimate and reports the verdict on data the selection never saw.

THE HONEST-SELECTION ARCHITECTURE (a strict time-ordered split):

    |<-------------- train + validation -------------->|<--- frozen test --->|
    |  purged-CV selects ML hyperparameters (by IC)     |                     |
    |  validation OOS tail selects the portfolio levers |   the verdict is    |
    |            (by net Sharpe, per model)             |  measured only here |

  - ML prediction hyperparameters → PurgedKFold on the train+val panel, scored
    by information coefficient (`model_selection.select_ml_hyperparameters`).
  - Portfolio levers (shrinkage_weight, per-model turnover_penalty) → net
    Sharpe of a real walk-forward over the train+val OOS tail
    (`model_selection.select_portfolio_levers`).
  - FREEZE, then evaluate the tuned strategy ONLY on the held-out test
    segment, alongside the Phase 4 hurdle (`regime_conditional`) and
    `equal_weight` re-evaluated on the SAME test dates — an apples-to-apples
    verdict, not a full-window number vs. a differently-windowed one.

Addresses: P4 — every headline number carries a block-bootstrap Sharpe CI, and
the DSR is deflated by the WHOLE search (every validation grid point is in the
accumulated trial ledger), not just the final comparison.

Usage:
    python src/run_phase5.py
"""

import json
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

from backtest import build_cost_vector, run_backtest
from purged_kfold import PurgedWalkForwardSplit
from metrics import (
    DSRTrialLedger,
    annualized_sharpe,
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    max_drawdown,
    paired_block_bootstrap,
)
from model_selection import (
    build_fold_audit,
    select_ml_hyperparameters,
    select_portfolio_levers,
)
from run_phase4 import build_strategies as build_phase4_strategies
from run_phase4 import load_features, load_universe
from strategies import RandomForestSignalStrategy, XGBoostSignalStrategy
from utils import configure_mlflow, load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_phase5")

ROOT = Path(__file__).resolve().parents[1]

# (model_type, Strategy class, display name) for the two F7 families tuned here.
MODELS = [
    ("random_forest", RandomForestSignalStrategy, "rf_signal_tuned"),
    ("xgboost", XGBoostSignalStrategy, "xgb_signal_tuned"),
]


def split_train_test(returns: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Time-ordered split: the final `test_frac` of rows is the frozen test set.

    Returns the train+validation frame and the first test date. The test
    segment is never passed to any selector — only to the final evaluation.
    """
    split_pos = int(round(len(returns) * (1.0 - test_frac)))
    split_pos = max(1, min(split_pos, len(returns) - 1))
    test_start = returns.index[split_pos]
    train_val = returns.iloc[:split_pos]
    return train_val, test_start


def evaluate_on_test(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    strategy,
    test_start: pd.Timestamp,
    backtest_params: dict,
    cost_vector: pd.Series,
    max_weight: float,
) -> pd.Series:
    """
    Walk-forward over the FULL series, return net returns on the test segment.

    Training at each test-segment rebalance legitimately uses all data up to
    that date (including train+val) — that is allowed, because the
    HYPERPARAMETERS were selected without ever seeing the test segment. Only
    the test-dated net returns are returned, so the reported Sharpe is a pure
    out-of-sample-of-selection number.
    """
    result = run_backtest(
        returns, strategy,
        rebalance_freq=backtest_params["rebalance_freq"],
        min_train_days=backtest_params["min_train_days"],
        cost_bps=cost_vector,
        extras={"features": features},
        universe_name=backtest_params.get("universe_name", ""),
        max_weight=max_weight,
    )
    return result.net_returns.loc[result.net_returns.index >= test_start]


def run_phase5() -> dict:
    """Execute Phase 5 end to end; write and return `phase5_results.json`."""
    params = load_params()
    bp = params["backtest"]
    sp = params["ml_signals"]
    p5 = params["phase5"]
    cv = params["purged_cv"]
    wf = params["walk_forward_cv"]
    boot = p5["bootstrap"]
    max_weight = bp["max_weight"]
    rf = bp["risk_free_annual"]

    regime_kwargs = dict(
        n_states=params["regime"]["n_states"],
        n_restarts=params["regime"]["n_restarts"],
        random_state_base=params["regime"]["random_state_base"],
        covariance_type=params["regime"]["covariance_type"],
        min_regime_train_days=params["regime"]["min_regime_train_days"],
    )

    results_path = ROOT / p5["results_path"]
    ledger_path = ROOT / p5["ledger_path"]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.unlink(missing_ok=True)          # fresh search each run
    ledger = DSRTrialLedger(path=ledger_path)

    hurdle_path = ROOT / "data" / "gold" / "phase4_results.json"
    stored_hurdle = json.loads(hurdle_path.read_text()) if hurdle_path.exists() else {}

    configure_mlflow()
    mlflow.set_experiment("phase5_oos_evaluation")
    output: dict[str, dict] = {}
    fold_audit: dict[str, dict] = {}
    paired_rows: list[dict] = []

    with mlflow.start_run(run_name="phase5_oos_evaluation"):
        mlflow.log_params({
            "test_frac": p5["test_frac"],
            "cv_n_splits": cv["n_splits"],
            "cv_embargo_frac": cv["embargo_frac"],
            "bootstrap_n": boot["n_boot"],
        })

        for universe_name, path in bp["universes"].items():
            returns = load_universe(path)
            features = load_features(params["ml_features"]["outputs"][universe_name])
            cost_vector = build_cost_vector(
                returns.columns,
                etf_cost_bps=bp["costs_bps"]["etf"], bvc_cost_bps=bp["costs_bps"]["bvc"],
            )
            train_val, test_start = split_train_test(returns, p5["test_frac"])
            tv_features = features.loc[features.index <= train_val.index[-1]]
            log.info(
                "=== %s: %d rows; train+val %s→%s, frozen test %s→%s ===",
                universe_name, len(returns),
                train_val.index.min().date(), train_val.index.max().date(),
                test_start.date(), returns.index.max().date(),
            )

            backtest_kwargs = {
                "rebalance_freq": bp["rebalance_freq"],
                "min_train_days": bp["min_train_days"],
                "cost_bps": cost_vector,
                "max_weight": max_weight,
                "risk_free_annual": rf,
                "min_train_rows": sp["min_train_rows"],
                "condition_on_regime": sp["condition_on_regime"],
                "short_window": sp["short_window"],
                "long_window": sp["long_window"],
                "momentum_windows": sp["momentum_windows"],
                "universe_name": universe_name,
            }

            tuned_entries = {}
            tuned_series = {}          # display -> test net-return series (reused for DSR)
            for model_type, cls, display in MODELS:
                grid = p5["rf_grid"] if model_type == "random_forest" else p5["xgb_grid"]
                # Forward-only folds: training never post-dates validation.
                splitter = PurgedWalkForwardSplit(
                    min_train_dates=wf["min_train_dates"], val_dates=wf["val_dates"],
                    n_splits=wf["n_splits"], embargo_dates=wf["embargo_dates"],
                    label_horizon=wf["label_horizon"], mode=wf["mode"],
                    step_dates=wf["step_dates"],
                )
                ml_params, cv_table = select_ml_hyperparameters(
                    train_val, tv_features, grid, model_type=model_type,
                    short_window=sp["short_window"], long_window=sp["long_window"],
                    momentum_windows=sp["momentum_windows"],
                    condition_on_regime=sp["condition_on_regime"], regime_kwargs=regime_kwargs,
                    splitter=splitter,
                )
                # Every hyperparameter configuration is part of the search and
                # must count toward N. Schema 1 omitted these entirely, which
                # understated the search in the project's own favour.
                for _, row in cv_table.iterrows():
                    cfg = {k: row[k] for k in grid}
                    ledger.record(
                        universe_name, None,
                        label=f"{model_type}__" + "_".join(f"{k}={v}" for k, v in cfg.items()),
                        kind="ml_grid", params=cfg, score=float(row["mean_ic"]),
                    )
                if universe_name not in fold_audit:
                    fold_audit[universe_name] = {}
                fold_audit[universe_name][model_type] = {
                    "folds": build_fold_audit(
                        train_val, tv_features, splitter,
                        short_window=sp["short_window"], long_window=sp["long_window"],
                        momentum_windows=sp["momentum_windows"],
                        condition_on_regime=sp["condition_on_regime"],
                        regime_kwargs=regime_kwargs,
                    ),
                    "selected_hyperparameters": ml_params,
                    "fold_ics_of_selected": list(cv_table.iloc[0]["fold_ics"]),
                    "mean_ic_of_selected": round(float(cv_table.iloc[0]["mean_ic"]), 6),
                }
                levers, lever_table = select_portfolio_levers(
                    train_val, tv_features, model_type=model_type, ml_params=ml_params,
                    shrink_grid=p5["shrink_grid"], penalty_grid=p5["penalty_grid"],
                    backtest_kwargs=backtest_kwargs, ledger=ledger, universe=universe_name,
                )

                # FREEZE and evaluate on the held-out test segment.
                tuned = cls(
                    name=display, mu_transform="shrink",
                    shrinkage_weight=levers["shrinkage_weight"],
                    turnover_penalty=levers["turnover_penalty"],
                    model_params=ml_params, max_weight=max_weight, risk_free_annual=rf,
                    min_train_rows=sp["min_train_rows"],
                    short_window=sp["short_window"], long_window=sp["long_window"],
                    momentum_windows=sp["momentum_windows"],
                    condition_on_regime=sp["condition_on_regime"],
                    n_states=params["regime"]["n_states"],
                    n_restarts=params["regime"]["n_restarts"],
                    random_state_base=params["regime"]["random_state_base"],
                    covariance_type=params["regime"]["covariance_type"],
                    min_regime_train_days=params["regime"]["min_regime_train_days"],
                )
                test_net = evaluate_on_test(
                    returns, features, tuned, test_start, backtest_kwargs, cost_vector, max_weight,
                )
                ledger.record(universe_name, test_net)
                tuned_series[display] = test_net
                point, lo, hi = block_bootstrap_sharpe_ci(
                    test_net, block_len=boot["block_len"], n_boot=boot["n_boot"],
                    alpha=boot["alpha"], risk_free_annual=rf, seed=boot["seed"],
                )
                tuned_entries[display] = {
                    "selected_ml_params": ml_params,
                    "selected_levers": levers,
                    "best_cv_ic": round(float(cv_table.iloc[0]["mean_ic"]), 4),
                    "test_sharpe_net": round(point, 4),
                    "test_sharpe_ci": [round(lo, 4), round(hi, 4)],
                    "test_max_drawdown": round(max_drawdown(test_net), 4),
                }
                log.info("%s / %s: test Sharpe %.4f (90%% CI %.3f..%.3f), levers %s",
                         display, universe_name, point, lo, hi, levers)

            # Baselines (regime_conditional hurdle + equal_weight) on the SAME test dates.
            baselines = {s.name: s for s in build_phase4_strategies(params)}
            baseline_entries = {}
            baseline_series = {}
            for name in ("regime_conditional", "equal_weight"):
                test_net = evaluate_on_test(
                    returns, features, baselines[name], test_start,
                    backtest_kwargs, cost_vector, max_weight,
                )
                ledger.record(universe_name, test_net)
                point, lo, hi = block_bootstrap_sharpe_ci(
                    test_net, block_len=boot["block_len"], n_boot=boot["n_boot"],
                    alpha=boot["alpha"], risk_free_annual=rf, seed=boot["seed"],
                )
                baseline_entries[name] = {
                    "test_sharpe_net": round(point, 4),
                    "test_sharpe_ci": [round(lo, 4), round(hi, 4)],
                }
                baseline_series[name] = test_net

            # ── Paired comparisons: the evidence a superiority claim needs ──
            # Marginal CIs above describe each strategy's own uncertainty; they
            # do not test whether two strategies differ. These do, on identical
            # dates, on NET returns, with a null-centred p-value.
            for cand_name, cand_series in tuned_series.items():
                for bench_name, bench_series in baseline_series.items():
                    cmp_ = paired_block_bootstrap(
                        cand_series, bench_series,
                        block_len=boot["block_len"], n_boot=boot["n_boot"],
                        alpha=boot["alpha"], risk_free_annual=rf, seed=boot["seed"],
                    )
                    ci_lo, ci_hi = cmp_["sharpe_diff_ci"]
                    excludes_zero = ci_lo > 0.0 or ci_hi < 0.0
                    cmp_.update({
                        "universe": universe_name,
                        "candidate": cand_name,
                        "benchmark": bench_name,
                        "interpretation": _interpret(cmp_, excludes_zero),
                    })
                    paired_rows.append(cmp_)

            # Verdict: best tuned F7 vs the re-evaluated hurdle, on the test window.
            best_tuned_name = max(tuned_entries, key=lambda k: tuned_entries[k]["test_sharpe_net"])
            best_tuned = tuned_entries[best_tuned_name]["test_sharpe_net"]
            hurdle_test = baseline_entries["regime_conditional"]["test_sharpe_net"]
            pool = ledger.pool(universe_name)
            # DSR of the best tuned strategy's test returns vs. the WHOLE search
            # pool (every validation grid trial + the final test evaluations) —
            # reuses the series already computed above, no extra backtest.
            best_series_dsr = (
                deflated_sharpe_ratio(tuned_series[best_tuned_name], pool)
                if len(pool) >= 2 else float("nan")
            )

            output[universe_name] = {
                "train_val_start": str(train_val.index.min().date()),
                "train_val_end": str(train_val.index.max().date()),
                "test_start": str(test_start.date()),
                "test_end": str(returns.index.max().date()),
                "test_frac": p5["test_frac"],
                "n_search_trials": ledger.n_trials(universe_name),
                "tuned": tuned_entries,
                "baselines": baseline_entries,
                "best_tuned": best_tuned_name,
                "hurdle_on_test": {"strategy": "regime_conditional", "test_sharpe_net": hurdle_test},
                "beats_hurdle_on_test": bool(best_tuned > hurdle_test),
                "best_tuned_dsr_vs_search": round(best_series_dsr, 4)
                if best_series_dsr == best_series_dsr else None,
                "stored_full_window_hurdle": (
                    stored_hurdle.get(universe_name, {}).get("sharpe_net")
                ),
                "search_ledger": ledger.summary(universe_name),
                "validation_protocol": "purged_walk_forward_expanding",
            }
            log.info(
                "%s VERDICT: best tuned %s @ %.4f vs hurdle %.4f on test → %s",
                universe_name, best_tuned_name, best_tuned, hurdle_test,
                "BEATS" if best_tuned > hurdle_test else "does NOT beat",
            )

        ledger.save()

        # ── Artifact: the realised fold geometry ──────────────────────────────
        protocol_path = ROOT / "data" / "gold" / "phase5_validation_protocol.json"
        protocol_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(json.dumps({
            "protocol": "purged_walk_forward",
            "description": (
                "Forward-only model selection. Every fold satisfies "
                "train_end < embargo_start <= val_start <= val_end, and every "
                "validation window lies strictly inside the train+validation "
                "segment, so the frozen test segment is untouched by selection."
            ),
            "config": dict(wf),
            "universes": fold_audit,
        }, indent=2))
        mlflow.log_artifact(str(protocol_path))
        log.info("Validation protocol written → %s", protocol_path)

        # ── Artifact: paired comparisons ──────────────────────────────────────
        paired_path = ROOT / "data" / "gold" / "paired_comparison_results.json"
        paired_path.write_text(json.dumps({
            "method": (
                "Paired moving-block bootstrap on identical frozen-test dates, "
                "net of transaction costs. Both series are resampled with the SAME "
                "block indices, preserving serial dependence within each strategy "
                "and same-day correlation between them."
            ),
            "p_value_definition": (
                "One-sided, H0: no outperformance. The resampled differences are "
                "recentred on zero to simulate the null; the p-value is the share of "
                "that null distribution at or beyond the OBSERVED difference. It is "
                "NOT the fraction of draws above zero — that quantity is reported "
                "separately as prob_sharpe_diff_positive."
            ),
            "multiple_testing": _search_correction_note(ledger, bp["universes"]),
            "comparisons": paired_rows,
        }, indent=2))
        mlflow.log_artifact(str(paired_path))
        log.info("Paired comparisons written → %s (%d rows)", paired_path, len(paired_rows))

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(output, indent=2))
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(ledger_path))
        log.info("Phase 5 results → %s", results_path)
        log.info("=== Phase 5 complete. `mlflow ui` to inspect. ===")

    return output


def _interpret(cmp_: dict, excludes_zero: bool) -> str:
    """Plain-language reading of one paired comparison.

    Addresses: P4 — the wording rule of this project made executable. A
    non-significant difference is NOT evidence of equivalence (that needs an
    equivalence test against a pre-specified margin, which this project has
    not run), and a significant one is not licensed unless the paired interval
    actually excludes zero. Generating the sentence from the numbers stops a
    human writing a stronger claim than the artifact supports.
    """
    d = cmp_["sharpe_diff"]
    p = cmp_["p_value_no_outperformance"]
    direction = "higher" if d > 0 else "lower"
    economic = (
        f"observed net-Sharpe difference {d:+.3f} "
        f"({cmp_['ann_return_diff']:+.2%} annualised return)"
    )
    if excludes_zero and p < 0.05:
        return (
            f"Candidate is {direction} by {economic}; the paired 90% interval excludes "
            f"zero and the null-centred p-value is {p:.3f}. This is evidence of a "
            "difference on this frozen test window, before any correction for the "
            "number of configurations searched."
        )
    if abs(d) < 0.05:
        return (
            f"No evidence of outperformance: {economic}, paired interval "
            f"{cmp_['sharpe_diff_ci']} spans zero (p = {p:.3f}). The difference is "
            "also economically small. This is NOT a finding of equivalence — no "
            "equivalence test against a pre-specified margin was run."
        )
    return (
        f"Economically {direction} by {economic}, but statistically inconclusive: "
        f"the paired interval {cmp_['sharpe_diff_ci']} spans zero (p = {p:.3f}). "
        "Uplift of this size is worth noting and is not established."
    )


def _search_correction_note(ledger, universes) -> dict:
    """State honestly what multiple-testing correction the search supports.

    Addresses: P4 — the audit this note reports was the point of enlarging the
    ledger. A White Reality Check or Hansen SPA bootstraps the MAXIMUM
    statistic across the candidate return series, so it needs a return series
    for every searched alternative, all on a common date index.

    Phase 5's search is heterogeneous and only partly satisfies that:

      * portfolio-lever trials are Sharpe-scored and DO carry a return series,
        but that series lives on the VALIDATION window, not the frozen test
        window the final claim is made on;
      * ML hyperparameter trials are IC-scored on validation folds and have no
        portfolio return series at all, because a hyperparameter configuration
        is not a portfolio.

    Running a Reality Check over only the subset that happens to have series,
    on a window other than the one being claimed, would produce a number that
    LOOKS like a correction and silently under-counts the search — the exact
    "superficial implementation" that is worse than none, because it converts
    an admitted gap into a false reassurance.

    So the correction is reported as NOT ESTABLISHED, with the concrete
    experiment that would establish it named rather than hand-waved. The
    Deflated Sharpe Ratio still applies and is reported, and now counts the ML
    grid it previously omitted.
    """
    return {
        "status": "not_established",
        "reason": (
            "A correct White Reality Check / Hansen SPA needs the frozen-test "
            "return series of EVERY searched candidate on a common index. Phase 5 "
            "evaluates the frozen test only for the finally-selected configurations; "
            "lever trials carry validation-window series, and ML-grid trials are "
            "IC-scored and carry none. A Reality Check over the available subset "
            "would under-count the search and read as a false reassurance."
        ),
        "what_would_establish_it": (
            "Re-run the frozen-test walk-forward for every searched configuration "
            "(not only the winner), storing each net-return series on the test index, "
            "then bootstrap the max statistic across them. That is a materially more "
            "expensive experiment and a change of methodology, so it is deferred "
            "rather than approximated."
        ),
        "what_is_reported_instead": (
            "Deflated Sharpe Ratio against the recorded trial pool, now including the "
            "ML hyperparameter grid that schema 1 omitted, plus per-comparison paired "
            "bootstrap p-values that are NOT corrected for multiplicity."
        ),
        "search_size_by_universe": {u: ledger.summary(u) for u in universes},
    }


if __name__ == "__main__":
    try:
        run_phase5()
    except Exception as exc:
        log.error("Phase 5 run failed: %s", exc, exc_info=True)
        sys.exit(1)
