"""
run_reality_check.py — multiple-testing correction over the WHOLE search.

Addresses: P4 — the project's last stated statistical limitation. Every
comparison reported so far tests one candidate against a benchmark, but that
candidate was CHOSEN from a search, and the best of many candidates beats a
benchmark by chance more often than any single candidate does. Phase 5 recorded
the multiple-testing position as `not_established` because a correct Reality
Check needs the frozen-test return series of EVERY searched configuration and
only the winners had one. This runner produces them.

THE CANDIDATE SET IS THE REACHABLE SPACE, NOT THE LEDGER COUNT.

The search was hierarchical: ML hyperparameters chosen by cross-validated
information coefficient, then portfolio levers chosen by validation Sharpe
*conditional on* those hyperparameters. The DSR ledger recorded 51 trials, but
the space the search could have selected from is every combination:

    (6 RF + 9 XGB hyperparameter configs) x (4 shrink x 4 penalty) = 240

Correcting for 51 when 240 were reachable would understate the multiplicity.
The larger number is the honest one.

WHY THIS IS AFFORDABLE. The content-addressed prediction cache is keyed on the
model configuration and NOT on the levers, so the 16 lever variants of one
hyperparameter config share a single set of fitted models. Measured: 128s for
the first variant, 0.3-0.7s for each of the remaining 15. The loop is therefore
ordered hyperparameters-outer, levers-inner — reversing it would refit every
model 16 times.

EVALUATION IS `evaluate_on_test`, unchanged from Phase 5, so each candidate's
series is directly comparable to the published Phase 5 test Sharpes rather than
to a cheaper approximation of them.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

import metrics
from backtest import build_cost_vector
from run_phase5 import evaluate_on_test
from strategies import RandomForestSignalStrategy, XGBoostSignalStrategy

log = logging.getLogger("run_reality_check")

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
SERIES_PATH = GOLD / "reality_check_series.parquet"
RESULTS_PATH = GOLD / "reality_check_results.json"

UNIVERSES = {
    "full_2021": ("log_returns.parquet", "ml_features_full.parquet"),
    "etf_2017": ("log_returns_etf.parquet", "ml_features_etf.parquet"),
}

# Both benchmarks Phase 5 reported against. `regime_conditional` is the hurdle
# the ML layer was built to beat; `equal_weight` is the honest naive floor.
# The PRIMARY benchmark is pre-specified, not chosen after seeing the results.
# `regime_conditional` is the hurdle the ML layer was built to beat and the one
# every prior phase reported against. `equal_weight` is a naive floor that the
# regime system and even the dividend-corrected classical `max_sharpe` already
# clear, so a candidate beating it is not evidence the ML layer adds value —
# those comparisons are EXPLORATORY and labelled as such in the artifact.
PRIMARY_BENCHMARK = "regime_conditional"
EXPLORATORY_BENCHMARKS = ("equal_weight",)
BENCHMARKS = (PRIMARY_BENCHMARK, *EXPLORATORY_BENCHMARKS)

MODEL_CLASSES = {
    "random_forest": RandomForestSignalStrategy,
    "xgboost": XGBoostSignalStrategy,
}


def _grid(spec: dict) -> list[dict]:
    """Every point of a hyperparameter grid, as dicts."""
    keys = list(spec)
    return [dict(zip(keys, values)) for values in itertools.product(*(spec[k] for k in keys))]


def candidate_configs(params: dict) -> list[dict]:
    """The full reachable search space, hyperparameters outer for cache reuse."""
    phase5 = params["phase5"]
    configs = []
    for model_type, grid_key in (("random_forest", "rf_grid"), ("xgboost", "xgb_grid")):
        for ml_params in _grid(phase5[grid_key]):
            for shrink, penalty in itertools.product(
                phase5["shrink_grid"], phase5["penalty_grid"]
            ):
                configs.append({
                    "model_type": model_type,
                    "ml_params": ml_params,
                    "shrinkage_weight": float(shrink),
                    "turnover_penalty": float(penalty),
                })
    return configs


def _label(config: dict) -> str:
    ml = "_".join(f"{k}={v}" for k, v in sorted(config["ml_params"].items()))
    return (f"{config['model_type']}__{ml}__shrink={config['shrinkage_weight']}"
            f"__pen={config['turnover_penalty']}")


def evaluate_universe(universe: str, params: dict) -> tuple[pd.DataFrame, dict]:
    """Every candidate and both benchmarks on the SAME frozen test dates."""
    returns_file, features_file = UNIVERSES[universe]
    returns = pd.read_parquet(GOLD / returns_file)
    features = pd.read_parquet(GOLD / features_file)

    bt, sig = params["backtest"], params["ml_signals"]
    phase5_results = json.loads((GOLD / "phase5_results.json").read_text())[universe]
    test_start = pd.Timestamp(phase5_results["test_start"])

    costs = build_cost_vector(
        returns.columns, bt["costs_bps"]["etf"], bt["costs_bps"]["bvc"]
    )
    backtest_params = {
        "rebalance_freq": bt["rebalance_freq"],
        "min_train_days": bt["min_train_days"],
        "universe_name": universe,
    }
    common = dict(
        max_weight=bt["max_weight"],
        risk_free_annual=bt["risk_free_annual"],
        min_train_rows=sig["min_train_rows"],
        short_window=sig["short_window"],
        long_window=sig["long_window"],
        momentum_windows=sig["momentum_windows"],
        condition_on_regime=sig["condition_on_regime"],
    )

    series: dict[str, pd.Series] = {}
    configs = candidate_configs(params)
    started = time.time()
    for i, config in enumerate(configs, start=1):
        strategy = MODEL_CLASSES[config["model_type"]](
            name="reality_check",
            model_params=config["ml_params"],
            mu_transform="shrink",
            shrinkage_weight=config["shrinkage_weight"],
            turnover_penalty=config["turnover_penalty"],
            **common,
        )
        net = evaluate_on_test(
            returns, features, strategy, test_start,
            backtest_params, costs, bt["max_weight"],
        )
        series[_label(config)] = net
        if i % 16 == 0 or i == len(configs):
            log.info("%s: %d/%d candidates (%.0fs elapsed)",
                     universe, i, len(configs), time.time() - started)

    # Benchmarks, same dates, same evaluator.
    from run_phase5 import build_phase4_strategies

    baselines = {s.name: s for s in build_phase4_strategies(params)}
    benchmark_series = {}
    for name in BENCHMARKS:
        benchmark_series[name] = evaluate_on_test(
            returns, features, baselines[name], test_start,
            backtest_params, costs, bt["max_weight"],
        )

    frame = pd.DataFrame({**series, **benchmark_series})
    # Name the index before it reaches `reset_index`: the melt below keys on
    # "Date", and an unnamed DatetimeIndex becomes a column called "index".
    # That failure surfaced only AFTER 18 minutes of evaluation, so the cheap
    # guard is worth more than it looks.
    frame.index.name = "Date"
    meta = {
        "test_start": str(frame.index.min().date()),
        "test_end": str(frame.index.max().date()),
        "n_test_days": int(len(frame)),
        "n_candidates": len(series),
        "runtime_seconds": round(time.time() - started, 1),
    }
    return frame, meta


def _tag(benchmark: str, statistic: str) -> dict:
    """Status of one outer comparison, fixed in advance of any result.

    Addresses: P4 — eight outer comparisons are run (2 universes x 2 benchmarks
    x 2 statistics). Quoting whichever crosses 0.05 would be multiplicity one
    level above the one this module corrects. The primary/exploratory split is
    therefore declared here, in code, rather than chosen once the p-values are
    visible.
    """
    primary = benchmark == PRIMARY_BENCHMARK
    return {
        "benchmark": benchmark,
        "status": "primary" if primary else "exploratory",
        "status_reason": (
            "Pre-specified primary benchmark: the hurdle the ML layer was built to "
            "beat, reported against by every prior phase."
            if primary else
            "Exploratory. The regime system and the dividend-corrected classical "
            "max_sharpe already clear this floor, so a candidate beating it is not "
            "evidence that the ML layer adds value."
        ),
    }


def analyse(frame: pd.DataFrame, universe: str, meta: dict, params: dict) -> dict:
    """White RC and Hansen SPA against each benchmark, on both statistics."""
    boot = params["phase5"]["bootstrap"]
    risk_free = params["backtest"]["risk_free_annual"]
    candidates = {c: frame[c] for c in frame.columns if c not in BENCHMARKS}

    tests = {}
    for benchmark in BENCHMARKS:
        for statistic in ("mean_return", "sharpe"):
            tests[f"{benchmark}__{statistic}"] = _tag(benchmark, statistic) | metrics.reality_check(
                candidates, frame[benchmark],
                statistic=statistic,
                block_len=boot["block_len"],
                n_boot=boot["n_boot"],
                seed=boot["seed"],
                risk_free_annual=risk_free,
            )
    return {"universe": universe, **meta, "tests": tests}


def _payload(results: dict) -> dict:
    """Artifact metadata, shared by the full run and the analysis-only path."""
    return {
        "method": "White (2000) Reality Check and Hansen (2005) SPA",
        "null_hypothesis": (
            "No candidate in the searched set outperforms the benchmark on the "
            "frozen test window."
        ),
        "candidate_set_note": (
            "The candidate set is the REACHABLE search space — every combination of "
            "ML hyperparameters and portfolio levers the hierarchical search could "
            "have selected — not the 51 trials the DSR ledger happened to record. "
            "Correcting for the smaller number would understate the multiplicity."
        ),
        "statistic_note": (
            "'mean_return' is the textbook White formulation, whose recentring "
            "argument assumes the performance measure is a mean. 'sharpe' matches "
            "this project's headline metric but is a ratio of moments, so its "
            "asymptotic justification is weaker. Both are reported; neither alone."
        ),
        "cross_correlation_note": (
            "All candidates share the same bootstrap block draws. Configurations "
            "differing by one hyperparameter on one data window are nearly the same "
            "strategy repeated, and resampling them independently would treat them "
            "as independent bets and overstate the effective breadth of the search."
        ),
        "outer_multiplicity_note": (
            "Eight outer comparisons are reported (2 universes x 2 benchmarks x 2 "
            "statistics). Quoting whichever crosses 0.05 would be multiplicity one "
            "level ABOVE the one this artifact corrects. The primary benchmark is "
            f"pre-specified as {PRIMARY_BENCHMARK!r} in code; every other comparison "
            "is labelled exploratory and must not be read as an ML-success claim."
        ),
        "rc_vs_spa_note": (
            "RC and SPA are reported side by side and neither is to be quoted alone. "
            "SPA is the more powerful by construction — it studentizes each candidate "
            "and drops hopeless ones from the null recentring — so it will usually "
            "give the smaller p-value. Read `spa_candidates_retained` before "
            "attributing the gap to the trimming rule: when nearly all candidates are "
            "retained, the divergence is studentization alone."
        ),
        "universes": results,
    }


def analyse_from_persisted(universes: list[str] | None = None) -> Path:
    """Recompute the tests from the stored series, without re-evaluating.

    Addresses: P4 — evaluation costs ~3.4 hours on `etf_2017`; the tests
    themselves take seconds. Separating them means a change to how results are
    REPORTED (a benchmark designation, a threshold, an added statistic) does
    not require re-running the backtests that produced the series, and cannot
    silently perturb them either. The candidate series are the expensive,
    immutable evidence; the analysis is a cheap, revisable reading of it.
    """
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    if not SERIES_PATH.is_file():
        raise FileNotFoundError(
            f"{SERIES_PATH.relative_to(ROOT)} is missing — run the full evaluation "
            f"first (`python src/run_reality_check.py`)."
        )
    long = pd.read_parquet(SERIES_PATH)
    previous = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.is_file() else {}
    results = dict(previous.get("universes", {}))

    for universe in (universes or sorted(long["universe"].unique())):
        block = long[long["universe"] == universe]
        frame = block.pivot(index="Date", columns="candidate", values="net_return")
        frame.index.name = "Date"
        meta = {
            "test_start": str(frame.index.min().date()),
            "test_end": str(frame.index.max().date()),
            "n_test_days": int(len(frame)),
            "n_candidates": int(sum(c not in BENCHMARKS for c in frame.columns)),
            "runtime_seconds": (results.get(universe) or {}).get("runtime_seconds"),
        }
        results[universe] = analyse(frame, universe, meta, params)

    RESULTS_PATH.write_text(
        json.dumps(_payload(results), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return RESULTS_PATH


def run(universes: list[str] | None = None) -> tuple[Path, Path]:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    selected = universes or list(UNIVERSES)

    existing = (
        pd.read_parquet(SERIES_PATH) if SERIES_PATH.is_file() else pd.DataFrame()
    )
    previous = (
        json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.is_file() else {"universes": {}}
    )

    long_frames = [] if existing.empty else [existing[existing["universe"] != u]
                                             for u in selected][:1]
    results = dict(previous.get("universes", {}))

    for universe in selected:
        frame, meta = evaluate_universe(universe, params)
        long = frame.reset_index().melt(id_vars="Date", var_name="candidate", value_name="net_return")
        long.insert(0, "universe", universe)
        long_frames.append(long)
        results[universe] = analyse(frame, universe, meta, params)
        for key, test in results[universe]["tests"].items():
            log.info("%s / %s: RC p=%.4f  SPA p=%.4f  (best %s)",
                     universe, key, test["reality_check_p_value"],
                     test["spa_p_value"], test["best_candidate"][:48])

    combined = pd.concat(long_frames, ignore_index=True)
    SERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SERIES_PATH, index=False)

    payload = _payload(results)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                            encoding="utf-8")
    return SERIES_PATH, RESULTS_PATH


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", action="append", choices=list(UNIVERSES),
                        help="Restrict to one universe (repeatable).")
    args = parser.parse_args()
    for path in run(args.universe):
        print(path.relative_to(ROOT))
