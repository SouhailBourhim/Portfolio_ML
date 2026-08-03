"""profile_phase5.py — deterministic cost breakdown of the Phase 5 stage.

WHY. `phase5_compare` is the pipeline's long pole (~3 h), and the Phase 2
validity upgrade requires re-running it. Before optimising anything, measure
where the time actually goes — the Phase 4C lesson was that 80% of the cost
was duplicated work that no amount of intuition would have located.

PROFILING ONLY. This script changes no methodology: it does not alter fold
definitions, search grids, the frozen-test boundary, or any hyperparameter.
It times the real functions on the real data and counts what they call.

DECOMPOSITION. Phase 5 has four cost centres per (universe, model):

  1. CV fitting        select_ml_hyperparameters -> folds x grid points model fits
  2. Lever selection   select_portfolio_levers   -> |shrink| x |penalty| walk-forwards
  3. Frozen-test eval  evaluate_on_test          -> one walk-forward over the FULL series
  4. Bootstrap CIs     block_bootstrap_sharpe_ci -> n_boot resamples per strategy

Measured directly where affordable (full_2021 is small enough to run for
real), and extrapolated from a timed single call times an exact count where a
direct run would exceed the profiling budget (etf_2017's frozen-test
walk-forward alone is ~30 min). Extrapolated figures are timed at the WIDEST
training window, so they are upper bounds, and are labelled as estimates
rather than presented as measurements.

Usage:
    ./.venv/bin/python scripts/profile_phase5.py
    ./.venv/bin/python scripts/profile_phase5.py --universe full_2021 --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import _rebalance_schedule, build_cost_vector  # noqa: E402
from run_phase4 import load_features, load_universe  # noqa: E402
from run_phase5 import split_train_test  # noqa: E402
from utils import load_params  # noqa: E402

logging.basicConfig(level=logging.ERROR)

CALLS: dict[str, list[dict]] = defaultdict(list)


def _digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(p, index=True).to_numpy().tobytes())
            h.update("|".join(map(str, p.columns)).encode())
        elif isinstance(p, pd.Series):
            h.update(pd.util.hash_pandas_object(p, index=True).to_numpy().tobytes())
        else:
            h.update(repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def install_counters() -> None:
    """Count + time the primitives, and digest their inputs to expose repeats."""
    import dcc_garch
    import ml_signals
    import regime
    import strategies

    def wrap(module, name, key, digest_of):
        real = getattr(module, name)

        def counted(*a, **kw):
            t0 = time.perf_counter()
            out = real(*a, **kw)
            CALLS[key].append({
                "seconds": time.perf_counter() - t0,
                "digest": digest_of(a, kw),
            })
            return out

        setattr(module, name, counted)

    wrap(ml_signals, "fit_predict_expected_returns", "ml_signal_predict",
         lambda a, kw: _digest(a[0] if a else kw.get("train_returns"),
                               kw.get("model_type"),
                               tuple(sorted((kw.get("model_params") or {}).items()))))
    wrap(regime, "fit_hmm", "fit_hmm", lambda a, kw: _digest(a[0] if a else None))
    wrap(strategies, "estimate_covariance", "estimate_covariance",
         lambda a, kw: _digest(a[0] if a else kw.get("train_returns"), kw.get("estimator")))
    wrap(dcc_garch, "dcc_covariance", "dcc_covariance",
         lambda a, kw: _digest(a[0] if a else None))
    wrap(strategies, "_optimize_weights", "optimize_weights", lambda a, kw: "n/a")


def repeat_report(keys=None) -> dict:
    """Group recorded calls by input digest — repeats are duplicated work."""
    out = {}
    for key, calls in CALLS.items():
        if keys and key not in keys:
            continue
        groups: dict[str, list[float]] = defaultdict(list)
        for c in calls:
            groups[c["digest"]].append(c["seconds"])
        total = sum(sum(v) for v in groups.values())
        redundant = sum(sum(v[1:]) for v in groups.values())
        out[key] = {
            "calls": len(calls),
            "distinct_inputs": len(groups) if calls and calls[0]["digest"] != "n/a" else None,
            "total_seconds": round(total, 2),
            "seconds_in_repeat_calls": round(redundant, 2),
        }
    return out


def profile_universe(universe: str, params: dict, run_levers: bool) -> dict:
    bp, sp, p5, cv = params["backtest"], params["ml_signals"], params["phase5"], params["purged_cv"]
    returns = load_universe(bp["universes"][universe])
    features = load_features(params["ml_features"]["outputs"][universe])
    train_val, test_start = split_train_test(returns, p5["test_frac"])
    tv_features = features.loc[features.index <= train_val.index[-1]]
    cost_vector = build_cost_vector(
        returns.columns, etf_cost_bps=bp["costs_bps"]["etf"], bvc_cost_bps=bp["costs_bps"]["bvc"]
    )

    n_reb_tv = len(_rebalance_schedule(train_val.index, bp["rebalance_freq"], bp["min_train_days"]))
    n_reb_full = len(_rebalance_schedule(returns.index, bp["rebalance_freq"], bp["min_train_days"]))
    n_levers = len(p5["shrink_grid"]) * len(p5["penalty_grid"])
    grids = {"random_forest": p5["rf_grid"], "xgboost": p5["xgb_grid"]}

    def grid_size(g):
        n = 1
        for v in g.values():
            n *= len(v)
        return n

    report: dict = {
        "universe": universe,
        "rows": len(returns),
        "assets": returns.shape[1],
        "train_val_rows": len(train_val),
        "test_start": str(test_start.date()),
        "n_rebalances_train_val": n_reb_tv,
        "n_rebalances_full": n_reb_full,
        "n_lever_configs": n_levers,
        "grid_points": {m: grid_size(g) for m, g in grids.items()},
        "cv_folds": cv["n_splits"],
        "cost_centres": {},
    }

    regime_kwargs = dict(
        n_states=params["regime"]["n_states"], n_restarts=params["regime"]["n_restarts"],
        random_state_base=params["regime"]["random_state_base"],
        covariance_type=params["regime"]["covariance_type"],
        min_regime_train_days=params["regime"]["min_regime_train_days"],
    )

    # ── 1. CV fitting: time ONE grid point over all folds, scale by grid size ──
    from model_selection import select_ml_hyperparameters

    for model_type, grid in grids.items():
        one_point = {k: [v[0]] for k, v in grid.items()}
        CALLS.clear()
        t0 = time.perf_counter()
        select_ml_hyperparameters(
            train_val, tv_features, one_point, model_type=model_type,
            n_splits=cv["n_splits"], embargo_frac=cv["embargo_frac"],
            short_window=sp["short_window"], long_window=sp["long_window"],
            momentum_windows=sp["momentum_windows"],
            condition_on_regime=sp["condition_on_regime"], regime_kwargs=regime_kwargs,
        )
        per_point = time.perf_counter() - t0
        report["cost_centres"][f"cv_fitting__{model_type}"] = {
            "measured_seconds_for_one_grid_point": round(per_point, 2),
            "grid_points": grid_size(grid),
            "estimated_seconds": round(per_point * grid_size(grid), 1),
            "note": "one panel build is shared across grid points, so this over-estimates",
        }

    # ── 2. Lever selection: measured for real where affordable ────────────────
    if run_levers:
        from model_selection import select_portfolio_levers

        backtest_kwargs = {
            "rebalance_freq": bp["rebalance_freq"], "min_train_days": bp["min_train_days"],
            "cost_bps": cost_vector, "max_weight": bp["max_weight"],
            "risk_free_annual": bp["risk_free_annual"], "min_train_rows": sp["min_train_rows"],
            "condition_on_regime": sp["condition_on_regime"], "short_window": sp["short_window"],
            "long_window": sp["long_window"], "momentum_windows": sp["momentum_windows"],
            "universe_name": universe,
        }
        CALLS.clear()
        t0 = time.perf_counter()
        select_portfolio_levers(
            train_val, tv_features, model_type="random_forest",
            ml_params={k: v[0] for k, v in grids["random_forest"].items()},
            shrink_grid=p5["shrink_grid"], penalty_grid=p5["penalty_grid"],
            backtest_kwargs=backtest_kwargs,
        )
        elapsed = time.perf_counter() - t0
        report["cost_centres"]["lever_selection__random_forest"] = {
            "measured_seconds": round(elapsed, 1),
            "n_backtests": n_levers,
            "seconds_per_backtest": round(elapsed / n_levers, 2),
            "repeats": repeat_report(["ml_signal_predict", "estimate_covariance",
                                      "optimize_weights", "fit_hmm"]),
        }

    # ── 3. Frozen-test walk-forward: time one fit at the widest window ────────
    from strategies import RandomForestSignalStrategy, XGBoostSignalStrategy

    for model_type, cls in (("random_forest", RandomForestSignalStrategy),
                            ("xgboost", XGBoostSignalStrategy)):
        strategy = cls(
            max_weight=bp["max_weight"], risk_free_annual=bp["risk_free_annual"],
            model_params={k: v[0] for k, v in grids[model_type].items()},
            min_train_rows=sp["min_train_rows"], short_window=sp["short_window"],
            long_window=sp["long_window"], momentum_windows=sp["momentum_windows"],
            condition_on_regime=sp["condition_on_regime"],
            mu_transform="shrink", shrinkage_weight=0.5,
        )
        CALLS.clear()
        t0 = time.perf_counter()
        strategy.fit(returns, {"features": features})
        per_fit = time.perf_counter() - t0
        report["cost_centres"][f"frozen_test_eval__{model_type}"] = {
            "measured_seconds_per_fit_at_widest_window": round(per_fit, 2),
            "n_rebalances": n_reb_full,
            "estimated_seconds": round(per_fit * n_reb_full, 1),
            "note": "upper bound: every rebalance priced at the widest window",
        }

    # ── 4. Bootstrap ──────────────────────────────────────────────────────────
    from metrics import block_bootstrap_sharpe_ci

    test_len = len(returns.loc[returns.index >= test_start])
    synthetic = pd.Series(
        returns.iloc[-test_len:].mean(axis=1).to_numpy(),
        index=returns.index[-test_len:],
    )
    boot = p5["bootstrap"]
    t0 = time.perf_counter()
    block_bootstrap_sharpe_ci(
        synthetic, block_len=boot["block_len"], n_boot=boot["n_boot"],
        alpha=boot["alpha"], seed=boot["seed"],
    )
    one_boot = time.perf_counter() - t0
    report["cost_centres"]["bootstrap"] = {
        "measured_seconds_per_series": round(one_boot, 3),
        "series_per_universe": 4,
        "estimated_seconds": round(one_boot * 4, 1),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=["etf_2017", "full_2021"], default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--skip-levers", action="store_true",
                    help="skip the measured lever run (it is the slow part)")
    args = ap.parse_args()

    params = load_params()
    universes = [args.universe] if args.universe else list(params["backtest"]["universes"])
    install_counters()

    out = {"universes": []}
    for u in universes:
        # Lever selection is measured for real only on the small universe; on
        # etf_2017 it would exceed the profiling budget on its own.
        run_levers = (u == "full_2021") and not args.skip_levers
        print(f"\n=== {u} (levers measured: {run_levers}) ===", flush=True)
        rep = profile_universe(u, params, run_levers)
        out["universes"].append(rep)
        print(f"  {rep['rows']} rows, train+val {rep['train_val_rows']}, "
              f"{rep['n_rebalances_train_val']} tv / {rep['n_rebalances_full']} full rebalances")
        for name, c in rep["cost_centres"].items():
            secs = c.get("estimated_seconds", c.get("measured_seconds"))
            kind = "measured" if "measured_seconds" in c else "est."
            print(f"    {name:34s} {secs:>8.1f}s  ({kind})")
            for prim, r in (c.get("repeats") or {}).items():
                print(f"        {prim:<22} {r['calls']:>4} calls, "
                      f"{str(r['distinct_inputs']):>4} distinct, "
                      f"{r['seconds_in_repeat_calls']:>7.2f}s in repeats")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwritten → {args.json}")


if __name__ == "__main__":
    main()
