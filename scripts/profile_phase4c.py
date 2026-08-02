"""profile_phase4c.py — deterministic timing harness for the Phase 4C stage.

WHY THIS EXISTS. `dvc repro phase4c_compare` stopped completing: it ran for
over two hours without finishing on the current data. Guessing at the cause
would be exactly the wrong move, so this measures instead.

WHAT IT MEASURES, and why this decomposition is the honest one:

  The stage cost is, by construction,

      total  =  Σ_strategies  n_rebalances(universe) × cost_of_one_fit(strategy, τ)

  `n_rebalances` is a property of the engine's schedule and is counted
  exactly, not sampled. `cost_of_one_fit` is timed directly, at the LAST
  rebalance date of each universe — the worst case, because the training
  window is expanding, so this is an upper bound per call rather than an
  average. The product is therefore a conservative estimate of the whole
  stage, and it is reported as an estimate rather than passed off as a
  measured run.

  Separately, the harness COUNTS invocations of the three expensive
  primitives (`regime.fit_hmm`, the RandomForest/XGBoost fit inside
  `ml_signals.fit_predict_expected_returns`, and `dcc_garch.fit_dcc_garch`)
  and records the SHA-256 of each call's inputs. Identical digests across
  different strategies at the same τ are direct evidence of duplicated work,
  which no amount of per-strategy timing alone would prove.

Deterministic: no sampling, no randomness, no wall-clock dependence beyond
the timings themselves. Reads only committed Gold artifacts.

Usage:
    ./.venv/bin/python scripts/profile_phase4c.py            # both universes
    ./.venv/bin/python scripts/profile_phase4c.py --universe full_2021
    ./.venv/bin/python scripts/profile_phase4c.py --json out.json
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

from backtest import _rebalance_schedule  # noqa: E402
from run_phase4 import load_features, load_universe  # noqa: E402
from run_phase4c import build_strategies  # noqa: E402
from utils import load_params  # noqa: E402

logging.basicConfig(level=logging.ERROR)   # the strategies log WARNINGs we do not need here

CALLS: dict[str, list[dict]] = defaultdict(list)


def _digest(*frames: object) -> str:
    """Stable digest of the inputs a primitive was called with."""
    h = hashlib.sha256()
    for f in frames:
        if isinstance(f, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(f, index=True).values.tobytes())
            h.update("|".join(map(str, f.columns)).encode())
        elif isinstance(f, pd.Series):
            h.update(pd.util.hash_pandas_object(f, index=True).values.tobytes())
        else:
            h.update(repr(f).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def install_counters() -> None:
    """Wrap the three expensive primitives with counting + input-digest probes.

    Wrapping rather than editing the modules keeps this harness strictly
    observational: nothing it does can change a reported number.

    Patching the module attribute works — and is the only thing that works —
    because `strategies.py` imports each primitive lazily INSIDE `fit()`
    (`from regime import fit_hmm`, `from dcc_garch import dcc_covariance`,
    `from ml_signals import fit_predict_expected_returns`), so the name is
    resolved at call time, after this function has replaced it.
    """
    import dcc_garch
    import ml_signals
    import regime

    real_fit_hmm = regime.fit_hmm

    def counted_fit_hmm(feature_window, **kw):
        t0 = time.perf_counter()
        out = real_fit_hmm(feature_window, **kw)
        CALLS["fit_hmm"].append({
            "seconds": time.perf_counter() - t0,
            "digest": _digest(feature_window[kw.get("features") or regime.REGIME_FEATURES]),
            "rows": len(feature_window),
        })
        return out

    regime.fit_hmm = counted_fit_hmm

    real_predict = ml_signals.fit_predict_expected_returns

    def counted_predict(train_returns, extras=None, **kw):
        t0 = time.perf_counter()
        out = real_predict(train_returns, extras=extras, **kw)
        CALLS["ml_signal_predict"].append({
            "seconds": time.perf_counter() - t0,
            "model_type": kw.get("model_type", "random_forest"),
            # mu_transform is applied AFTER the model predicts, so two calls
            # differing only in it share a digest — which is the point.
            "digest": _digest(train_returns, kw.get("model_type"),
                              tuple(sorted((kw.get("model_params") or {}).items()))),
            "mu_transform": kw.get("mu_transform", "none"),
            "rows": len(train_returns),
        })
        return out

    ml_signals.fit_predict_expected_returns = counted_predict

    real_dcc = dcc_garch.dcc_covariance

    def counted_dcc(returns, *a, **kw):
        t0 = time.perf_counter()
        out = real_dcc(returns, *a, **kw)
        CALLS["dcc_covariance"].append({
            "seconds": time.perf_counter() - t0,
            "digest": _digest(returns, a, tuple(sorted(kw.items()))),
            "rows": len(returns),
        })
        return out

    dcc_garch.dcc_covariance = counted_dcc


def profile_universe(universe: str, params: dict) -> dict:
    backtest_params = params["backtest"]
    returns = load_universe(backtest_params["universes"][universe])
    features = load_features(params["ml_features"]["outputs"][universe])

    schedule = _rebalance_schedule(
        returns.index,
        backtest_params["rebalance_freq"],
        backtest_params["min_train_days"],
    )
    tau = schedule[-1]                       # worst case: the widest training window
    train = returns.loc[:tau].copy()
    extras = {"features": features.loc[:tau].copy()}

    rows = []
    for strategy in build_strategies(params):
        CALLS.clear()
        t0 = time.perf_counter()
        strategy.fit(train, extras)
        elapsed = time.perf_counter() - t0
        rows.append({
            "strategy": strategy.name,
            "seconds_per_fit": round(elapsed, 4),
            "estimated_total_s": round(elapsed * len(schedule), 1),
            "hmm_fits": len(CALLS["fit_hmm"]),
            "signal_fits": len(CALLS["ml_signal_predict"]),
            "dcc_fits": len(CALLS["dcc_covariance"]),
        })

    total = sum(r["estimated_total_s"] for r in rows)
    return {
        "universe": universe,
        "assets": returns.shape[1],
        "rows": len(returns),
        "n_rebalances": len(schedule),
        "profiled_at_tau": str(tau.date()),
        "per_strategy": sorted(rows, key=lambda r: -r["estimated_total_s"]),
        "estimated_universe_total_s": round(total, 1),
    }


def duplication_evidence(universe: str, params: dict) -> dict:
    """Run every strategy at ONE τ and group primitive calls by input digest.

    Two calls sharing a digest received identical inputs. Since `fit_hmm`,
    the panel model and `fit_dcc_garch` are all deterministic here (seeded
    restarts, `random_state=0`, fixed initial values), identical inputs imply
    identical outputs — so every call beyond the first in a digest group is
    recomputation, not a different question being asked.
    """
    backtest_params = params["backtest"]
    returns = load_universe(backtest_params["universes"][universe])
    features = load_features(params["ml_features"]["outputs"][universe])
    schedule = _rebalance_schedule(
        returns.index, backtest_params["rebalance_freq"], backtest_params["min_train_days"]
    )
    tau = schedule[-1]
    train = returns.loc[:tau].copy()
    extras = {"features": features.loc[:tau].copy()}

    CALLS.clear()
    for strategy in build_strategies(params):
        strategy.fit(train, extras)

    out = {}
    for primitive, calls in CALLS.items():
        groups: dict[str, list[float]] = defaultdict(list)
        for c in calls:
            groups[c["digest"]].append(c["seconds"])
        redundant = sum(sum(v[1:]) for v in groups.values())
        out[primitive] = {
            "calls": len(calls),
            "distinct_inputs": len(groups),
            "redundant_calls": len(calls) - len(groups),
            "seconds_spent_on_redundant_calls_at_one_tau": round(redundant, 3),
            "largest_group": max(len(v) for v in groups.values()) if groups else 0,
        }
    return {"universe": universe, "tau": str(tau.date()), "primitives": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=["etf_2017", "full_2021"], default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--skip-duplication", action="store_true")
    args = ap.parse_args()

    params = load_params()
    universes = [args.universe] if args.universe else list(params["backtest"]["universes"])

    install_counters()
    report: dict = {"universes": [], "duplication": []}

    for u in universes:
        print(f"\n=== profiling {u} ===", flush=True)
        prof = profile_universe(u, params)
        report["universes"].append(prof)
        print(f"  {prof['rows']} rows × {prof['assets']} assets, "
              f"{prof['n_rebalances']} rebalances, τ={prof['profiled_at_tau']}")
        print(f"  {'strategy':<22} {'s/fit':>8} {'est. total':>11}  hmm sig dcc")
        for r in prof["per_strategy"]:
            print(f"  {r['strategy']:<22} {r['seconds_per_fit']:>8.3f} "
                  f"{r['estimated_total_s']:>10.1f}s  "
                  f"{r['hmm_fits']:>3} {r['signal_fits']:>3} {r['dcc_fits']:>3}")
        print(f"  → estimated universe total: {prof['estimated_universe_total_s']/60:.1f} min")

        if not args.skip_duplication:
            dup = duplication_evidence(u, params)
            report["duplication"].append(dup)
            print(f"  duplication at τ={dup['tau']}:")
            for prim, d in dup["primitives"].items():
                print(f"    {prim:<20} {d['calls']:>3} calls, "
                      f"{d['distinct_inputs']:>2} distinct inputs, "
                      f"{d['redundant_calls']:>3} redundant "
                      f"({d['seconds_spent_on_redundant_calls_at_one_tau']:.2f}s wasted here)")

    grand = sum(u["estimated_universe_total_s"] for u in report["universes"])
    report["estimated_stage_total_s"] = round(grand, 1)
    print(f"\n=== estimated phase4c_compare total: {grand/60:.1f} min "
          f"({grand/3600:.2f} h) ===")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"written → {args.json}")


if __name__ == "__main__":
    main()
