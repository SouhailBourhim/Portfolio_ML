"""
nested_walkforward.py — Phase 5's documented stretch goal, executed.

THE LIMITATION THIS ATTACKS (CLAUDE.md §12D, stated there as the honest next
step): Phase 5 spends the final 35% of each universe as a single frozen test
segment. On `full_2021` that is ~455 rows / ~1.75 years, and the resulting
block-bootstrap intervals span ~2.2 Sharpe — wide enough that every strategy
comparison on that universe is inside the noise. The verdict is therefore not
wrong, it is UNDERPOWERED, and no amount of better modelling fixes that.

WHAT NESTED WALK-FORWARD CHANGES. Instead of selecting once and testing once,
re-select the F7 configuration at each of several outer boundaries and test on
the segment immediately after it, then concatenate every out-of-sample segment
into ONE continuous series. Selection never sees its own evaluation window —
each fold's hyperparameters and levers are chosen strictly from data before
that fold begins — so the concatenated series is honestly out-of-sample of
selection, while covering far more of the history than a single 35% tail.

  single split : |------------ select ------------|===== test =====|
  nested       : |-- select --|== t1 ==|
                 |------ select -------|== t2 ==|
                 |---------- select ---------|== t3 ==|   ... concatenated

Scoped to `full_2021` deliberately. `etf_2017`'s test segment grew from 3.3 to
7.6 years when the deep window was adopted (2026-07-27 rerun) and its intervals
already narrowed 28%; it is no longer the constrained case. Running both would
double the cost to re-answer a question one of them no longer asks.

PRE-REGISTERED OUTCOMES, fixed before the run so the result cannot be
rationalised afterwards:
  (A) Intervals narrow materially AND the ranking is unchanged
      → Phase 5's verdict was sound and is now better powered.
  (B) Intervals narrow materially AND the ranking changes
      → the single-split verdict was a window artefact; the extra power bought
        a different answer, not just a tighter one.
  (C) Intervals do NOT narrow materially
      → the limitation is not sample length; more OOS data will not settle this
        universe and the honest conclusion stays "indistinguishable".

Addresses: P4 — this is backtest-overfitting control at the evaluation level:
more independent out-of-sample observations, selection re-run per fold, error
bars on everything, DSR counting every configuration the whole search touched.

Usage:
    python experiments/nested_walkforward.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import build_cost_vector, run_backtest          # noqa: E402
from metrics import (                                          # noqa: E402
    DSRTrialLedger,
    annualized_sharpe,
    block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio,
    max_drawdown,
)
from model_selection import (                                  # noqa: E402
    select_ml_hyperparameters,
    select_portfolio_levers,
)
from run_phase4 import build_strategies as build_phase4_strategies   # noqa: E402
from run_phase4 import load_features, load_universe            # noqa: E402
from strategies import RandomForestSignalStrategy, XGBoostSignalStrategy  # noqa: E402
from utils import load_params                                  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("nested_wf")

UNIVERSE = "full_2021"
OUT_PATH = ROOT / "data" / "gold" / "nested_walkforward_results.json"

# Where the concatenated out-of-sample series begins, as a fraction of the
# universe. 0.40 leaves ~2 years to make the first selection on (the engine
# needs min_train_days=252 before its first rebalance) and yields ~60% of the
# history as OOS, against Phase 5's 35%.
OOS_START_FRAC = 0.40
# Trading days between re-selections (~6 months). Shorter re-selects more often
# and costs proportionally more; each round is a full purged-CV grid plus a
# lever grid for BOTH model families.
RESELECT_EVERY = 126

MODELS = [
    ("random_forest", RandomForestSignalStrategy, "rf_signal_tuned"),
    ("xgboost", XGBoostSignalStrategy, "xgb_signal_tuned"),
]


def build_folds(n_rows: int) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping [start, end) OOS segments covering the tail."""
    folds: list[tuple[int, int]] = []
    pos = int(round(n_rows * OOS_START_FRAC))
    while pos < n_rows:
        end = min(pos + RESELECT_EVERY, n_rows)
        # Absorb a runt final segment into the previous fold rather than
        # reporting a fold too short to bootstrap.
        if end - pos < RESELECT_EVERY // 3 and folds:
            folds[-1] = (folds[-1][0], end)
        else:
            folds.append((pos, end))
        pos = end
    return folds


def evaluate_segment(returns, features, strategy, lo, hi, bt, cost_vector, max_weight):
    """Walk forward over data up to `hi`, return net returns inside [lo, hi).

    The backtest is run only to `hi` (not over the full series) so no part of
    this call can touch data after the segment being scored — belt and braces
    on top of the engine's own causality, which already guarantees it.
    """
    window = returns.iloc[:hi]
    result = run_backtest(
        window, strategy,
        rebalance_freq=bt["rebalance_freq"],
        min_train_days=bt["min_train_days"],
        cost_bps=cost_vector,
        extras={"features": features.loc[features.index <= window.index[-1]]},
        universe_name=UNIVERSE,
        max_weight=max_weight,
    )
    net = result.net_returns
    lo_date, hi_date = returns.index[lo], window.index[-1]
    return net.loc[(net.index >= lo_date) & (net.index <= hi_date)], result


def main() -> dict:
    params = load_params()
    bp, sp, p5, cv = params["backtest"], params["ml_signals"], params["phase5"], params["purged_cv"]
    boot = p5["bootstrap"]
    max_weight, rf = bp["max_weight"], bp["risk_free_annual"]

    regime_kwargs = dict(
        n_states=params["regime"]["n_states"],
        n_restarts=params["regime"]["n_restarts"],
        random_state_base=params["regime"]["random_state_base"],
        covariance_type=params["regime"]["covariance_type"],
        min_regime_train_days=params["regime"]["min_regime_train_days"],
    )

    returns = load_universe(bp["universes"][UNIVERSE])
    features = load_features(params["ml_features"]["outputs"][UNIVERSE])
    cost_vector = build_cost_vector(
        returns.columns,
        etf_cost_bps=bp["costs_bps"]["etf"], bvc_cost_bps=bp["costs_bps"]["bvc"],
    )

    bt = {
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
        "universe_name": UNIVERSE,
    }

    folds = build_folds(len(returns))
    log.info(
        "=== %s: %d rows; %d nested folds, OOS %s → %s ===",
        UNIVERSE, len(returns), len(folds),
        returns.index[folds[0][0]].date(), returns.index[-1].date(),
    )

    ledger = DSRTrialLedger()
    per_model_segments: dict[str, list[pd.Series]] = {d: [] for _, _, d in MODELS}
    fold_records = []

    for i, (lo, hi) in enumerate(folds, start=1):
        train_val = returns.iloc[:lo]
        tv_features = features.loc[features.index <= train_val.index[-1]]
        log.info(
            "--- fold %d/%d: select on %s→%s (%d rows), score %s→%s ---",
            i, len(folds),
            train_val.index.min().date(), train_val.index.max().date(), len(train_val),
            returns.index[lo].date(), returns.index[hi - 1].date(),
        )

        record = {
            "fold": i,
            "select_end": str(train_val.index.max().date()),
            "oos_start": str(returns.index[lo].date()),
            "oos_end": str(returns.index[hi - 1].date()),
            "n_oos_rows": hi - lo,
            "selected": {},
        }

        for model_type, cls, display in MODELS:
            grid = p5["rf_grid"] if model_type == "random_forest" else p5["xgb_grid"]
            ml_params, _ = select_ml_hyperparameters(
                train_val, tv_features, grid, model_type=model_type,
                n_splits=cv["n_splits"], embargo_frac=cv["embargo_frac"],
                short_window=sp["short_window"], long_window=sp["long_window"],
                momentum_windows=sp["momentum_windows"],
                condition_on_regime=sp["condition_on_regime"], regime_kwargs=regime_kwargs,
            )
            levers, _ = select_portfolio_levers(
                train_val, tv_features, model_type, ml_params,
                shrink_grid=p5["shrink_grid"], penalty_grid=p5["penalty_grid"],
                backtest_kwargs=bt, ledger=ledger, universe=UNIVERSE,
            )
            strategy = cls(
                name=f"{display}__fold{i}",
                max_weight=max_weight, risk_free_annual=rf,
                model_params=dict(ml_params),
                min_train_rows=sp["min_train_rows"],
                short_window=sp["short_window"], long_window=sp["long_window"],
                momentum_windows=sp["momentum_windows"],
                condition_on_regime=sp["condition_on_regime"],
                mu_transform="shrink",
                shrinkage_weight=levers["shrinkage_weight"],
                turnover_penalty=levers["turnover_penalty"],
                **regime_kwargs,
            )
            segment, _ = evaluate_segment(
                returns, features, strategy, lo, hi, bt, cost_vector, max_weight
            )
            per_model_segments[display].append(segment)
            record["selected"][display] = {
                "ml_params": {k: (float(v) if isinstance(v, float) else int(v))
                              for k, v in ml_params.items()},
                "levers": levers,
                "segment_sharpe_net": round(float(annualized_sharpe(segment, rf)), 4),
                "n_rows": int(len(segment)),
            }
            log.info("    %s fold %d: %d rows, segment Sharpe %.4f",
                     display, i, len(segment), annualized_sharpe(segment, rf))

        fold_records.append(record)

    # ── Baselines on the IDENTICAL concatenated dates ────────────────────────
    oos_index = pd.DatetimeIndex(
        sorted(set().union(*[s.index for s in per_model_segments[MODELS[0][2]]]))
    )
    baselines = {}
    baseline_series = {}
    # build_strategies returns a LIST of freshly-instantiated strategies (they
    # must not be reused across universes — RegimeConditionalStrategy carries a
    # per-instance regime_log), so select by `.name`.
    wanted = ("regime_conditional", "equal_weight", "max_sharpe", "min_variance_lw")
    for strategy in build_phase4_strategies(params):
        name = strategy.name
        if name not in wanted:
            continue
        result = run_backtest(
            returns, strategy,
            rebalance_freq=bp["rebalance_freq"], min_train_days=bp["min_train_days"],
            cost_bps=cost_vector, extras={"features": features},
            universe_name=UNIVERSE, max_weight=max_weight,
        )
        series = result.net_returns.reindex(oos_index).dropna()
        baseline_series[name] = series
        ledger.record(UNIVERSE, series)
        baselines[name] = series

    # ── Verdict ──────────────────────────────────────────────────────────────
    def summarize(series: pd.Series) -> dict:
        point, lo_ci, hi_ci = block_bootstrap_sharpe_ci(
            series, block_len=boot["block_len"], n_boot=boot["n_boot"],
            alpha=boot["alpha"], risk_free_annual=rf,
        )
        return {
            "sharpe_net": round(float(point), 4),
            "ci": [round(float(lo_ci), 4), round(float(hi_ci), 4)],
            "ci_width": round(float(hi_ci - lo_ci), 4),
            "max_drawdown": round(float(max_drawdown(series)), 4),
            "n_rows": int(len(series)),
        }

    strategies_out = {}
    for display in per_model_segments:
        concatenated = pd.concat(per_model_segments[display]).sort_index()
        ledger.record(UNIVERSE, concatenated)
        strategies_out[display] = summarize(concatenated)
    for name, series in baselines.items():
        strategies_out[name] = summarize(series)

    pool = ledger.pool(UNIVERSE)
    best = max(strategies_out, key=lambda k: strategies_out[k]["sharpe_net"])
    best_series = (
        pd.concat(per_model_segments[best]).sort_index()
        if best in per_model_segments else baseline_series[best]
    )

    out = {
        "universe": UNIVERSE,
        "design": {
            "oos_start_frac": OOS_START_FRAC,
            "reselect_every_days": RESELECT_EVERY,
            "n_folds": len(folds),
            "oos_start": str(returns.index[folds[0][0]].date()),
            "oos_end": str(returns.index[-1].date()),
            "n_oos_rows": int(len(oos_index)),
        },
        "folds": fold_records,
        "strategies": strategies_out,
        "best": best,
        "n_search_trials": ledger.n_trials(UNIVERSE),
        "best_dsr_vs_search": round(float(deflated_sharpe_ratio(best_series, pool)), 4),
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    log.info("=" * 70)
    log.info("NESTED WALK-FORWARD — %s, %d folds, %d OOS rows",
             UNIVERSE, len(folds), len(oos_index))
    for name in sorted(strategies_out, key=lambda k: -strategies_out[k]["sharpe_net"]):
        s = strategies_out[name]
        log.info("  %-22s %6.3f  [%6.3f, %6.3f]  width %.3f",
                 name, s["sharpe_net"], s["ci"][0], s["ci"][1], s["ci_width"])
    log.info("  best=%s  DSR=%.4f over %d trials",
             best, out["best_dsr_vs_search"], out["n_search_trials"])
    log.info("wrote %s", OUT_PATH)
    return out


if __name__ == "__main__":
    main()
