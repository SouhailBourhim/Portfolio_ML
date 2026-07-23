"""
fundamentals_portfolio.py — Does the fundamentals IC lift produce a Sharpe lift?

The IC-lift experiment (`fundamentals_ic_lift.py`) found +0.03 mean IC on
RandomForest, +0.01 on XGBoost, with fundamentals attached (`full_2021`, strict
same-(date, asset) comparison). Feature importance showed FUND_pb alone
carries 15.75% of the RF's total importance — the tree really uses it.

But the deep-Morocco experiment already taught us the hard lesson: prediction
accuracy ≠ portfolio performance. Even a doubled IC produced no significant
portfolio edge because the ML signal, when acted on with realistic constraints
and costs, gets swamped by turnover and estimation noise. This experiment is
the honest test of whether fundamentals cross that gap.

Design — small and bounded, following the deep-Morocco pattern:

  * Frozen 35% test window from Phase 5 (same as every prior OOS number).
  * FOUR strategies, one config each, no lever grid (the Phase 5 lever grid
    is what makes runs take hours; this experiment is scoped to a verdict,
    not a tuning pass):
      1. equal_weight              — the true baseline
      2. regime_conditional        — the Phase 4 hurdle
      3. rf_signal_baseline        — F7 RF, prices only (Phase 5 exact)
      4. rf_signal_fundamentals    — F7 RF, prices + fundamentals
  * All Phase 4/4B/5 machinery unchanged — only the fundamentals frame is
    injected as a new `extras["fundamentals"]` key on strategy #4.
  * RF hyperparameters: the ones the IC experiment selected for the treatment
    (max_depth=3, min_samples_leaf=10, n_estimators=200). Same config for
    baseline for a fair comparison — no per-treatment tuning, which would
    itself be a P4 violation on this held-out window.
  * Report block-bootstrap 90% CI on each strategy's net Sharpe, then compare.

Honest expectations up front (so they can be tested against, not rationalized
after): given deep-Morocco's precedent, "no significant portfolio lift despite
a positive IC" is the more likely outcome. The valuable finding either way is:
if the null holds a THIRD time (deep-Morocco data, F7 prices, F7+fundamentals),
the ML approach's ceiling is well-supported empirically, not just claimed.

Usage:
    .venv/bin/python experiments/fundamentals_portfolio.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from backtest import build_cost_vector, run_backtest
from metrics import annualized_sharpe, block_bootstrap_sharpe_ci
from strategies import (
    EqualWeight,
    MaxSharpe,
    MinVarianceLW,
    RandomForestSignalStrategy,
    RegimeConditionalStrategy,
)
from utils import load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fundamentals_portfolio")

# Fixed config — the RF hyperparameters the IC experiment's treatment side
# picked as best. Using the SAME config for baseline (no per-arm tuning) makes
# the "did fundamentals help?" comparison strictly about the added feature.
RF_PARAMS = {"max_depth": 3, "min_samples_leaf": 10, "n_estimators": 200}


def _load_fundamentals(root: Path, fund_cfg: dict) -> pd.DataFrame:
    """Load and rename the fundamentals panel to match .CS-suffixed universe columns."""
    fund = pd.read_parquet(root / fund_cfg["output_path"])
    rename = {}
    for c in fund.columns:
        if "__FUND_" in c:
            ticker, rest = c.split("__FUND_")
            rename[c] = f"{ticker}.CS__FUND_{rest}"
    return fund.rename(columns=rename)


def main() -> None:
    params = load_params()
    universe = "full_2021"
    returns_path = ROOT / "data/gold/log_returns.parquet"
    features_path = ROOT / "data/gold/ml_features_full.parquet"

    returns = pd.read_parquet(returns_path)
    market_features = pd.read_parquet(features_path).reindex(returns.index).ffill()
    fundamentals = _load_fundamentals(ROOT, params["fundamentals"])

    p5 = params["phase5"]
    test_frac = p5["test_frac"]
    n = int(len(returns) * (1 - test_frac))
    train_val_returns = returns.iloc[:n]
    test_returns = returns.iloc[n:]
    log.info(
        "test window: %s → %s (%d rows, %.1f years)",
        test_returns.index[0].date(),
        test_returns.index[-1].date(),
        len(test_returns),
        len(test_returns) / 252,
    )

    # Common backtest kwargs (from params.yaml.backtest, verified via the
    # existing run_phase5 config path).
    bt = params["backtest"]
    ms = params["ml_signals"]
    cost_vector = build_cost_vector(
        list(returns.columns),
        etf_cost_bps=bt["costs_bps"]["etf"],
        bvc_cost_bps=bt["costs_bps"]["bvc"],
    )
    bt_kwargs = dict(
        rebalance_freq=bt["rebalance_freq"],
        min_train_days=bt["min_train_days"],
        cost_bps=cost_vector,
        max_weight=bt["max_weight"],
        universe_name=universe,
    )

    # Build the four strategies.
    strategies = {
        "equal_weight": EqualWeight(),
        "regime_conditional": RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=bt["max_weight"]),
            bear_strategy=MinVarianceLW(max_weight=bt["max_weight"]),
        ),
        "rf_signal_baseline": RandomForestSignalStrategy(
            name="rf_signal_baseline",
            max_weight=bt["max_weight"],
            model_params=RF_PARAMS,
            min_train_rows=ms["min_train_rows"],
            short_window=ms["short_window"],
            long_window=ms["long_window"],
            momentum_windows=ms["momentum_windows"],
            condition_on_regime=True,
        ),
        "rf_signal_fundamentals": RandomForestSignalStrategy(
            name="rf_signal_fundamentals",
            max_weight=bt["max_weight"],
            model_params=RF_PARAMS,
            min_train_rows=ms["min_train_rows"],
            short_window=ms["short_window"],
            long_window=ms["long_window"],
            momentum_windows=ms["momentum_windows"],
            condition_on_regime=True,
        ),
    }

    # Only the fundamentals strategy receives extras["fundamentals"]; all four
    # receive extras["features"] so regime conditioning is available.
    extras_baseline = {"features": market_features}
    extras_fundamentals = {"features": market_features, "fundamentals": fundamentals}

    # Run each strategy on the FULL returns series — the engine does the walk-
    # forward, respects min_train_days, and only earns test-window returns after
    # its warmup. We then slice each strategy's net_returns to the test window.
    boot_cfg = p5["bootstrap"]
    results = {}
    for name, strat in strategies.items():
        extras = extras_fundamentals if name == "rf_signal_fundamentals" else extras_baseline
        t0 = time.time()
        log.info("running strategy: %s", name)
        result = run_backtest(returns, strat, extras=extras, **bt_kwargs)
        elapsed = time.time() - t0
        log.info("  %s completed in %.1fs", name, elapsed)

        # Slice net_returns to the frozen test window
        test_net = result.net_returns.loc[test_returns.index[0]:]
        sharpe_pt = float(annualized_sharpe(test_net))
        _, sh_lo, sh_hi = block_bootstrap_sharpe_ci(
            test_net,
            block_len=boot_cfg["block_len"],
            n_boot=boot_cfg["n_boot"],
            alpha=boot_cfg["alpha"],
            seed=boot_cfg["seed"],
        )
        avg_turnover = float(result.turnover.loc[test_returns.index[0]:].mean())
        log.info(
            "  %s | test net Sharpe %.4f  90%% CI [%.4f, %.4f]  turnover %.3f",
            name, sharpe_pt, sh_lo, sh_hi, avg_turnover,
        )
        results[name] = {
            "test_net_sharpe": sharpe_pt,
            "sharpe_ci_lo_90": float(sh_lo),
            "sharpe_ci_hi_90": float(sh_hi),
            "avg_test_turnover": avg_turnover,
            "test_window_start": str(test_returns.index[0].date()),
            "test_window_end": str(test_returns.index[-1].date()),
            "elapsed_seconds": elapsed,
        }

    # ── Report ───────────────────────────────────────────────────────────
    log.info("=" * 78)
    log.info("SUMMARY — full_2021 test window Sharpe with 90%% bootstrap CIs")
    log.info("=" * 78)
    for name, r in results.items():
        log.info(
            "  %-30s  Sharpe %.4f  [%.4f, %.4f]  turnover %.3f",
            name,
            r["test_net_sharpe"], r["sharpe_ci_lo_90"], r["sharpe_ci_hi_90"],
            r["avg_test_turnover"],
        )

    # The headline comparison
    base = results["rf_signal_baseline"]
    fund = results["rf_signal_fundamentals"]
    lift = fund["test_net_sharpe"] - base["test_net_sharpe"]
    log.info("")
    log.info(
        "rf_signal LIFT from adding fundamentals: %+.4f Sharpe "
        "(baseline %.4f → fundamentals %.4f)",
        lift, base["test_net_sharpe"], fund["test_net_sharpe"],
    )
    # Rough overlap check: is the fundamentals CI entirely above the baseline CI?
    if fund["sharpe_ci_lo_90"] > base["sharpe_ci_hi_90"]:
        verdict = "SIGNIFICANT: fundamentals CI entirely above baseline CI"
    elif fund["sharpe_ci_hi_90"] < base["sharpe_ci_lo_90"]:
        verdict = "SIGNIFICANT (worse): fundamentals CI entirely below baseline CI"
    else:
        verdict = "NOT SIGNIFICANT: fundamentals and baseline CIs overlap"
    log.info("verdict: %s", verdict)
    log.info("=" * 78)

    out = ROOT / "data/gold/fundamentals_portfolio.json"
    out.write_text(json.dumps({
        "universe": universe,
        "rf_hyperparams": RF_PARAMS,
        "test_window": {
            "start": str(test_returns.index[0].date()),
            "end": str(test_returns.index[-1].date()),
            "n_rows": len(test_returns),
        },
        "results": results,
        "verdict": verdict,
        "rf_signal_lift_sharpe": float(lift),
    }, indent=2))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
