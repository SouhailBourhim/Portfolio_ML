"""
deep_morocco_starvation.py — Did the ML underperform because it was DATA-STARVED?

Phase 5 found the F7 return-prediction signal statistically indistinguishable
from the regime baseline. Two explanations: the signal is genuinely absent, OR
it was starved — 9 assets, ~1.7-yr test window, huge confidence intervals. This
experiment tests the STARVATION hypothesis on a deep Moroccan universe built
from 20-year investing.com histories (2005-2024), assembled from the CSVs the
team downloaded on 2026-07-22.

Two stages, cheapest-decisive-first:

  STAGE A (no backtest): purged-CV INFORMATION COEFFICIENT for RF/XGB.
    THE headline. If IC rises above Phase 5's ~0.02 on the deep data, the
    signal was starved. If it stays ~0.02, the ceiling is data QUALITY.

  STAGE B: ONE held-out test backtest per model with FIXED sensible levers
    (shrink=0.5, penalty=1.0 — NO grid, deliberately: the grid is what made an
    earlier run take 10h), alongside regime_conditional / equal_weight /
    max_sharpe, with block-bootstrap 90% CIs.

Deterministic (all seeds fixed) so re-running reproduces the numbers. Reuses
the committed pipeline modules unchanged; writes a results artifact + the test-
window equity curves for the notebook. It is a RESEARCH experiment (self-
contained, not part of the tested medallion pipeline), hence its home in
`experiments/` rather than `src/`.

Data note: the raw CSVs live under `data/bronze/morocco_investing/` (gitignored,
like all of data/). Prices are investing.com UNADJUSTED close in MAD; the
5,000-row free-download cap truncates the oldest names ~2024, so the window
ends 2024-05. Splicing to today (via BVCscrap) and dividend adjustment are
production concerns, irrelevant to the starvation question this run answers.

Usage:
    python experiments/deep_morocco_starvation.py
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from backtest import run_backtest
from metrics import (
    DSRTrialLedger, annualized_sharpe, block_bootstrap_sharpe_ci,
    deflated_sharpe_ratio, max_drawdown,
)
from ml_features import build_return_features
from model_selection import select_ml_hyperparameters
from strategies import (
    EqualWeight, MaxSharpe, MinVarianceLW, RegimeConditionalStrategy,
    RandomForestSignalStrategy, XGBoostSignalStrategy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("deep_morocco")

# The DEEP 12-asset universe: names with continuous history from 2005, across
# banking / telecom / cement / mining / steel / consumer / energy / insurance.
FILE_TO_TICKER = {
    "Afriquia Gaz": "GAZ", "Bmce Bank": "BOA", "Ciments Du Maroc": "CMA",
    "Compagnie Sucrerie": "CSR", "LafargeHolcim": "LHM", "Managem": "MNG",
    "Siderurgie": "SID", "Wafa Assurance": "WAA", "Attijariwafa": "ATW",
    "BCP Stock": "BCP", "CIH Stock": "CIH", "Itissalat": "IAM",
}
START, END = "2005-01-03", "2024-05-31"
TEST_FRAC, MAX_W, RF = 0.35, 0.20, 0.0
BVC_COST_BPS = 30.0
RAW_DIR = ROOT / "data" / "bronze" / "morocco_investing"
OUT_JSON = ROOT / "data" / "gold" / "deep_morocco_results.json"
OUT_EQUITY = ROOT / "data" / "gold" / "deep_morocco_equity.parquet"
PHASE5_IC_REF = "0.015-0.036"

REGIME_KW = dict(n_states=2, n_restarts=5, random_state_base=0,
                 covariance_type="diag", min_regime_train_days=252)
RF_GRID = {"max_depth": [3, 4, 6], "min_samples_leaf": [10, 20], "n_estimators": [150]}
XGB_GRID = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [150]}


def load_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar-align the raw CSVs into a log-return matrix + market features."""
    files = glob.glob(str(RAW_DIR / "*Stock Price History.csv"))
    series = {}
    for sub, tk in FILE_TO_TICKER.items():
        match = [f for f in files if sub in os.path.basename(f)]
        if not match:
            raise FileNotFoundError(f"Missing raw CSV for {tk} (pattern '{sub}') in {RAW_DIR}")
        df = pd.read_csv(match[0], thousands=",")
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        series[tk] = df.sort_values("Date").set_index("Date")["Price"].astype(float)
    # Business-day reference calendar; ffill capped at 5 (clean.py convention).
    cal = pd.bdate_range(START, END)
    prices = pd.DataFrame({tk: v.reindex(cal).ffill(limit=5) for tk, v in series.items()}).dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()
    log_ret.index.name = "Date"
    return log_ret, build_return_features(log_ret)


def main() -> dict:
    log_ret, features = load_universe()
    split = int(round(len(log_ret) * (1 - TEST_FRAC)))
    train_val = log_ret.iloc[:split]
    test_start = log_ret.index[split]
    tv_feat = features.loc[features.index <= train_val.index[-1]]
    log.info("UNIVERSE: %d stocks x %d days [%s -> %s], pooled ~%d rows",
             log_ret.shape[1], log_ret.shape[0], log_ret.index.min().date(),
             log_ret.index.max().date(), log_ret.shape[1] * log_ret.shape[0])
    log.info("SPLIT: train+val -> %s | frozen test %s -> %s (%d rows)",
             train_val.index.max().date(), test_start.date(),
             log_ret.index.max().date(), len(log_ret) - split)

    # ---- STAGE A: purged-CV information coefficient (the headline) ----------
    ic = {}
    for mt, disp, grid in [("random_forest", "rf", RF_GRID), ("xgboost", "xgb", XGB_GRID)]:
        best, tab = select_ml_hyperparameters(
            train_val, tv_feat, grid, model_type=mt, n_splits=5, embargo_frac=0.02,
            short_window=21, long_window=63, momentum_windows=(5, 21, 63),
            condition_on_regime=True, regime_kwargs=REGIME_KW,
        )
        ic[disp] = {"mean_ic": round(float(tab.iloc[0]["mean_ic"]), 4), "params": best}
        log.info("STAGE A %s: CV IC=%.4f params=%s", disp, ic[disp]["mean_ic"], best)

    # ---- STAGE B: held-out test backtests + bootstrap CIs -------------------
    ledger = DSRTrialLedger()
    strategies = {
        "rf_tuned": RandomForestSignalStrategy(
            name="rf_tuned", mu_transform="shrink", shrinkage_weight=0.5, turnover_penalty=1.0,
            model_params=ic["rf"]["params"], max_weight=MAX_W, risk_free_annual=RF,
            min_train_rows=504, short_window=21, long_window=63, momentum_windows=(5, 21, 63),
            condition_on_regime=True, **REGIME_KW),
        "xgb_tuned": XGBoostSignalStrategy(
            name="xgb_tuned", mu_transform="shrink", shrinkage_weight=0.5, turnover_penalty=1.0,
            model_params=ic["xgb"]["params"], max_weight=MAX_W, risk_free_annual=RF,
            min_train_rows=504, short_window=21, long_window=63, momentum_windows=(5, 21, 63),
            condition_on_regime=True, **REGIME_KW),
        "regime_conditional": RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=MAX_W), bear_strategy=MinVarianceLW(max_weight=MAX_W)),
        "equal_weight": EqualWeight(),
        "max_sharpe": MaxSharpe(max_weight=MAX_W),
    }
    strat_results, equity = {}, {}
    for name, strat in strategies.items():
        r = run_backtest(log_ret, strat, rebalance_freq="ME", min_train_days=252,
                         cost_bps=BVC_COST_BPS, extras={"features": features},
                         universe_name="deep_morocco", max_weight=MAX_W)
        test_net = r.net_returns.loc[r.net_returns.index >= test_start]
        pt, lo, hi = block_bootstrap_sharpe_ci(test_net, block_len=21, n_boot=2000,
                                               alpha=0.10, risk_free_annual=RF, seed=0)
        ledger.record("deep_morocco", test_net)
        equity[name] = (1 + test_net).cumprod()
        strat_results[name] = {
            "test_sharpe_net": round(pt, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "test_max_drawdown": round(max_drawdown(test_net), 4),
            "avg_turnover": round(float(r.turnover.mean()), 4),
        }
        log.info("STAGE B %s: Sharpe %.4f [%.3f, %.3f]", name, pt, lo, hi)

    best_f7 = max(["rf_tuned", "xgb_tuned"], key=lambda k: strat_results[k]["test_sharpe_net"])
    pool = ledger.pool("deep_morocco")
    dsr = deflated_sharpe_ratio(equity[best_f7].pct_change().dropna(), pool) if len(pool) >= 2 else float("nan")

    out = {
        "experiment": "deep_morocco_data_starvation",
        "date": "2026-07-23",
        "universe": {
            "n_assets": int(log_ret.shape[1]), "n_days": int(log_ret.shape[0]),
            "tickers": list(log_ret.columns), "start": str(log_ret.index.min().date()),
            "end": str(log_ret.index.max().date()),
            "pooled_rows": int(log_ret.shape[1] * log_ret.shape[0]),
            "test_start": str(test_start.date()),
            "comparison_current": {"n_assets": 9, "pooled_rows_approx": 11700,
                                   "note": "current full_2021 universe (2021-07 -> today)"},
        },
        "information_coefficient": {**ic, "phase5_reference": PHASE5_IC_REF},
        "strategies": strat_results,
        "best_f7": best_f7,
        "best_f7_dsr_vs_search": round(dsr, 4) if dsr == dsr else None,
        "n_search_trials": ledger.n_trials("deep_morocco"),
        "levers": {"mu_transform": "shrink", "shrinkage_weight": 0.5, "turnover_penalty": 1.0,
                   "note": "FIXED (not selected) — this run answers the DATA question, not the tuning one"},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    pd.DataFrame(equity).to_parquet(OUT_EQUITY)
    log.info("Wrote %s and %s", OUT_JSON.name, OUT_EQUITY.name)
    log.info("VERDICT: best ML %s @ %.3f | regime %.3f | Markowitz %.3f | 1/N %.3f | CV IC rose to %.3f/%.3f",
             best_f7, strat_results[best_f7]["test_sharpe_net"],
             strat_results["regime_conditional"]["test_sharpe_net"],
             strat_results["max_sharpe"]["test_sharpe_net"],
             strat_results["equal_weight"]["test_sharpe_net"],
             ic["rf"]["mean_ic"], ic["xgb"]["mean_ic"])
    return out


if __name__ == "__main__":
    main()
