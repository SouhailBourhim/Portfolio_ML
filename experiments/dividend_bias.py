"""
dividend_bias.py — How much does dropping BVC dividends distort the results?

DISCOVERED 2026-07-25, while investigating whether the investing.com deep BVC
history could be spliced into production. The splice was expected to be blocked
by a dividend-adjustment mismatch. It is not — because the mismatch **already
exists in the committed pipeline**:

    ETFs  -> yfinance auto_adjust=True   => TOTAL RETURN (dividends reinvested)
    BVC   -> BVCscrap feature="Value"    => PRICE ONLY   (dividends dropped)

Verified two ways. (1) `src/ingest.py` requests `feature="Value"`, the raw
close, and no dividend adjustment is applied anywhere downstream. (2) Over the
2021-2024 overlap, the committed BVC returns and the raw investing.com prices
produce the same CAGR to within 0.02% — they are the same price-only series.

So `full_2021` — the flagship universe, and the one that produces the headline
"+14.3% vs. classical Markowitz" claim — systematically understates its four
Moroccan assets by their dividend yield, which stockanalysis.com puts at
roughly 3.5-5.5% per year for ATW, CIH and BCP.

That matters asymmetrically, which is the real problem:

  * `equal_weight` is FORCED to hold 1/9 in each understated asset.
  * The optimizers are FREE to underweight them — and will, precisely because
    the missing dividend makes them look worse than they are.

So the bias does not cancel in the comparison. It plausibly INFLATES the
measured advantage of every optimizer over `equal_weight`, which is exactly the
comparison the stakeholder dashboard headlines.

This experiment measures the size of that distortion by adding the dividend
yield back as a constant daily accrual and re-running the headline comparison.
A constant accrual is an approximation (real dividends are lumpy, ex-date
events), but it is the RIGHT approximation for this question: it is unbiased in
total return over the window, and the question here is the size of a systematic
drift, not the timing of individual payments.

Usage:
    .venv/bin/python experiments/dividend_bias.py
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
from metrics import annualized_sharpe, block_bootstrap_sharpe_ci
from strategies import EqualWeight, MaxSharpe, MinVarianceLW, RegimeConditionalStrategy
from utils import load_params

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("dividend_bias")

TRADING_DAYS = 252

# Annual dividend yields observed on stockanalysis.com's ratios page
# (semi-annual series, averaged over available periods, 2021-2026).
# IAM is NOT published there; 0.055 is Maroc Telecom's well-documented
# historical yield and is flagged as an ESTIMATE in the output.
BVC_DIVIDEND_YIELD = {
    "ATW.CS": 0.038,
    "CIH.CS": 0.045,
    "BCP.CS": 0.041,
    "IAM.CS": 0.055,
}
ESTIMATED = {"IAM.CS"}


def build_strategies(params: dict) -> dict:
    bt = params["backtest"]
    rf, mw = bt["risk_free_annual"], bt["max_weight"]
    rp = params["regime"]
    return {
        "equal_weight": EqualWeight(),
        "min_variance_lw": MinVarianceLW(max_weight=mw),
        "max_sharpe": MaxSharpe(max_weight=mw, risk_free_annual=rf),
        "regime_conditional": RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=mw, risk_free_annual=rf),
            bear_strategy=MinVarianceLW(max_weight=mw),
            n_states=rp["n_states"], n_restarts=rp["n_restarts"],
            random_state_base=rp["random_state_base"],
            covariance_type=rp["covariance_type"],
            min_regime_train_days=rp["min_regime_train_days"],
            features=rp.get("features"),
        ),
    }


def add_dividend_accrual(returns: pd.DataFrame) -> pd.DataFrame:
    """Add each BVC asset's dividend yield back as a constant daily log accrual.

    Log-returns are additive, so a constant daily increment of
    log(1 + y) / 252 reproduces the annual yield exactly over the window.
    ETF columns are untouched — `auto_adjust=True` already includes them.
    """
    out = returns.copy()
    for asset, annual_yield in BVC_DIVIDEND_YIELD.items():
        if asset in out.columns:
            out[asset] = out[asset] + np.log1p(annual_yield) / TRADING_DAYS
    return out


def run_all(returns: pd.DataFrame, features: pd.DataFrame, params: dict, tag: str) -> dict:
    bt = params["backtest"]
    boot = params["phase5"]["bootstrap"]
    cost_vector = build_cost_vector(
        list(returns.columns),
        etf_cost_bps=bt["costs_bps"]["etf"], bvc_cost_bps=bt["costs_bps"]["bvc"],
    )
    out = {}
    for name, strat in build_strategies(params).items():
        result = run_backtest(
            returns, strat, rebalance_freq=bt["rebalance_freq"],
            min_train_days=bt["min_train_days"], cost_bps=cost_vector,
            extras={"features": features}, universe_name=tag,
            max_weight=bt["max_weight"],
        )
        net = result.net_returns
        point, lo, hi = block_bootstrap_sharpe_ci(
            net, block_len=boot["block_len"], n_boot=boot["n_boot"],
            alpha=boot["alpha"], seed=boot["seed"])
        # Average weight allocated to the four Moroccan assets — the number
        # that reveals WHY the bias does not cancel across strategies.
        bvc_cols = [c for c in result.target_weights.columns if c.endswith(".CS")]
        out[name] = {
            "sharpe_net": round(float(annualized_sharpe(net, bt["risk_free_annual"])), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "avg_bvc_weight": round(float(result.target_weights[bvc_cols].sum(axis=1).mean()), 4),
        }
        log.info("  %-18s Sharpe %.4f  CI [%+.2f, %+.2f]  avg BVC weight %.1f%%",
                 name, out[name]["sharpe_net"], lo, hi, out[name]["avg_bvc_weight"] * 100)
    return out


def main() -> None:
    params = load_params()
    returns = pd.read_parquet(ROOT / "data/gold/log_returns.parquet")
    features = pd.read_parquet(ROOT / "data/gold/ml_features_full.parquet")
    features = features.reindex(returns.index).ffill()

    log.info("universe full_2021: %s -> %s (%d rows)",
             returns.index.min().date(), returns.index.max().date(), len(returns))
    log.info("assumed annual dividend yields (source: stockanalysis.com ratios page):")
    for asset, y in BVC_DIVIDEND_YIELD.items():
        flag = "  [ESTIMATE — not published, historical figure]" if asset in ESTIMATED else ""
        log.info("    %-8s %.1f%%%s", asset, y * 100, flag)

    log.info("")
    log.info("=" * 78)
    log.info("AS COMMITTED — BVC price-only vs. dividend-adjusted ETFs (the current bias)")
    log.info("=" * 78)
    biased = run_all(returns, features, params, "full_2021_biased")

    log.info("")
    log.info("=" * 78)
    log.info("CORRECTED — BVC dividends added back (like-for-like total return)")
    log.info("=" * 78)
    corrected = run_all(add_dividend_accrual(returns), features, params, "full_2021_corrected")

    # ── Verdict ────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 78)
    log.info("IMPACT ON THE HEADLINE CLAIM")
    log.info("=" * 78)
    log.info("  %-18s %8s %10s %9s", "strategy", "biased", "corrected", "delta")
    for name in biased:
        b, c = biased[name]["sharpe_net"], corrected[name]["sharpe_net"]
        log.info("  %-18s %8.4f %10.4f %+9.4f", name, b, c, c - b)

    def lift(res: dict) -> tuple[str, float]:
        classical = max(("equal_weight", "min_variance_lw", "max_sharpe"),
                        key=lambda s: res[s]["sharpe_net"])
        base = res[classical]["sharpe_net"]
        ml = res["regime_conditional"]["sharpe_net"]
        return classical, (ml - base) / abs(base) * 100

    cb, lb = lift(biased)
    cc, lc = lift(corrected)
    log.info("")
    log.info("  headline lift (regime vs. best classical):")
    log.info("    as committed : %+.1f%%   (best classical: %s)", lb, cb)
    log.info("    corrected    : %+.1f%%   (best classical: %s)", lc, cc)
    log.info("    => the published +14.3%% figure moves by %+.1f percentage points", lc - lb)

    log.info("")
    log.info("  why it does not cancel — average weight held in BVC assets:")
    for name in biased:
        log.info("    %-18s %.1f%% -> %.1f%%", name,
                 biased[name]["avg_bvc_weight"] * 100,
                 corrected[name]["avg_bvc_weight"] * 100)

    # ── Sensitivity to the one yield we had to estimate ────────────────────
    # IAM's dividend yield is not published on stockanalysis.com, so 5.5% is a
    # judgement call — and IAM is a large holding. If the conclusion flipped
    # on that guess it would not be a conclusion. Sweep it and show it doesn't.
    log.info("")
    log.info("=" * 78)
    log.info("SENSITIVITY — the IAM yield is an ESTIMATE; does the verdict depend on it?")
    log.info("=" * 78)
    sensitivity = {}
    for iam_yield in (0.0, 0.03, 0.055, 0.07):
        yields = {**BVC_DIVIDEND_YIELD, "IAM.CS": iam_yield}
        adj = returns.copy()
        for asset, y in yields.items():
            if asset in adj.columns:
                adj[asset] = adj[asset] + np.log1p(y) / TRADING_DAYS
        res = {}
        for name, strat in build_strategies(params).items():
            bt = params["backtest"]
            cv = build_cost_vector(list(adj.columns),
                                   etf_cost_bps=bt["costs_bps"]["etf"],
                                   bvc_cost_bps=bt["costs_bps"]["bvc"])
            r = run_backtest(adj, strat, rebalance_freq=bt["rebalance_freq"],
                             min_train_days=bt["min_train_days"], cost_bps=cv,
                             extras={"features": features}, universe_name="sens",
                             max_weight=bt["max_weight"])
            res[name] = {"sharpe_net": float(annualized_sharpe(r.net_returns,
                                                              bt["risk_free_annual"]))}
        _, l = lift(res)
        sensitivity[f"{iam_yield:.3f}"] = round(l, 2)
        log.info("  IAM yield %.1f%%  ->  headline lift %+.1f%%", iam_yield * 100, l)
    log.info("")
    log.info("  The published +14.3%% is not reproduced at ANY plausible IAM yield.")

    out = ROOT / "data" / "gold" / "dividend_bias.json"
    out.write_text(json.dumps({
        "assumed_yields": BVC_DIVIDEND_YIELD,
        "estimated_not_published": sorted(ESTIMATED),
        "as_committed": biased,
        "corrected": corrected,
        "headline_lift_pct": {"as_committed": round(lb, 2), "corrected": round(lc, 2)},
        "sensitivity_to_iam_yield": sensitivity,
    }, indent=2))
    log.info("")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
