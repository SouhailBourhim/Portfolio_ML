"""
etf_deep_history.py — Does extending the ETF universe back to 2004 help?

The project's recurring blocker is not model quality, it is STATISTICAL POWER.
Phase 5, the deep-Morocco experiment and the fundamentals experiment all ended
the same way: every block-bootstrap confidence interval spans roughly two
Sharpe points, because the out-of-sample windows are 1.7-1.8 years long. No
model comparison can be resolved at that width.

Meanwhile `etf_2017` starts in 2017 purely because of a project decision made
in Phase 1. yfinance actually serves all five ETFs back to 2004-11-18 (GLD's
inception is the binding constraint; SPY reaches 1993). That is roughly twelve
extra years, free, dividend-adjusted, and it contains the 2008 crisis — an
event the regime detector has never been tested on.

This experiment measures whether taking it changes anything, and separates
two effects that are easy to conflate:

  EFFECT A — more TRAINING history.
    Compare the two universes on the SAME out-of-sample window (2018+).
    Only the amount of history available to each fit differs. Answers:
    "does seeing 2005-2017, including 2008, make the strategies better?"

  EFFECT B — more TEST history.
    Compare each universe on its OWN full out-of-sample window. The deep
    universe gets ~20 years of OOS instead of ~8.5. Answers: "do the
    confidence intervals actually tighten?" — the question that matters for
    every conclusion this project has been unable to draw.

Deliberately a RESEARCH experiment, not a pipeline change: it builds its own
Gold matrices under a separate filename and never touches the committed
`log_returns_etf.parquet`, so every number already in the deliverables stays
valid and the comparison stays controlled (same assets, same strategies, same
engine — only the window differs).

Usage:
    .venv/bin/python experiments/etf_deep_history.py
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
from clean import align_calendars, compute_log_returns
from metrics import annualized_sharpe, block_bootstrap_sharpe_ci, max_drawdown
from ml_features import build_ml_feature_set
from strategies import (
    EqualWeight,
    MaxSharpe,
    MinVarianceLW,
    RegimeConditionalStrategy,
)
from utils import load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etf_deep_history")

ETFS = ["SPY", "QQQ", "EEM", "GLD", "TLT"]
# GLD's inception (2004-11-18) is the binding constraint across the five.
DEEP_START = "2004-11-18"
# The committed universe's start, for the controlled comparison.
SHALLOW_START = "2017-01-01"
# Where EFFECT A's shared out-of-sample window begins (etf_2017's first
# rebalance clears min_train_days=252 in early 2018).
SHARED_OOS_START = "2018-01-01"

CACHE = ROOT / "data" / "bronze" / "etf_deep_prices.parquet"


def fetch_deep_prices(force: bool = False) -> pd.DataFrame:
    """Adjusted close for the five ETFs back to GLD's inception.

    `auto_adjust=True` is non-negotiable (CLAUDE.md §15.3) — it is also what
    makes this extension methodologically free, unlike the BVC deep history,
    whose investing.com prices are UNADJUSTED and would inject a systematic
    dividend bias if spliced against these.
    """
    if CACHE.exists() and not force:
        log.info("using cached deep prices: %s", CACHE)
        return pd.read_parquet(CACHE)

    import yfinance as yf

    frames = {}
    for ticker in ETFS:
        hist = yf.Ticker(ticker).history(start=DEEP_START, auto_adjust=True)
        frames[ticker] = hist["Close"]
        log.info("  %s: %d rows from %s", ticker, len(hist), hist.index.min().date())

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(CACHE)
    log.info("cached %d rows x %d assets -> %s", *prices.shape, CACHE)
    return prices


def build_universe(prices: pd.DataFrame, start: str, params: dict
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log-returns + ML features for one window, via the committed pipeline functions.

    Reuses `clean.align_calendars` / `clean.to_log_returns` /
    `ml_features.build_ml_feature_set` unchanged, so the deep universe is
    constructed exactly the way the committed one was — the comparison
    isolates the WINDOW, not the construction.
    """
    window = prices.loc[start:]
    aligned = align_calendars(window, ffill_limit=params["clean"]["ffill_limit"])
    returns = compute_log_returns(aligned)

    # Macro features need a macro frame. The committed Gold macro matrix only
    # starts in 2017, so for the deep window we fall back to return-derived
    # market features only — the regime model's three inputs (MARKET_RETURN,
    # MARKET_VOL_SHORT, AVG_PAIRWISE_CORR) are all return-derived anyway, so
    # the HMM is unaffected. Documented, not silent.
    empty_macro = pd.DataFrame(index=returns.index)
    features = build_ml_feature_set(returns, empty_macro, params["ml_features"])
    return returns, features


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


def evaluate(net: pd.Series, boot: dict, rf: float) -> dict:
    point, lo, hi = block_bootstrap_sharpe_ci(
        net, block_len=boot["block_len"], n_boot=boot["n_boot"],
        alpha=boot["alpha"], seed=boot["seed"],
    )
    return {
        "sharpe_net": round(float(annualized_sharpe(net, rf)), 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "ci_width": round(float(hi - lo), 4),
        "max_drawdown": round(float(max_drawdown(net)), 4),
        "n_days": int(len(net)),
        "years": round(len(net) / 252, 2),
    }


def run_window(label: str, returns: pd.DataFrame, features: pd.DataFrame,
               params: dict, oos_start: str | None) -> dict:
    """Backtest all four strategies; score on the full OOS or a restricted slice."""
    bt = params["backtest"]
    cost_vector = build_cost_vector(
        list(returns.columns),
        etf_cost_bps=bt["costs_bps"]["etf"], bvc_cost_bps=bt["costs_bps"]["bvc"],
    )
    out = {}
    for name, strat in build_strategies(params).items():
        result = run_backtest(
            returns, strat,
            rebalance_freq=bt["rebalance_freq"], min_train_days=bt["min_train_days"],
            cost_bps=cost_vector, extras={"features": features},
            universe_name=label, max_weight=bt["max_weight"],
        )
        net = result.net_returns
        if oos_start is not None:
            net = net.loc[oos_start:]
        out[name] = evaluate(net, params["phase5"]["bootstrap"], bt["risk_free_annual"])
        out[name]["avg_turnover"] = round(float(result.turnover.mean()), 4)
        log.info("  %-18s Sharpe %+.3f  CI [%+.2f, %+.2f]  width %.2f  (%.1f yr)",
                 name, out[name]["sharpe_net"], out[name]["ci_lo"],
                 out[name]["ci_hi"], out[name]["ci_width"], out[name]["years"])
    return out


def diagnose_cap_degeneracy(returns: pd.DataFrame, features: pd.DataFrame,
                            params: dict, caps=(0.25, 0.30, 0.40, 0.60, 1.00)) -> dict:
    """EFFECT C — is the weight cap, not the model, deciding the allocation?

    Surfaced by Effect A: on the deep universe `min_variance_lw`,
    `max_sharpe` and `regime_conditional` returned BYTE-IDENTICAL weights
    after 2018. That is not a coding error, it is arithmetic. With 5 assets
    and a 25% cap, 5 x 0.25 = 1.25, so ANY feasible long-only portfolio must
    hold at least four assets, and the optimizer's only remaining freedom is
    which one to drop. Every objective lands on the same corner.

    This sweep is the causal test: hold everything fixed and vary only the
    cap. If the degeneracy is caused by the constraint, loosening it must
    restore the optimizers' ability to differentiate.
    """
    bt = params["backtest"]
    cost_vector = build_cost_vector(
        list(returns.columns),
        etf_cost_bps=bt["costs_bps"]["etf"], bvc_cost_bps=bt["costs_bps"]["bvc"],
    )
    out = {}
    for cap in caps:
        row = {}
        for name, strat in (
            ("min_variance_lw", MinVarianceLW(max_weight=cap)),
            ("max_sharpe", MaxSharpe(max_weight=cap, risk_free_annual=bt["risk_free_annual"])),
        ):
            result = run_backtest(
                returns, strat,
                rebalance_freq=bt["rebalance_freq"], min_train_days=bt["min_train_days"],
                cost_bps=cost_vector, extras={"features": features},
                universe_name="cap_sweep", max_weight=cap,
            )
            tw = result.target_weights
            n_at_cap = (np.abs(tw.to_numpy() - cap) < 1e-6).sum(axis=1)
            row[name] = {
                "distinct_allocations": len(set(map(tuple, np.round(tw.to_numpy(), 6)))),
                "n_rebalances": int(len(tw)),
                "pct_at_degenerate_corner": round(
                    float((n_at_cap >= tw.shape[1] - 1).mean() * 100), 1),
                "sharpe_net": round(float(annualized_sharpe(
                    result.net_returns, bt["risk_free_annual"])), 4),
            }
        out[f"{cap:.2f}"] = row
        log.info("  cap %.2f | min_var %3d allocs (Sharpe %.3f) | max_sharpe %3d allocs (Sharpe %.3f)",
                 cap, row["min_variance_lw"]["distinct_allocations"],
                 row["min_variance_lw"]["sharpe_net"],
                 row["max_sharpe"]["distinct_allocations"],
                 row["max_sharpe"]["sharpe_net"])
    return out


def main() -> None:
    params = load_params()
    prices = fetch_deep_prices()
    log.info("deep price history: %s -> %s (%d rows)",
             prices.index.min().date(), prices.index.max().date(), len(prices))

    log.info("building universes ...")
    shallow_ret, shallow_feat = build_universe(prices, SHALLOW_START, params)
    deep_ret, deep_feat = build_universe(prices, DEEP_START, params)
    log.info("  etf_2017 (shallow): %s -> %s (%d rows)",
             shallow_ret.index.min().date(), shallow_ret.index.max().date(), len(shallow_ret))
    log.info("  etf_2005 (deep)   : %s -> %s (%d rows)",
             deep_ret.index.min().date(), deep_ret.index.max().date(), len(deep_ret))

    # ── EFFECT A: same OOS window, different training history ──────────────
    log.info("")
    log.info("=" * 78)
    log.info("EFFECT A — more TRAINING history (both scored on %s onward)", SHARED_OOS_START)
    log.info("=" * 78)
    log.info("[shallow: trains from 2017]")
    a_shallow = run_window("etf_2017", shallow_ret, shallow_feat, params, SHARED_OOS_START)
    log.info("[deep: trains from 2004, incl. 2008]")
    a_deep = run_window("etf_2005", deep_ret, deep_feat, params, SHARED_OOS_START)

    # ── EFFECT B: each on its own full OOS window ──────────────────────────
    log.info("")
    log.info("=" * 78)
    log.info("EFFECT B — more TEST history (each on its own full OOS window)")
    log.info("=" * 78)
    log.info("[shallow: ~8.5 yr OOS]")
    b_shallow = run_window("etf_2017", shallow_ret, shallow_feat, params, None)
    log.info("[deep: ~20 yr OOS]")
    b_deep = run_window("etf_2005", deep_ret, deep_feat, params, None)

    # ── Report ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 78)
    log.info("VERDICT")
    log.info("=" * 78)

    log.info("EFFECT A — does 2005-2017 training history improve the 2018+ result?")
    for name in a_shallow:
        d = a_deep[name]["sharpe_net"] - a_shallow[name]["sharpe_net"]
        log.info("  %-18s %.3f -> %.3f  (%+.3f)",
                 name, a_shallow[name]["sharpe_net"], a_deep[name]["sharpe_net"], d)

    log.info("")
    log.info("EFFECT B — do the confidence intervals tighten with a longer test window?")
    for name in b_shallow:
        ws, wd = b_shallow[name]["ci_width"], b_deep[name]["ci_width"]
        pct = (wd - ws) / ws * 100 if ws else float("nan")
        log.info("  %-18s width %.2f (%.1f yr) -> %.2f (%.1f yr)   %+.1f%%",
                 name, ws, b_shallow[name]["years"], wd, b_deep[name]["years"], pct)

    mean_shallow = float(np.mean([v["ci_width"] for v in b_shallow.values()]))
    mean_deep = float(np.mean([v["ci_width"] for v in b_deep.values()]))
    log.info("")
    log.info("  mean CI width: %.3f -> %.3f  (%+.1f%%)",
             mean_shallow, mean_deep, (mean_deep - mean_shallow) / mean_shallow * 100)

    # Does the ML-vs-Markowitz verdict change on the deep universe?
    log.info("")
    log.info("VERDICT FLIP CHECK — ML vs. best classical, per window:")
    for tag, res in (("shallow (2018+ OOS)", b_shallow), ("deep (2005+ OOS)", b_deep)):
        classical = max(("equal_weight", "min_variance_lw", "max_sharpe"),
                        key=lambda s: res[s]["sharpe_net"])
        ml = res["regime_conditional"]["sharpe_net"]
        base = res[classical]["sharpe_net"]
        log.info("  %-22s best classical %-16s %.3f  |  regime %.3f  |  lift %+.1f%%",
                 tag, classical, base, ml, (ml - base) / abs(base) * 100)

    # ── EFFECT C: is the CAP, not the model, deciding? ─────────────────────
    log.info("")
    log.info("=" * 78)
    log.info("EFFECT C — cap-degeneracy diagnostic (deep universe, cap swept)")
    log.info("=" * 78)
    cap_sweep = diagnose_cap_degeneracy(deep_ret, deep_feat, params)

    baseline_cap = f"{params['backtest']['max_weight']:.2f}"
    n_alloc = cap_sweep[baseline_cap]["min_variance_lw"]["distinct_allocations"]
    log.info("")
    log.info("  At the project's cap of %s, min_variance_lw produced %d distinct "
             "allocation(s) across %d rebalances.", baseline_cap, n_alloc,
             cap_sweep[baseline_cap]["min_variance_lw"]["n_rebalances"])
    if n_alloc <= 2:
        log.info("  => The CONSTRAINT is choosing the portfolio, not the covariance model.")
        log.info("     With %d assets and a %s cap, any feasible long-only portfolio must",
                 deep_ret.shape[1], baseline_cap)
        log.info("     hold at least %d of them at the cap — every objective lands on the",
                 int(np.ceil(1.0 / params["backtest"]["max_weight"])))
        log.info("     same corner. This CONFOUNDS every etf_2017 conclusion in the project.")

    out = ROOT / "data" / "gold" / "etf_deep_history.json"
    out.write_text(json.dumps({
        "deep_start": DEEP_START, "shallow_start": SHALLOW_START,
        "shared_oos_start": SHARED_OOS_START,
        "effect_a_training_history": {"shallow": a_shallow, "deep": a_deep},
        "effect_b_test_history": {"shallow": b_shallow, "deep": b_deep},
        "mean_ci_width": {"shallow": round(mean_shallow, 4), "deep": round(mean_deep, 4)},
        "effect_c_cap_degeneracy": cap_sweep,
    }, indent=2))
    log.info("")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
