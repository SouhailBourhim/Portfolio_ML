"""
risk_free_sensitivity.py — Every Sharpe in this project is excess-of-zero.
How much does that choice matter, and does it matter differently per universe?

WHY THIS EXPERIMENT EXISTS. `params.yaml` carries

    risk_free_annual: 0.0      # MVP simplification; revisit in Phase 5

and the project is at Phase 8. An external review flagged it: Sharpe falls by
`rf / sigma`, so a non-zero rate penalises LOW-volatility strategies more than
high-volatility ones and the ranking is not invariant to the choice. Over
2021-2026 Morocco's policy rate ran 1.50% -> 3.00% -> 2.25%, and this project
already ingests it as `TAUX_DIR`.

Rather than assert a rate, this measures with the one the data actually
records, and reports the whole sensitivity curve alongside.

THE NUMERAIRE POINT, which is the part worth understanding. A risk-free rate
must be denominated in the same currency as the returns it is subtracted from.
After the base-currency correction the two universes no longer share a
numeraire:

  * `full_2021` is MAD. `TAUX_DIR` is a MAD policy rate, so it is the
    numeraire-CONSISTENT choice and the realised average over that universe's
    own out-of-sample window is computable from Bronze.
  * `etf_2017` is USD. The Moroccan policy rate is the WRONG rate for it, and
    this project ingests no USD risk-free series. So for that universe the
    honest output is a sensitivity curve with an explicit statement that the
    numeraire-matched rate is UNAVAILABLE, not a number borrowed from the
    other universe.

Subtracting a MAD policy rate from USD returns would be a currency error of
exactly the kind the MAD correction existed to remove. It is not done here.

WHAT THIS CAN AND CANNOT SHOW. It shows how the reported Sharpes, and the
ordering of strategies, respond to the risk-free assumption. It does NOT
re-run any optimisation: `risk_free_annual` also enters the `max_sharpe`
objective, so a project that genuinely adopted a non-zero rate would have to
re-optimise, not merely re-score. That is a full methodology change --
rate timing, annual-to-daily convention, instrument choice, and a complete
rebuild -- and is deliberately NOT attempted here. This is a sensitivity
analysis of the reporting, and is labelled as such in the artifact.

Usage:
    .venv/bin/python experiments/risk_free_sensitivity.py
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

from provenance import build_provenance
from utils import load_params

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rf_sensitivity")

TRADING_DAYS = 252

# The reporting grid. 0.0 is what the project publishes; the rest bracket the
# range Morocco's policy rate actually occupied over the sample.
RF_GRID = (0.0, 0.015, 0.0225, 0.025, 0.03)

# Which universe has a numeraire-consistent rate available, and which does not.
# Stated as data rather than prose so the artifact carries the caveat.
NUMERAIRE_RATE = {
    "full_2021": {
        "base_currency": "MAD",
        "series": "TAUX_DIR",
        "source": "data/bronze/raw_bam_macro.parquet",
        "available": True,
        "note": (
            "Bank Al-Maghrib policy rate. MAD-denominated, matching this "
            "universe's numeraire after the base-currency correction."
        ),
    },
    "etf_2017": {
        "base_currency": "USD",
        "series": None,
        "source": None,
        "available": False,
        "note": (
            "This universe is USD-denominated and the project ingests no USD "
            "risk-free series. TAUX_DIR is a MAD rate and is NOT substituted: "
            "subtracting it from USD returns would be the same class of "
            "currency error the base-currency correction removed. The grid "
            "below is a sensitivity curve, not a corrected Sharpe."
        ),
    },
}


def realised_policy_rate(oos_index: pd.DatetimeIndex) -> dict:
    """
    Time-average of the Moroccan policy rate over an out-of-sample window.

    Addresses: P4 — the alternative to a measured rate is a chosen one, and a
    chosen input is exactly what invites the objection this experiment answers.
    The rate is a step function (it changes only at Bank Al-Maghrib meetings),
    so the mean over the window is its realised average level, not a smoothing.
    """
    macro = pd.read_parquet(ROOT / "data/bronze/raw_bam_macro.parquet")
    if "TAUX_DIR" not in macro.columns:
        raise KeyError("TAUX_DIR absent from raw_bam_macro.parquet.")
    # Forward-fill only: the policy rate holds until the next decision, and a
    # backfill would let a future decision set a past day's rate.
    series = macro["TAUX_DIR"].reindex(
        macro.index.union(oos_index)).ffill().reindex(oos_index)
    series = series.dropna()
    if series.empty:
        raise ValueError("No TAUX_DIR observations inside the OOS window.")
    return {
        "mean_pct": round(float(series.mean()), 4),
        "min_pct": round(float(series.min()), 4),
        "max_pct": round(float(series.max()), 4),
        "n_days": int(len(series)),
        "coverage": round(float(len(series) / len(oos_index)), 4),
    }


def sharpe_at(net: pd.Series, rf_annual: float) -> float:
    """Annualised Sharpe of a daily net series at a given annual risk-free."""
    mu = float(net.mean()) * TRADING_DAYS
    sd = float(net.std()) * np.sqrt(TRADING_DAYS)
    return (mu - rf_annual) / sd if sd > 0 else float("nan")


def analyse_universe(equity: pd.DataFrame, universe: str) -> dict:
    frame = equity[equity["universe"] == universe]
    if frame.empty:
        raise ValueError(f"No rows for {universe}.")

    per_strategy: dict[str, dict] = {}
    for name, group in frame.groupby("strategy"):
        net = group.sort_values("Date")["net_return"]
        per_strategy[name] = {
            "ann_return": round(float(net.mean()) * TRADING_DAYS, 6),
            "ann_vol": round(float(net.std()) * np.sqrt(TRADING_DAYS), 6),
            "sharpe_by_rf": {
                f"{rf:.4f}": round(sharpe_at(net, rf), 4) for rf in RF_GRID
            },
        }

    oos = pd.to_datetime(frame["Date"].unique())
    oos = pd.DatetimeIndex(sorted(oos))

    meta = dict(NUMERAIRE_RATE[universe])
    if meta["available"]:
        meta["realised"] = realised_policy_rate(oos)
        # Percent in the source series -> decimal for the Sharpe arithmetic.
        rf_matched = meta["realised"]["mean_pct"] / 100.0
        meta["numeraire_matched_rf"] = round(rf_matched, 6)
        for name, group in frame.groupby("strategy"):
            net = group.sort_values("Date")["net_return"]
            per_strategy[name]["sharpe_at_numeraire_matched_rf"] = round(
                sharpe_at(net, rf_matched), 4)

    # Ranking at each rf, and where it changes.
    rankings = {}
    for rf in RF_GRID:
        key = f"{rf:.4f}"
        ordered = sorted(per_strategy, key=lambda s: -per_strategy[s]["sharpe_by_rf"][key])
        rankings[key] = ordered
    baseline_order = rankings[f"{RF_GRID[0]:.4f}"]
    rank_changes = {
        rf: order for rf, order in rankings.items() if order != baseline_order
    }

    # The comparison the headline turns on: regime vs the best classical.
    classical = [s for s in per_strategy if s != "regime_conditional"]
    gaps = {}
    for rf in RF_GRID:
        key = f"{rf:.4f}"
        best = max(classical, key=lambda s: per_strategy[s]["sharpe_by_rf"][key])
        r = per_strategy["regime_conditional"]["sharpe_by_rf"][key]
        b = per_strategy[best]["sharpe_by_rf"][key]
        gaps[key] = {
            "best_classical": best,
            "regime_sharpe": r,
            "best_classical_sharpe": b,
            "relative_gap_pct": round(100.0 * (r / b - 1.0), 4) if b != 0 else None,
        }

    return {
        "universe": universe,
        "oos_start": str(oos.min().date()),
        "oos_end": str(oos.max().date()),
        "n_oos_days": int(len(oos)),
        "risk_free_availability": meta,
        "per_strategy": per_strategy,
        "rankings_by_rf": rankings,
        "rankings_that_differ_from_rf_zero": rank_changes,
        "regime_vs_best_classical": gaps,
    }


def main() -> None:
    params = load_params()
    published_rf = params["backtest"]["risk_free_annual"]
    equity = pd.read_parquet(ROOT / "data/gold/dashboard_equity.parquet")

    results = {}
    for universe in ("full_2021", "etf_2017"):
        res = analyse_universe(equity, universe)
        results[universe] = res

        log.info("")
        log.info("=" * 74)
        log.info("%s  (%s -> %s, %d OOS days, %s)",
                 universe, res["oos_start"], res["oos_end"], res["n_oos_days"],
                 res["risk_free_availability"]["base_currency"])
        log.info("=" * 74)

        avail = res["risk_free_availability"]
        if avail["available"]:
            r = avail["realised"]
            log.info("  numeraire-matched rate: %s mean %.4f%% "
                     "(range %.2f-%.2f%%, %d days, coverage %.0f%%)",
                     avail["series"], r["mean_pct"], r["min_pct"], r["max_pct"],
                     r["n_days"], 100 * r["coverage"])
        else:
            log.info("  numeraire-matched rate: UNAVAILABLE (%s universe, no USD "
                     "risk-free ingested) — grid is a sensitivity curve only",
                     avail["base_currency"])

        header = "  %-20s %8s %8s" % ("strategy", "ann.ret", "ann.vol")
        header += "".join(f"{f'rf={rf:.2%}':>10s}" for rf in RF_GRID)
        log.info("")
        log.info(header)
        log.info("  " + "-" * (len(header) - 2))
        ordered = sorted(res["per_strategy"],
                         key=lambda s: -res["per_strategy"][s]["sharpe_by_rf"]["0.0000"])
        for name in ordered:
            s = res["per_strategy"][name]
            row = f"  {name:<20s} {s['ann_return']:8.2%} {s['ann_vol']:8.2%}"
            row += "".join(f"{s['sharpe_by_rf'][f'{rf:.4f}']:10.4f}" for rf in RF_GRID)
            log.info(row)

        log.info("")
        log.info("  regime_conditional vs best classical:")
        for rf in RF_GRID:
            g = res["regime_vs_best_classical"][f"{rf:.4f}"]
            log.info("    rf=%.2f%%  %+8.2f%%  (vs %s)",
                     100 * rf, g["relative_gap_pct"], g["best_classical"])

        if res["rankings_that_differ_from_rf_zero"]:
            log.info("")
            log.info("  RANKING CHANGES vs rf=0:")
            for key, order in res["rankings_that_differ_from_rf_zero"].items():
                log.info("    rf=%.2f%%: %s", 100 * float(key), " > ".join(order))
        else:
            log.info("")
            log.info("  ranking is unchanged across the whole grid.")

    returns = pd.read_parquet(ROOT / "data/gold/log_returns.parquet")
    out = ROOT / "data/gold/risk_free_sensitivity.json"
    out.write_text(json.dumps({
        "provenance": build_provenance(
            universe="full_2021",
            returns=returns,
            source_artifacts=[
                "data/gold/dashboard_equity.parquet",
                "data/bronze/raw_bam_macro.parquet",
                "data/gold/currency_manifest.json",
                "params.yaml",
                "experiments/risk_free_sensitivity.py",
            ],
        ),
        "published_risk_free_annual": published_rf,
        "rf_grid": list(RF_GRID),
        "scope": (
            "Sensitivity of the REPORTED Sharpe ratios and their ordering to the "
            "risk-free assumption. No optimisation is re-run: risk_free_annual "
            "also enters the max_sharpe objective, so genuinely adopting a "
            "non-zero rate requires re-optimisation and a full rebuild, not "
            "re-scoring. Not attempted here and not implied by these numbers."
        ),
        "numeraire_rule": (
            "A risk-free rate must share the numeraire of the returns it is "
            "subtracted from. TAUX_DIR is MAD and is applied only to full_2021. "
            "It is NOT applied to the USD etf_2017 universe."
        ),
        "results": results,
    }, indent=2))
    log.info("")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
