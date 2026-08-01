"""
crisis_windows.py — Does the system actually protect when diversification fails?

WHY THIS EXISTS. P3 — "cross-asset correlations spike during crises, eliminating
diversification exactly when it is most needed" — is one of the four problems
this project was built to address, and it is the one with the least direct
evidence. Every phase reports whole-period Sharpe, drawdown and Calmar; none
reports what happened *during the crises themselves*. A strategy can post an
attractive average while being no better than 1/N in the only periods a
portfolio manager loses sleep over.

The brief asks for "la pertinence financière des résultats". For an institution
allocating real capital, behaviour in a drawdown is the financially relevant
question — arguably more so than an average Sharpe.

TWO QUESTIONS, and the second is the one only this project can ask:

  A. PORTFOLIO BEHAVIOUR. In each crisis, what did each strategy actually do —
     cumulative return, worst drawdown, and how long to recover?

  B. REGIME DETECTION. The HMM is unsupervised: it has never been shown a
     crisis date, a recession label, or any external event. Did it independently
     flag these periods as "bear"? If a model trained only on returns,
     volatility and correlation rediscovers the GFC and COVID unprompted, that
     is direct evidence it learned something real (P2) rather than fitting
     noise — and it is testable here because `dashboard_regime.parquet` stores
     the detected regime at all 248 `etf_2017` rebalances back to 2005.

METHODOLOGY — the part that keeps this honest:

  * Crisis windows are defined by EXTERNAL, published S&P 500 peak-to-trough
    dates, fixed before looking at any result. Deriving them from our own
    portfolios' drawdowns would be circular — you would be selecting the
    periods where your strategy looks best and calling it a finding.
  * Drawdown is measured against the running all-time peak, not the
    within-window peak, because that is what an investor actually experiences.
  * ALL strategies are reported for every window, not just the winner.
  * Five windows is five observations. NO significance claim is possible or
    made; this is descriptive evidence, explicitly labelled as such.

Addresses: P3 (diversification breakdown), P2 (regime detection validity),
and the brief's "pertinence financière".

Usage:
    python experiments/crisis_windows.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("crisis")

OUT_PATH = ROOT / "data" / "gold" / "crisis_windows.json"

# Externally-defined S&P 500 peak-to-trough episodes. Fixed BEFORE inspecting
# any portfolio result — see the methodology note above for why that matters.
CRISES = {
    "gfc_2008": {
        "label": "Global Financial Crisis",
        "start": "2007-10-09", "end": "2009-03-09",
        "note": "S&P 500 peak to trough, −56.8%",
    },
    "eu_debt_2011": {
        "label": "EU sovereign debt crisis",
        "start": "2011-04-29", "end": "2011-10-03",
        "note": "S&P 500 peak to trough, −19.4%",
    },
    "q4_2018": {
        "label": "Q4 2018 selloff",
        "start": "2018-09-20", "end": "2018-12-24",
        "note": "S&P 500 peak to trough, −19.8%",
    },
    "covid_2020": {
        "label": "COVID-19 crash",
        "start": "2020-02-19", "end": "2020-03-23",
        "note": "S&P 500 peak to trough, −33.9% in 23 sessions",
    },
    "rate_shock_2022": {
        "label": "2022 rate shock",
        "start": "2022-01-03", "end": "2022-10-12",
        "note": "S&P 500 peak to trough, −25.4%",
    },
}


def wealth_and_drawdown(net: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Cumulative wealth and drawdown vs the RUNNING ALL-TIME peak."""
    wealth = (1.0 + net).cumprod()
    return wealth, wealth / wealth.cummax() - 1.0


def analyse_window(net: pd.Series, start: str, end: str) -> dict | None:
    """Per-crisis statistics for one strategy, or None if not covered."""
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if net.index.min() > a or net.index.max() < a:
        return None
    window = net.loc[(net.index >= a) & (net.index <= b)]
    if len(window) < 5:
        return None

    wealth, drawdown = wealth_and_drawdown(net)
    peak_before = float(wealth.loc[wealth.index <= a].max()) if (wealth.index <= a).any() else float(wealth.iloc[0])

    # Recovery: first date AFTER the window where wealth regains the pre-crisis
    # peak. Still-underwater is reported as None rather than a large number, so
    # it can never be silently averaged into a misleading mean.
    after = wealth.loc[wealth.index > b]
    recovered = after[after >= peak_before]
    recovery_days = int((recovered.index[0] - b).days) if len(recovered) else None

    return {
        "n_days": int(len(window)),
        "cum_return": round(float((1.0 + window).prod() - 1.0), 4),
        "max_drawdown": round(float(drawdown.loc[window.index].min()), 4),
        "worst_day": round(float(window.min()), 4),
        "ann_vol": round(float(window.std() * (252 ** 0.5)), 4),
        "recovery_days": recovery_days,
        "partial_window": bool(window.index.min() > a + pd.Timedelta(days=7)),
    }


def main() -> dict:
    equity = pd.read_parquet(ROOT / "data" / "gold" / "dashboard_equity.parquet")
    regime = pd.read_parquet(ROOT / "data" / "gold" / "dashboard_regime.parquet")

    out: dict = {
        "crises": CRISES,
        "methodology": (
            "Crisis windows are external S&P 500 peak-to-trough dates fixed before "
            "inspecting results; drawdown is measured against the running all-time "
            "peak. Five windows = five observations: descriptive evidence only, no "
            "significance claim."
        ),
        "universes": {},
        "regime_detection": {},
    }

    # ── A. Portfolio behaviour ───────────────────────────────────────────────
    for universe, ug in equity.groupby("universe"):
        per_crisis: dict = {}
        for key, meta in CRISES.items():
            rows = {}
            for strategy, sg in ug.groupby("strategy"):
                net = sg.set_index("Date")["net_return"].sort_index()
                stats = analyse_window(net, meta["start"], meta["end"])
                if stats:
                    rows[strategy] = stats
            if rows:
                per_crisis[key] = rows
        out["universes"][universe] = per_crisis

    # ── B. Did the unsupervised HMM find the crises? ─────────────────────────
    for universe, rg in regime.groupby("universe"):
        r = rg.set_index("Date")["regime"].sort_index()
        in_crisis = pd.Series(False, index=r.index)
        for meta in CRISES.values():
            a, b = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
            in_crisis |= (r.index >= a) & (r.index <= b)

        crisis_obs, calm_obs = r[in_crisis], r[~in_crisis]
        if len(crisis_obs) == 0:
            continue
        detail = {}
        for key, meta in CRISES.items():
            a, b = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
            w = r[(r.index >= a) & (r.index <= b)]
            if len(w):
                detail[key] = {
                    "n_rebalances": int(len(w)),
                    "n_bear": int((w == "bear").sum()),
                    "bear_rate": round(float((w == "bear").mean()), 3),
                }
        n_c, n_o = len(crisis_obs), len(calm_obs)
        b_c, b_o = int((crisis_obs == "bear").sum()), int((calm_obs == "bear").sum())
        rate_c, rate_o = b_c / n_c, (b_o / n_o if n_o else float("nan"))

        # Two tests, deliberately, because they bracket the serial-dependence
        # problem from both sides:
        #
        #   * Fisher exact over all rebalances is the LIBERAL bound. Regimes are
        #     serially correlated — consecutive months inside one crisis are not
        #     independent draws — so this p-value is optimistic and must not be
        #     quoted alone.
        #   * The sign test treats each CRISIS as a single observation (n=5),
        #     which is the conservative reading: it discards all within-crisis
        #     information and asks only whether each distinct episode exceeded
        #     the calm-period base rate. Serial dependence inside a window
        #     cannot inflate it.
        #
        # Reporting both, and leading with the conservative one, is the honest
        # framing for a project whose central lesson is that overlapping error
        # bars kill point comparisons.
        from scipy import stats

        odds, p_fisher = stats.fisher_exact(
            [[b_c, n_c - b_c], [b_o, n_o - b_o]], alternative="greater"
        )
        exceed = sum(1 for v in detail.values() if v["bear_rate"] > rate_o)
        p_sign = float(0.5 ** len(detail)) if exceed == len(detail) else None

        out["regime_detection"][universe] = {
            "bear_rate_in_crisis": round(rate_c, 3),
            "bear_rate_outside": round(rate_o, 3),
            "risk_ratio": round(rate_c / rate_o, 2) if rate_o else None,
            "n_crisis_rebalances": n_c,
            "n_calm_rebalances": n_o,
            "per_crisis": detail,
            "significance": {
                "fisher_exact_p_liberal": float(f"{p_fisher:.3g}"),
                "fisher_odds_ratio": round(float(odds), 1),
                "crises_exceeding_base_rate": f"{exceed}/{len(detail)}",
                "sign_test_p_conservative": p_sign,
                "note": (
                    "Lead with the conservative sign test. Fisher treats serially "
                    "correlated monthly rebalances as independent and is therefore "
                    "optimistic."
                ),
            },
        }

    OUT_PATH.write_text(json.dumps(out, indent=2))

    # ── Report ───────────────────────────────────────────────────────────────
    for universe, per_crisis in out["universes"].items():
        if not per_crisis:
            continue
        log.info("=" * 78)
        log.info("%s", universe)
        for key, rows in per_crisis.items():
            m = CRISES[key]
            log.info("  %s (%s → %s) — %s", m["label"], m["start"], m["end"], m["note"])
            for s, v in sorted(rows.items(), key=lambda kv: -kv[1]["cum_return"]):
                rec = f"{v['recovery_days']}d" if v["recovery_days"] is not None else "not yet"
                partial = "  [PARTIAL WINDOW]" if v["partial_window"] else ""
                log.info("     %-20s ret %+7.2f%%  maxDD %+7.2f%%  worst day %+6.2f%%  "
                         "recovery %-8s%s",
                         s, 100 * v["cum_return"], 100 * v["max_drawdown"],
                         100 * v["worst_day"], rec, partial)
    for universe, d in out["regime_detection"].items():
        log.info("=" * 78)
        log.info("REGIME DETECTION — %s (HMM never saw a crisis date)", universe)
        log.info("  bear rate INSIDE crisis windows : %.1f%%  (%d rebalances)",
                 100 * d["bear_rate_in_crisis"], d["n_crisis_rebalances"])
        log.info("  bear rate OUTSIDE               : %.1f%%  (%d rebalances)",
                 100 * d["bear_rate_outside"], d["n_calm_rebalances"])
        for k, v in d["per_crisis"].items():
            log.info("     %-18s %d/%d bear (%.0f%%)",
                     CRISES[k]["label"], v["n_bear"], v["n_rebalances"], 100 * v["bear_rate"])
        sig = d["significance"]
        log.info("  risk ratio %.2fx | %s crises exceed the base rate",
                 d["risk_ratio"], sig["crises_exceeding_base_rate"])
        log.info("  CONSERVATIVE sign test (each crisis = 1 obs): p = %s",
                 sig["sign_test_p_conservative"])
        log.info("  liberal Fisher exact (ignores serial corr.): p = %s, OR %.1f",
                 sig["fisher_exact_p_liberal"], sig["fisher_odds_ratio"])
    log.info("wrote %s", OUT_PATH)
    return out


if __name__ == "__main__":
    main()
