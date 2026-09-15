"""
jump_model_vs_hmm.py — does the jump model deliver the persistence it promises?

WHY THIS EXPERIMENT EXISTS. `docs/LITERATURE_IMPROVEMENTS.md` item 2.1 reports
that HMMs under high persistence, low signal-to-noise and limited data produce
state sequences that lack persistence and stability, and that a statistical jump
model fixes this by charging an explicit penalty at every transition. The
project shows the symptom: six warm-up fallbacks in `regime_conditional`, and
posteriors that move ~0.7 on a refit with states merely relabelled.

THE COMPARISON IS RUN CAUSALLY ON BOTH SIDES, which is the whole reason
`regime.filtered_posterior_series` was built first. Item 2.1's caveat:
a jump model fitted over a full window has the same in-window lookahead as a
smoothed HMM, so comparing a full-window jump model against a smoothed HMM would
measure the estimator difference plus the causality defect and call the sum an
estimator difference. Here BOTH estimators are refitted on each expanding prefix
and only the last state -- the one that saw no future -- is kept.

TWO CLAIMS ARE TESTED SEPARATELY, because the literature makes two.

  PERSISTENCE -- how often the live state sequence switches. This is the one
  that costs money: every switch is a full portfolio rotation in
  `RegimeConditionalStrategy`, whose turnover is already this project's
  measured weak point.

  STABILITY UNDER REFIT -- whether an estimator's later view of history matches
  what it said at the time. Fit on the full panel, read the state it now
  assigns to an earlier date, and compare against what the online run assigned
  to that same date live. An estimator that rewrites its own history is the
  instability §3 describes, and it is invisible to any single fit.

LAMBDA IS SWEPT, NOT SELECTED. Shu-Yu-Mulvey choose it by cross-validation on
strategy performance. `docs/EVALUATION_LIMITS.md` §6 measures what selection
costs on this data -- a degradation slope near -1, unchanged on 3.6x the panel
-- so this reports the whole sweep and lets the reader see the shape. Picking
the best-looking lambda here would be a search, and a one-point search reported
as a finding is exactly what the PBO work warns against.
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

from jump_model import fit_jump_model, label_states_by_volatility, online_jump_states
from regime import REGIME_FEATURES, fit_hmm
from regime import filtered_posterior_series

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("jump_vs_hmm")

GOLD = ROOT / "data" / "gold"
OUT_JSON = GOLD / "jump_model_vs_hmm.json"

PANEL = "full_2021"
FEATURES = GOLD / "ml_features_full.parquet"
N_STATES, MIN_TRAIN = 2, 252
LAMBDAS = [0.0, 1.0, 2.5, 5.0, 10.0, 25.0]
HMM_KW = dict(n_states=N_STATES, n_restarts=5, random_state_base=0)


def _switch_rate(labels: pd.Series) -> tuple[int, float]:
    vals = labels.to_numpy()
    n = int((vals[1:] != vals[:-1]).sum())
    return n, round(n / max(len(vals) - 1, 1), 4)


def _online_hmm_states(feats: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Causal HMM states: refit per prefix, read the FILTERED posterior's last row."""
    out = {}
    for date in dates:
        prefix = feats.loc[feats.index <= date]
        if len(prefix) < MIN_TRAIN:
            continue
        fit = fit_hmm(prefix, **HMM_KW)
        if not fit.converged:
            continue
        post = filtered_posterior_series(fit, prefix)
        if post.empty:
            continue
        out[date] = "bull" if float(post["bull"].iloc[-1]) >= 0.5 else "bear"
    return pd.Series(out, name="regime", dtype=object)


def main() -> dict:
    feats = pd.read_parquet(FEATURES)[REGIME_FEATURES].dropna()
    dates = pd.DatetimeIndex(
        [g.index[-1] for _, g in feats.groupby(pd.Grouper(freq="ME")) if len(g)]
    )
    dates = dates[dates >= feats.index[MIN_TRAIN - 1]]
    log.info("PANEL %s: %d feature rows | %d monthly evaluation dates (%s -> %s)",
             PANEL, len(feats), len(dates), dates[0].date(), dates[-1].date())

    log.info("Running the HMM online (refit per prefix, filtered posterior)...")
    hmm_live = _online_hmm_states(feats, dates)
    hmm_n, hmm_rate = _switch_rate(hmm_live)

    # Stability: what does a full-panel HMM fit now say about those same dates?
    full_fit = fit_hmm(feats, **HMM_KW)
    full_post = filtered_posterior_series(full_fit, feats)
    hmm_retro = pd.Series(
        {d: ("bull" if float(full_post["bull"].loc[d]) >= 0.5 else "bear")
         for d in hmm_live.index if d in full_post.index}, dtype=object)
    shared = hmm_live.index.intersection(hmm_retro.index)
    hmm_rewrite = round(float((hmm_live[shared] != hmm_retro[shared]).mean()), 4)
    log.info("HMM       switches %2d (rate %.3f) | rewrites %.1f%% of its own history",
             hmm_n, hmm_rate, 100 * hmm_rewrite)

    rows = {"hmm_filtered": {"n_switches": hmm_n, "switch_rate": hmm_rate,
                             "history_rewrite_rate": hmm_rewrite,
                             "n_dates": int(len(hmm_live))}}

    for lam in LAMBDAS:
        live = online_jump_states(feats, dates, n_states=N_STATES, jump_penalty=lam,
                                  min_train_rows=MIN_TRAIN, n_init=3, random_state=0)
        n_sw, rate = _switch_rate(live)

        full_jump = fit_jump_model(feats, n_states=N_STATES, jump_penalty=lam,
                                   n_init=3, random_state=0)
        labels = label_states_by_volatility(full_jump, feats)
        retro = pd.Series(
            {d: labels.get(int(full_jump.states[feats.index.get_loc(d)]), "unknown")
             for d in live.index if d in feats.index}, dtype=object)
        sh = live.index.intersection(retro.index)
        rewrite = round(float((live[sh] != retro[sh]).mean()), 4)

        agree = live.index.intersection(hmm_live.index)
        agreement = round(float((live[agree] == hmm_live[agree]).mean()), 4)

        rows[f"jump_lambda_{lam}"] = {
            "jump_penalty": lam, "n_switches": n_sw, "switch_rate": rate,
            "history_rewrite_rate": rewrite, "agreement_with_hmm": agreement,
            "n_dates": int(len(live)),
            "full_panel_switches": int(full_jump.n_switches),
        }
        log.info("jump λ=%5.1f switches %2d (rate %.3f) | rewrites %.1f%% | agrees with HMM %.1f%%",
                 lam, n_sw, rate, 100 * rewrite, 100 * agreement)

    best_stability = min(rows.items(), key=lambda kv: kv[1]["history_rewrite_rate"])
    out = {
        "experiment": "jump_model_vs_hmm",
        "panel": PANEL,
        "n_evaluation_dates": int(len(dates)),
        "causality": ("both estimators refit on expanding prefixes; only the last state of "
                      "each prefix is kept, so neither sees its own future"),
        "lambda_policy": "swept and reported in full, not selected — selection would be a search",
        "results": rows,
        "lowest_history_rewrite": best_stability[0],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUT_JSON.name)
    return out


if __name__ == "__main__":
    main()
