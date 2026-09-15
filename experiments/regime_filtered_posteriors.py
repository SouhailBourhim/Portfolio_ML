"""
regime_filtered_posteriors.py — how large is Limit #3, really?

WHY THIS EXPERIMENT EXISTS. `docs/LITERATURE_IMPROVEMENTS.md` item 2.1 wants a
statistical jump model compared against the HMM, and attaches a caveat that
fixes the order of work: a jump model fitted over a full window carries the
same in-window lookahead as a smoothed HMM, so comparing the two would confound
the estimator change with the causality defect. Fix causality first.

Doing that first turned out to matter for a second reason. `docs/EVALUATION_LIMITS.md`
Limit #3 bounds the defect at "max drift ~1e-3 in probability units, fewer than
10 of 141 dates move at all". That bound is measured correctly and answers a
different question from the one it is used for.

  What §3 measures: hold the fitted model FIXED, hand `predict_proba` a longer
  window, and see how far already-computed rows move. This experiment
  reproduces it -- extending 141 rows to 1,239 moves shared rows by at most
  1.2e-06, zero rows past 1e-3 -- so the published number is not wrong.

  What the train/serve mismatch actually is: the gap between the SMOOTHED value
  a row carries at training time, P(state_t | x_1..x_T), and the FILTERED value
  the same row would have at inference, P(state_t | x_1..x_t). The backward
  pass contributes ALL future observations to the former, not merely the
  increment between two window lengths.

Those are different quantities and they differ by three orders of magnitude.
The incremental one is small because an old row is already pinned by everything
that followed it in BOTH windows being compared; that is a statement about
marginal information, not about how far the training feature sits from the
servable one.

`regime.filtered_posterior_series` computes the filtered posterior
by an explicit forward recursion, validated against hmmlearn's own smoothed
output at t = T where the two must agree by construction. This measures the gap
on every committed feature panel.

Nothing is fitted for selection and no strategy is run. The HMM is estimated
once per panel with the project's own `fit_hmm` settings, and both posteriors
are read off that single fit, so the refit effect §3 warns about -- posteriors
moving ~0.7 with states merely relabelled -- cannot contaminate the comparison.
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

from regime import REGIME_FEATURES, fit_hmm, predict_regime_posterior_series
from regime import filtered_matches_smoothed_at_final_row, filtered_posterior_series

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("regime_filtered")

GOLD = ROOT / "data" / "gold"
OUT_JSON = GOLD / "regime_filtered_posteriors.json"

PANELS = {
    "full_2021": "ml_features_full.parquet",
    "etf_2017": "ml_features_etf.parquet",
    "global_2004": "ml_features_global.parquet",
}
HMM_KW = dict(n_states=2, n_restarts=5, random_state_base=0)
SWITCH_RADIUS = 5


def main() -> dict:
    out = {
        "experiment": "regime_filtered_posteriors",
        "what_section_3_measures": ("incremental: same fit, longer window, movement of "
                                    "already-computed rows"),
        "what_the_mismatch_is": ("total: smoothed P(state_t | x_1..x_T) against filtered "
                                 "P(state_t | x_1..x_t), same fit"),
        "panels": {},
    }

    for panel, fname in PANELS.items():
        path = GOLD / fname
        if not path.exists():
            log.warning("%s: %s absent, skipping", panel, fname)
            continue
        feats = pd.read_parquet(path)
        missing = [c for c in REGIME_FEATURES if c not in feats.columns]
        if missing:
            log.warning("%s: missing regime features %s, skipping", panel, missing)
            continue

        fit = fit_hmm(feats, **HMM_KW)
        if not fit.converged:
            log.warning("%s: HMM did not converge, skipping", panel)
            continue

        # Positive control: at t = T the two posteriors are the same object.
        ok, dev = filtered_matches_smoothed_at_final_row(fit, feats)
        if not ok:
            raise AssertionError(
                f"{panel}: filtered != smoothed at t=T (dev {dev:.3e}) — the forward "
                f"recursion is wrong and nothing below should be read"
            )

        filtered = filtered_posterior_series(fit, feats)
        smoothed = predict_regime_posterior_series(fit, feats)
        gap = (filtered["bull"] - smoothed["bull"]).abs()

        # Where the backward pass should matter most: around a regime switch.
        state = (smoothed["bull"] > 0.5).astype(int)
        switch = state.diff().abs().fillna(0) > 0
        near = switch.rolling(2 * SWITCH_RADIUS + 1, center=True,
                              min_periods=1).max().astype(bool)

        # Reproduce §3's own diagnostic on this panel, for comparability.
        n_all = len(feats)
        n_short = min(141, max(20, n_all // 8))
        short = predict_regime_posterior_series(fit, feats.iloc[:n_short])["bull"]
        long_ = predict_regime_posterior_series(fit, feats.iloc[:n_all])["bull"].reindex(short.index)
        incremental = (short - long_).abs()

        row = {
            "n_rows": int(len(gap)),
            "positive_control_max_deviation": float(f"{dev:.3e}"),
            "total_gap": {
                "max": round(float(gap.max()), 6),
                "mean": round(float(gap.mean()), 6),
                "median": round(float(gap.median()), 6),
                "share_above_1e-3": round(float((gap > 1e-3).mean()), 4),
                "share_above_0.05": round(float((gap > 0.05).mean()), 4),
                "share_above_0.20": round(float((gap > 0.20).mean()), 4),
                "mean_near_switch": round(float(gap[near].mean()), 6),
                "mean_away_from_switch": round(float(gap[~near].mean()), 6),
                "n_near_switch": int(near.sum()),
            },
            "incremental_gap_section_3_style": {
                "short_window": int(n_short), "long_window": int(n_all),
                "max": float(f"{incremental.max():.3e}"),
                "n_above_1e-3": int((incremental > 1e-3).sum()),
                "n_compared": int(len(incremental)),
            },
        }
        out["panels"][panel] = row
        log.info("%s: n=%d | TOTAL gap max %.4f mean %.4f | near-switch %.4f vs away %.4f "
                 "| %.1f%% move >1e-3", panel, row["n_rows"], row["total_gap"]["max"],
                 row["total_gap"]["mean"], row["total_gap"]["mean_near_switch"],
                 row["total_gap"]["mean_away_from_switch"],
                 100 * row["total_gap"]["share_above_1e-3"])
        log.info("%s: INCREMENTAL (§3 style) max %.3e, %d of %d rows above 1e-3",
                 panel, incremental.max(), row["incremental_gap_section_3_style"]["n_above_1e-3"],
                 len(incremental))

    ratios = [
        p["total_gap"]["max"] / max(p["incremental_gap_section_3_style"]["max"], 1e-12)
        for p in out["panels"].values()
    ]
    if ratios:
        out["total_over_incremental_max_ratio"] = {
            "min": round(float(np.min(ratios)), 1),
            "max": round(float(np.max(ratios)), 1),
        }
        log.info("TOTAL / INCREMENTAL max-gap ratio across panels: %.0fx to %.0fx",
                 np.min(ratios), np.max(ratios))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUT_JSON.name)
    return out


if __name__ == "__main__":
    main()
