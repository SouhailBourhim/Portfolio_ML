"""
risk_separation_bound.py — how much risk separation can a universe support at all?

WHY THIS EXPERIMENT EXISTS. `docs/REACHABLE_CLAIMS.md` §3 used the log
volatility contrast between `max_sharpe` and `min_variance_lw` as a POSITIVE
CONTROL: a minimum-variance optimiser produces lower variance than a
maximum-Sharpe one by construction, so recovering it at |t| = 6.10 on
`full_2021` was evidence the instrument works. On the deep Moroccan panel the
same by-construction contrast lands at |t| = 2.14 with a detectability ratio of
0.86 -- it fails before any correction is applied.

`deep_morocco_weight_concentration.py` found the reason, and it was not the one
first proposed. The portfolios are NOT collapsed onto the weight cap: effective
N spans 6.67 to 12.00, a 1.80x spread. They differ. What does not follow is the
volatility: `min_variance_lw` holds 1.32x the effective names of `max_sharpe`
and that buys a vol ratio of only 1.07.

Two channels produce a volatility difference, and only one of them is
diversification. Under equicorrelation -- every pair at the panel's mean
correlation rho, each asset at its OWN volatility sigma_i -- portfolio variance
is exactly

    (1 - rho) * SUM_i w_i^2 sigma_i^2  +  rho * (SUM_i w_i sigma_i)^2

The first term is diversification, and rho caps what it can buy: it is the term
that shrinks with effective N, and it is multiplied by (1 - rho). The second is
a LOW-VOLATILITY TILT -- holding quieter assets -- and it does not depend on
how many names are held at all.

An earlier version of this script kept only the first channel, by assuming a
common sigma. It fitted the deep panel and then failed on both others: on
`etf_2017` `min_variance_lw` and `max_sharpe` hold the SAME effective N (4.00
against 4.02) while realising a 10% volatility gap, which a diversification-only
model must predict as zero. The tilt is the channel that was missing, and on
that panel it is the only one operating.

WHAT THIS SCRIPT SET OUT TO DO, AND WHAT HAPPENED. The hope was a pre-flight
rule: predict from rho and the strategies' weights, before running any test,
how much log-vol separation a universe affords -- the quantity
`paired_estimand_mde` compares against its detection threshold. Checked against
the panels where the answer is already known, BOTH equicorrelation models fail,
and the second fails after being given the channel the first was missing:

| panel        | observed sep | diversification-only | + low-vol tilt |
|--------------|-------------:|---------------------:|---------------:|
| `full_2021`  |      +0.1565 |              +0.0854 |        +0.1177 |
| `etf_2017`   |      +0.0996 |              -0.0008 |        +0.0268 |

Both under-predict, `etf_2017` by a factor of four. The reason is structural
rather than a missing term: equicorrelation imposes ONE correlation on every
pair, and a variance minimiser's entire edge is in the deviations from that --
it seeks pairs that are mutually uncorrelated. A model that averages the
correlation matrix away has deleted the thing being optimised over, so no
repair of its volatility or weighting terms will fix it.

The hunt stops here rather than continuing into a third functional form. What
survives is DESCRIPTIVE, and it is enough for the decision that prompted it:

  - `deep_morocco`: effective N spans 6.67-12.00, a 1.80x spread. The
    portfolios differ; the weight cap is NOT collapsing them. Separation is
    nonetheless only +0.0678, below its own detection threshold.
  - `full_2021`: N_eff ratio 1.42x, separation +0.1565 -- found at |t| = 6.10.
  - `etf_2017`: N_eff ratio 1.00x, separation +0.0996. Identical effective
    diversification, substantial separation: here it comes entirely from WHICH
    assets are held, not how many.

So how much risk separation a universe affords is real, varies widely across
panels, and is not predicted by any simple structural summary computed here.
That is a reason to MEASURE it per panel with a positive control before
trusting a null, which is the protocol `docs/REACHABLE_CLAIMS.md` now carries,
and not a reason to trust a formula instead.

It reads committed Gold artifacts only: `dashboard_weights.parquet`,
`dashboard_equity.parquet` and the matching log-return panels. Nothing is
fitted, selected or re-run, so this costs no recompute and touches no frozen
test. `deep_morocco` is not here because its weights are not a committed
artifact; its row is produced by `deep_morocco_weight_concentration.py`.
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("risk_separation")

GOLD = ROOT / "data" / "gold"
OUT_JSON = GOLD / "risk_separation_bound.json"

PANELS = {
    "full_2021": "log_returns.parquet",
    "etf_2017": "log_returns_etf.parquet",
}
PAIR = ("max_sharpe", "min_variance_lw")   # higher-vol first, by construction


def _effective_n(w: np.ndarray) -> float:
    s = float(np.sum(w * w))
    return float(1.0 / s) if s > 0 else float("nan")


def main() -> dict:
    weights = pd.read_parquet(GOLD / "dashboard_weights.parquet")
    equity = pd.read_parquet(GOLD / "dashboard_equity.parquet")
    out = {
        "experiment": "risk_separation_bound",
        "finding": ("equicorrelation models under-predict achievable log-vol separation "
                    "on both checkable panels; the descriptive per-panel numbers stand, "
                    "the predictive rule does not"),
        "refuted_models": [
            "equicorrelation with a common asset volatility (diversification channel only)",
            "equicorrelation with per-asset volatilities (adds the low-vol tilt channel)",
        ],
        "why": ("equicorrelation imposes one correlation on every pair, deleting exactly "
                "the structure a variance minimiser optimises over"),
        "panels": {},
    }

    for universe, ret_file in PANELS.items():
        w_u = weights[weights["universe"] == universe]
        e_u = equity[equity["universe"] == universe]
        rets = pd.read_parquet(GOLD / ret_file)

        start = pd.Timestamp(w_u["Date"].min())
        simple = (np.exp(rets) - 1.0).loc[rets.index >= start]
        corr = simple.corr().to_numpy()
        iu = np.triu_indices_from(corr, k=1)
        rho = float(corr[iu].mean())
        asset_vol = simple.std(ddof=1) * np.sqrt(252)
        sigma = float(asset_vol.mean())

        panel = {"n_assets": int(rets.shape[1]),
                 "window_start": str(start.date()),
                 "n_days": int(len(simple)),
                 "mean_pairwise_correlation": round(rho, 4),
                 "mean_asset_vol_annual": round(sigma, 4),
                 "strategies": {}}
        log.info("%s: %d assets | rho %.4f | mean asset vol %.4f | from %s",
                 universe, rets.shape[1], rho, sigma, start.date())

        for strat, g in w_u.groupby("strategy"):
            piv = g.pivot_table(index="Date", columns="asset", values="weight",
                                aggfunc="last").fillna(0.0)
            piv = piv.reindex(columns=asset_vol.index, fill_value=0.0)
            W = piv.to_numpy(dtype=float)
            sv = asset_vol.to_numpy(dtype=float)
            eff = float(np.mean([_effective_n(r) for r in W]))
            # Exact equicorrelation variance, per rebalance, then averaged.
            diversification = (1.0 - rho) * np.sum((W * sv) ** 2, axis=1)
            tilt = rho * (W @ sv) ** 2
            pred = float(np.sqrt(np.mean(diversification + tilt)))
            # The tilt channel on its own: the weighted mean asset volatility.
            wavg_vol = float(np.mean(W @ sv))
            net = e_u[e_u["strategy"] == strat].set_index("Date")["net_return"]
            net = net.loc[net.index >= start]
            realised = float(net.std(ddof=1) * np.sqrt(252))
            panel["strategies"][strat] = {
                "effective_n": round(eff, 3),
                "weighted_avg_asset_vol": round(wavg_vol, 4),
                "realised_vol_annual": round(realised, 4),
                "predicted_vol_equicorr": round(pred, 4),
            }
            log.info("   %-20s N_eff %5.2f | w-avg asset vol %.4f | realised %.4f | predicted %.4f",
                     strat, eff, wavg_vol, realised, pred)

        hi, lo = PAIR
        if hi in panel["strategies"] and lo in panel["strategies"]:
            a, b = panel["strategies"][hi], panel["strategies"][lo]
            obs = float(np.log(a["realised_vol_annual"] / b["realised_vol_annual"]))
            prd = float(np.log(a["predicted_vol_equicorr"] / b["predicted_vol_equicorr"]))
            panel["positive_control"] = {
                "pair": f"{hi} vs {lo}",
                "effective_n_ratio": round(b["effective_n"] / a["effective_n"], 4),
                "tilt_log_ratio": round(float(np.log(
                    a["weighted_avg_asset_vol"] / b["weighted_avg_asset_vol"])), 4),
                "observed_log_vol_separation": round(obs, 4),
                "predicted_log_vol_separation": round(prd, 4),
                "predicted_over_observed": round(prd / obs, 4) if obs else None,
            }
            log.info("   CONTROL %s vs %s: N_eff ratio %.2fx | observed log-vol sep %+.4f "
                     "| predicted %+.4f", hi, lo,
                     b["effective_n"] / a["effective_n"], obs, prd)
        out["panels"][universe] = panel

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUT_JSON.name)
    return out


if __name__ == "__main__":
    main()
