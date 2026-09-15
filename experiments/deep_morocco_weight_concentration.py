"""
deep_morocco_weight_concentration.py — is the CAP doing the portfolio construction?

WHY THIS EXPERIMENT EXISTS. `docs/REACHABLE_CLAIMS.md` §5 explained the deep
panel's null by the absence of a variance minimiser from the family. Adding
`min_variance_lw` falsified that: the `max_sharpe` vs `min_variance_lw` log
volatility contrast is true BY CONSTRUCTION and was §3's positive control at
|t| = 6.10 on `full_2021`, yet lands at |t| = 2.14 on the deep panel, with a
detectability ratio of 0.86 -- it fails as a single pre-specified test before
any correction.

The replacement explanation is that all six strategies realise annualised
volatility inside 0.1272-0.1422, a 1.12x band, because a long-only book of 12
names capped at MAX_W = 0.20 has a floor of five effective positions and the
constraint, not the objective, picks the portfolio.

That explanation was INFERRED from the volatility compression. §5 was wrong in
exactly that way -- it inferred a mechanism instead of measuring one -- so this
measures it: effective N (inverse Herfindahl) and how often the cap actually
binds, per strategy, on the realised target weights.

The measurement is genuinely two-sided, which is why it is worth running.
Mean pairwise correlation is only 0.176 on this panel, low enough that a
minimiser SHOULD be able to separate. If effective N is pinned near the cap
floor for every strategy, the constraint explanation holds. If the weights
differ sharply while volatility does not, the explanation is wrong and the
interesting finding is that weight dispersion does not translate into risk
dispersion here -- which would need saying before anything is written.

SCOPE. The four non-ML strategies only. They re-run in seconds; `rf_tuned` and
`xgb_tuned` cost about half an hour and cannot change the verdict, since the
decisive contrast is the by-construction one between `max_sharpe` and
`min_variance_lw`. Their exclusion is a cost decision, stated rather than
hidden, and it is recorded in the artifact.

Nothing here selects anything. It re-runs committed strategies through the
unchanged backtest and describes the weights they produced.
"""
from __future__ import annotations

import importlib.util
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

from backtest import run_backtest
from strategies import EqualWeight, MaxSharpe, MinVarianceLW, RegimeConditionalStrategy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("deep_morocco_weights")

OUT_JSON = ROOT / "data" / "gold" / "deep_morocco_weight_concentration.json"
CAP_TOL = 1e-6


def _starvation():
    """Import the experiment module by path; `experiments/` is not a package."""
    path = ROOT / "experiments" / "deep_morocco_starvation.py"
    spec = importlib.util.spec_from_file_location("deep_morocco_starvation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _effective_n(w: np.ndarray) -> float:
    """Inverse Herfindahl. Equals the asset count when equally weighted, and 1
    when a single name holds everything, so it reads as 'how many positions is
    this portfolio really taking'."""
    s = float(np.sum(w * w))
    return float(1.0 / s) if s > 0 else float("nan")


def main() -> dict:
    dm = _starvation()
    log_ret, features = dm.load_universe()
    split = int(round(len(log_ret) * (1 - dm.TEST_FRAC)))
    test_start = log_ret.index[split]
    n_assets = log_ret.shape[1]
    cap = dm.MAX_W
    floor_positions = int(np.ceil(1.0 / cap))
    log.info("UNIVERSE: %d assets | cap %.2f | long-only floor = %d positions | equal-weight N = %d",
             n_assets, cap, floor_positions, n_assets)

    strategies = {
        "regime_conditional": RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=cap),
            bear_strategy=MinVarianceLW(max_weight=cap)),
        "equal_weight": EqualWeight(),
        "max_sharpe": MaxSharpe(max_weight=cap),
        "min_variance_lw": MinVarianceLW(max_weight=cap),
    }

    rows = {}
    for name, strat in strategies.items():
        r = run_backtest(log_ret, strat, rebalance_freq="ME", min_train_days=252,
                         cost_bps=dm.BVC_COST_BPS, extras={"features": features},
                         universe_name="deep_morocco_weights", max_weight=cap)
        w = r.target_weights.loc[r.target_weights.index >= test_start]
        arr = w.to_numpy(dtype=float)
        eff = np.array([_effective_n(row) for row in arr])
        at_cap = (arr >= cap - CAP_TOL).sum(axis=1)
        rows[name] = {
            "n_rebalances": int(len(w)),
            "effective_n_mean": round(float(eff.mean()), 3),
            "effective_n_min": round(float(eff.min()), 3),
            "effective_n_max": round(float(eff.max()), 3),
            "names_at_cap_mean": round(float(at_cap.mean()), 3),
            "share_of_weight_at_cap": round(float(
                arr[arr >= cap - CAP_TOL].sum() / arr.sum()), 4),
            "max_weight_observed": round(float(arr.max()), 4),
            "rebalances_with_any_name_at_cap": round(float((at_cap > 0).mean()), 4),
        }
        test_net = r.net_returns.loc[r.net_returns.index >= test_start]
        rows[name]["realised_vol_annual"] = round(
            float(test_net.std(ddof=1) * np.sqrt(252)), 4)
        log.info("%-20s eff_N %.2f [%.2f-%.2f] | %.2f names at cap | %.1f%% capped | vol %.4f",
                 name, rows[name]["effective_n_mean"], rows[name]["effective_n_min"],
                 rows[name]["effective_n_max"], rows[name]["names_at_cap_mean"],
                 100 * rows[name]["share_of_weight_at_cap"],
                 rows[name]["realised_vol_annual"])

    # WHY weight dispersion need not become risk dispersion. Under an
    # equicorrelation approximation -- every pair at the panel's mean
    # correlation rho, every asset at the panel's mean volatility sigma --
    # portfolio variance is sigma^2 * (rho + (1 - rho) / N_eff). The common
    # factor sets a floor no amount of diversification can go below, so once
    # rho is well above zero, large changes in N_eff buy small changes in vol.
    # This is a PREDICTION: it is computed from rho and N_eff alone, with no
    # reference to the realised series, and then compared against it.
    test_ret = log_ret.loc[log_ret.index >= test_start]
    simple = np.exp(test_ret) - 1.0
    corr = simple.corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    rho = float(corr[iu].mean())
    sigma = float((simple.std(ddof=1) * np.sqrt(252)).mean())
    log.info("PANEL: mean pairwise correlation %.4f | mean asset vol %.4f", rho, sigma)
    for name, v in rows.items():
        pred = sigma * float(np.sqrt(rho + (1.0 - rho) / v["effective_n_mean"]))
        v["predicted_vol_equicorr"] = round(pred, 4)
        v["predicted_over_realised"] = round(pred / v["realised_vol_annual"], 4)
        log.info("   %-20s N_eff %.2f -> predicted vol %.4f vs realised %.4f (%.3fx)",
                 name, v["effective_n_mean"], pred, v["realised_vol_annual"],
                 v["predicted_over_realised"])

    n_hi, n_lo = rows["min_variance_lw"], rows["max_sharpe"]
    eff_ratio = n_hi["effective_n_mean"] / n_lo["effective_n_mean"]
    vol_ratio = n_lo["realised_vol_annual"] / n_hi["realised_vol_annual"]
    pred_ratio = n_lo["predicted_vol_equicorr"] / n_hi["predicted_vol_equicorr"]
    log.info("TRANSLATION: min_variance_lw holds %.2fx the effective names of max_sharpe, "
             "and that buys a vol ratio of %.4f (equicorrelation predicts %.4f)",
             eff_ratio, vol_ratio, pred_ratio)

    effs = {k: v["effective_n_mean"] for k, v in rows.items()}
    spread = max(effs.values()) / min(effs.values())
    verdict = ("cap-bound" if spread < 1.25 else "weights-differ")
    log.info("SPREAD in effective N: %.3fx (max %s, min %s) -> %s",
             spread, max(effs, key=effs.get), min(effs, key=effs.get), verdict)

    out = {
        "experiment": "deep_morocco_weight_concentration",
        "universe": "deep_morocco",
        "n_assets": int(n_assets),
        "max_weight_cap": cap,
        "long_only_floor_positions": floor_positions,
        "test_start": str(test_start.date()),
        "strategies": rows,
        "effective_n_spread": round(float(spread), 4),
        "verdict": verdict,
        "equicorrelation": {
            "mean_pairwise_correlation": round(rho, 4),
            "mean_asset_vol_annual": round(sigma, 4),
            "effective_n_ratio_minvar_over_maxsharpe": round(float(eff_ratio), 4),
            "realised_vol_ratio_maxsharpe_over_minvar": round(float(vol_ratio), 4),
            "predicted_vol_ratio_maxsharpe_over_minvar": round(float(pred_ratio), 4),
        },
        "excluded": {
            "strategies": ["rf_tuned", "xgb_tuned"],
            "reason": ("about 30 minutes of refitting; the decisive contrast is the "
                       "by-construction max_sharpe vs min_variance_lw pair, which is cheap"),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUT_JSON.name)
    return out


if __name__ == "__main__":
    main()
