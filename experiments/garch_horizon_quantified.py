"""
garch_horizon_quantified.py — how large are Limit #4's two defects?

WHY THIS EXPERIMENT EXISTS. `docs/EVALUATION_LIMITS.md` Limit #4 diagnoses two
defects in `dcc_garch._dcc_covariance_uncached` by inspection and closes with
"Diagnosed, **not quantified**... Stated as an open question rather than a
finding." This measures both on the real panels, which is the step that turns
them into findings or dismisses them.

  OFF-BY-ONE (line 234). `sigma_t = sigmas[-1]` is conditional on information
  through tau-1. The wanted object is the one-step forecast
  omega + alpha r_tau^2 + beta sigma_tau^2.

  HORIZON MISMATCH (line 236). A one-day variance is scaled by 252 and fed to
  an optimizer holding for ~21 days, while GARCH mean-reverts across that hold.

Both are measured per asset from the SAME fitted GARCH parameters, so the
comparison isolates the two corrections from any difference in estimation.

MEASURED ACROSS EVERY DATE, not at one. The first version of this script
evaluated the corrections at the final observation only and reported a combined
effect of about 2% on volatility. That is a misleading way to size this defect:
the horizon correction pulls the one-step variance toward the unconditional
level, so its magnitude depends entirely on how far from that level the market
happens to be on the chosen day, and it is near zero on a day that is already
average. The correction exists for the days that are NOT average -- after a
shock, which is when a risk model earns its place. So the ratio is computed at
every date and reported as a distribution, with the high-volatility decile
called out separately.

The horizon correction is the closed form in `src/garch_horizon.py`, checked
against the explicit forecast recursion it summarises -- 500 random parameter
draws, zero mismatches -- before any number here is read.

SCOPE. This covers the variance channel. DCC's correlation matrix mean-reverts
over the same horizon and a fully consistent multi-period covariance would
aggregate it too; the DCC persistence a + b is reported alongside so the size of
that unimplemented second correction is visible rather than buried.
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

from garch_horizon import (
    aggregation_factor, multi_horizon_average_variance, one_step_forecast_variance,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("garch_horizon")

GOLD = ROOT / "data" / "gold"
OUT_JSON = GOLD / "garch_horizon_quantified.json"

PANELS = {"full_2021": "log_returns.parquet", "etf_2017": "log_returns_etf.parquet"}
HOLD_DAYS = 21          # the ~monthly rebalance the optimizer actually holds for
RESCALE = 100.0         # arch fits better on percent returns


def _fit_garch(series: pd.Series) -> dict | None:
    from arch import arch_model

    scaled = series.to_numpy() * RESCALE
    res = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", rescale=False).fit(
        disp="off", show_warning=False)
    if res.convergence_flag != 0:
        return None
    omega = float(res.params["omega"]) / RESCALE ** 2
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])
    cond_var = (np.asarray(res.conditional_volatility) / RESCALE) ** 2
    return {"omega": omega, "alpha": alpha, "beta": beta,
            "cond_var": cond_var, "returns": series.to_numpy()}


def main() -> dict:
    out = {"experiment": "garch_horizon_quantified", "hold_days": HOLD_DAYS, "panels": {}}

    for panel, fname in PANELS.items():
        rets = pd.read_parquet(GOLD / fname)
        assets, skipped = {}, []
        for col in rets.columns:
            fit = _fit_garch(rets[col].dropna())
            if fit is None:
                skipped.append(col)
                continue
            p = fit["alpha"] + fit["beta"]
            stale = float(fit["cond_var"][-1])
            one_step = one_step_forecast_variance(
                fit["omega"], fit["alpha"], fit["beta"],
                float(fit["returns"][-1]), float(fit["cond_var"][-1]))
            aggregated = multi_horizon_average_variance(
                fit["omega"], fit["alpha"], fit["beta"], one_step, HOLD_DAYS)

            # The same two corrections at EVERY date, not just the last one.
            cv = fit["cond_var"]
            r = fit["returns"]
            step_all = fit["omega"] + fit["alpha"] * r[:-1] ** 2 + fit["beta"] * cv[:-1]
            stale_all = cv[:-1]
            pers = min(p, 1.0)
            if pers >= 1.0:
                agg_all = step_all.copy()
            else:
                uncond = fit["omega"] / (1.0 - pers)
                agg_all = uncond + (step_all - uncond) * aggregation_factor(pers, HOLD_DAYS)
            horizon_ratio_all = np.sqrt(agg_all / step_all)
            combined_ratio_all = np.sqrt(agg_all / stale_all)
            hi = stale_all >= np.quantile(stale_all, 0.9)
            assets[col] = {
                "alpha": round(fit["alpha"], 4), "beta": round(fit["beta"], 4),
                "persistence": round(p, 4),
                "aggregation_factor": round(aggregation_factor(min(p, 1.0), HOLD_DAYS), 4),
                "vol_stale_annual": round(float(np.sqrt(stale * 252)), 4),
                "vol_one_step_annual": round(float(np.sqrt(one_step * 252)), 4),
                "vol_aggregated_annual": round(float(np.sqrt(aggregated * 252)), 4),
                "off_by_one_vol_ratio": round(float(np.sqrt(one_step / stale)), 4),
                "horizon_vol_ratio": round(float(np.sqrt(aggregated / one_step)), 4),
                "combined_vol_ratio": round(float(np.sqrt(aggregated / stale)), 4),
                "horizon_ratio_all_dates_mean": round(float(horizon_ratio_all.mean()), 4),
                "horizon_ratio_all_dates_p05": round(float(np.quantile(horizon_ratio_all, 0.05)), 4),
                "horizon_ratio_all_dates_p95": round(float(np.quantile(horizon_ratio_all, 0.95)), 4),
                "horizon_ratio_high_vol_decile": round(float(horizon_ratio_all[hi].mean()), 4),
                "combined_ratio_all_dates_mean": round(float(combined_ratio_all.mean()), 4),
                "combined_ratio_high_vol_decile": round(float(combined_ratio_all[hi].mean()), 4),
                "max_abs_horizon_deviation_pct": round(
                    float(100 * np.max(np.abs(horizon_ratio_all - 1.0))), 2),
            }

        if not assets:
            log.warning("%s: no asset converged, skipping", panel)
            continue

        frame = pd.DataFrame(assets).T
        summary = {
            "n_assets": len(assets), "skipped": skipped,
            "persistence_mean": round(float(frame["persistence"].mean()), 4),
            "persistence_min": round(float(frame["persistence"].min()), 4),
            "persistence_max": round(float(frame["persistence"].max()), 4),
            "aggregation_factor_mean": round(float(frame["aggregation_factor"].mean()), 4),
            "off_by_one_vol_ratio": {
                "mean": round(float(frame["off_by_one_vol_ratio"].mean()), 4),
                "min": round(float(frame["off_by_one_vol_ratio"].min()), 4),
                "max": round(float(frame["off_by_one_vol_ratio"].max()), 4)},
            "horizon_vol_ratio": {
                "mean": round(float(frame["horizon_vol_ratio"].mean()), 4),
                "min": round(float(frame["horizon_vol_ratio"].min()), 4),
                "max": round(float(frame["horizon_vol_ratio"].max()), 4)},
            "combined_vol_ratio": {
                "mean": round(float(frame["combined_vol_ratio"].mean()), 4),
                "min": round(float(frame["combined_vol_ratio"].min()), 4),
                "max": round(float(frame["combined_vol_ratio"].max()), 4)},
            "horizon_all_dates": {
                "mean": round(float(frame["horizon_ratio_all_dates_mean"].mean()), 4),
                "p05": round(float(frame["horizon_ratio_all_dates_p05"].mean()), 4),
                "p95": round(float(frame["horizon_ratio_all_dates_p95"].mean()), 4),
                "high_vol_decile": round(float(frame["horizon_ratio_high_vol_decile"].mean()), 4),
                "max_abs_deviation_pct": round(
                    float(frame["max_abs_horizon_deviation_pct"].max()), 2)},
            "combined_all_dates": {
                "mean": round(float(frame["combined_ratio_all_dates_mean"].mean()), 4),
                "high_vol_decile": round(float(frame["combined_ratio_high_vol_decile"].mean()), 4)},
        }
        out["panels"][panel] = {"summary": summary, "assets": assets}

        log.info("%s: %d assets | persistence %.3f [%.3f-%.3f] | A(p,21) %.3f",
                 panel, len(assets), summary["persistence_mean"],
                 summary["persistence_min"], summary["persistence_max"],
                 summary["aggregation_factor_mean"])
        log.info("   off-by-one  vol x %.4f [%.3f-%.3f]  (fixing it moves the input this much)",
                 summary["off_by_one_vol_ratio"]["mean"],
                 summary["off_by_one_vol_ratio"]["min"], summary["off_by_one_vol_ratio"]["max"])
        log.info("   horizon     vol x %.4f [%.3f-%.3f]  (1-day -> 21-day average)",
                 summary["horizon_vol_ratio"]["mean"],
                 summary["horizon_vol_ratio"]["min"], summary["horizon_vol_ratio"]["max"])
        log.info("   combined    vol x %.4f [%.3f-%.3f]",
                 summary["combined_vol_ratio"]["mean"],
                 summary["combined_vol_ratio"]["min"], summary["combined_vol_ratio"]["max"])
        log.info("   ACROSS ALL DATES  horizon ratio mean %.4f, 5-95%% [%.3f, %.3f], "
                 "high-vol decile %.4f, max deviation %.1f%%",
                 summary["horizon_all_dates"]["mean"], summary["horizon_all_dates"]["p05"],
                 summary["horizon_all_dates"]["p95"],
                 summary["horizon_all_dates"]["high_vol_decile"],
                 summary["horizon_all_dates"]["max_abs_deviation_pct"])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUT_JSON.name)
    return out


if __name__ == "__main__":
    main()
