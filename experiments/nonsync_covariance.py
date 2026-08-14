"""
nonsync_covariance.py — How sensitive is the covariance input to
non-synchronous trading between Casablanca and New York?

WHY THIS EXPERIMENT EXISTS. The covariance matrix is the single input every
strategy in this project consumes: `min_variance`, `min_variance_lw`,
`min_variance_ewma`, `dcc_garch`, and the Sharpe objective behind `max_sharpe`
and both F7 signal models. The whole Phase 4 ablation ladder is an argument
about how best to ESTIMATE it. That argument silently assumes the quantity
being estimated is the one we mean.

It may not be. The Casablanca Stock Exchange closes hours before New York, and
`align_calendars` forward-fills BVC prices across Moroccan holidays. Both push
in the same direction: a daily BVC return reflects information that US markets
priced the day BEFORE. If so, a same-day covariance understates BVC-vs-ETF
co-movement, the optimizer sees Moroccan assets as nearly uncorrelated with
everything, and diversification into Morocco looks cheaper than it is —
directly touching P1 (noisy covariance) and P3 (diversification breakdown).

An external review raised this; nothing in the repo had ever measured it. This
experiment does, in two stages.

  Stage A — DIAGNOSIS. Measure the asynchrony signature directly: lead/lag
    cross-correlation against SPY, AR(1), zero-return-day share, and the
    Lo-MacKinlay variance ratio. Includes a CONTROL: the same measurements on
    `etf_2017`, which is five US-listed ETFs trading on one calendar. If the
    signature appears there too, it is not asynchrony and the diagnosis is
    wrong.

  Stage B — PROPAGATION. Re-estimate the covariance three ways and run each
    through the UNMODIFIED walk-forward engine, so the comparison is
    apples-to-apples down to the cost model:
      1. `daily_lw`   — Ledoit-Wolf on daily returns (what production does)
      2. `weekly_lw`  — Ledoit-Wolf on weekly returns, rescaled
      3. `dimson_lw`  — daily Ledoit-Wolf plus the lead-lag autocovariance
                        term, Sigma + Sigma_1 + Sigma_1', PSD-projected
    Then report how far the ALLOCATIONS move, not just the Sharpe.

WHAT THIS CAN AND CANNOT SHOW. It measures how much the covariance input, and
the allocation built on it, depend on a market-calendar and liquidity effect
rather than on economics. That is a statement about the SENSITIVITY of the
input and therefore about what can be inferred from the covariance-model
ladder. It is NOT a bound on achievable performance, and no result here says
what any model could or could not have achieved — there is no such estimate in
this file and none may be quoted from it.

PRE-REGISTERED OUTCOMES, fixed before the run so nothing can be rationalised
afterwards:

  A. The signature is present on `full_2021`, absent on the control, AND the
     three estimators produce materially different allocations and Sharpes.
     -> The covariance input is a first-order sensitivity. Every ladder
        conclusion is conditioned on a measurement choice nobody made
        deliberately, and that belongs in the limits of the report.
  B. The signature is present and allocations move, but net Sharpe does not.
     -> The bias is real but the 25% cap absorbs it, consistent with the
        finding that the constraint out-regularises every covariance model
        tried (AGENTS.md 10.1). The ladder's conclusions survive; the reason
        they survive is the constraint, not the estimator.
  C. The signature is present but neither allocations nor Sharpe move.
     -> Asynchrony exists in the correlations and does not propagate. Worth
        stating once and closing.
  D. The signature appears on the CONTROL too.
     -> The diagnosis is wrong: this is not a calendar effect. Report as a
        refutation of the premise.

DISCLOSURE ON STAGE C. Stages A and B, and the outcome rule above, were fixed
before any number existed. Stage C — the paired block bootstrap of the
DIFFERENCE — was added AFTER Stage A and B results had been seen, prompted by
this project's own standing note that a paired test is the missing instrument
and that overlapping marginal intervals license nothing in either direction
(AGENTS.md 5.2, 12I.1). Its result was not known when it was added, and it does
not feed the outcome letter: outcome A turns on the allocation sensitivity,
which is measured directly. It is recorded here rather than folded in silently.

A NOTE ON THE WEEKLY ARM, stated up front because it is a real confound rather
than a footnote. Weekly returns over the same calendar window carry one fifth
the observations, so `weekly_lw` trades a synchronisation bias for estimation
variance. It is not a strictly better estimator and is not offered as one; it
is the standard diagnostic for whether the daily figure is calendar-distorted.
`dimson_lw` is the arm that keeps the daily sample size, which is why both are
run rather than either alone.

Usage:
    .venv/bin/python experiments/nonsync_covariance.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from backtest import build_cost_vector, run_backtest
from metrics import (
    annualized_sharpe,
    block_bootstrap_sharpe_ci,
    max_drawdown,
    paired_block_bootstrap,
)
from provenance import build_provenance
from strategies import Strategy, TRADING_DAYS_PER_YEAR, _optimize_weights
from utils import load_params

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("nonsync")

WEEKS_PER_YEAR = 52
# Friday-anchored weeks. Log returns are additive, so a weekly return is the
# SUM of its daily returns -- no compounding approximation enters here.
WEEKLY_RULE = "W-FRI"

# The reference asset for lead/lag: the most liquid, earliest-closing-last
# market in the panel. SPY is in BOTH universes, which is what makes the
# control comparable.
REFERENCE = "SPY"

# Assets whose exchange closes before New York. Declared explicitly rather than
# inferred from a ".CS" suffix so that adding a third venue cannot silently
# inherit the wrong assumption.
NON_US_ASSETS = ("IAM.CS", "ATW.CS", "CIH.CS", "BCP.CS")

# An allocation difference below this is rounding, not a decision change.
WEIGHT_EPS = 1e-4


# ---------------------------------------------------------------- Stage A


def _weekly(returns: pd.DataFrame) -> pd.DataFrame:
    """Sum daily log returns into Friday-anchored weeks."""
    return returns.resample(WEEKLY_RULE).sum()


def asynchrony_diagnostics(returns: pd.DataFrame, universe: str) -> list[dict]:
    """
    Measure the stale-price / non-synchronous-trading signature per asset.

    Addresses: P1, P3 — the covariance every strategy consumes is built from
    these same series. If a daily return reflects information the reference
    market priced yesterday, the same-day covariance is measuring the wrong
    quantity, however well it is estimated.

    Returns:
        One row per asset. `corr_lag1` is the asset TODAY against the
        reference YESTERDAY; under synchronous trading it should be ~0 and
        far below `corr_same_day`.
    """
    if REFERENCE not in returns.columns:
        raise KeyError(f"{REFERENCE!r} absent from {universe}; the lead/lag arm needs it.")

    ref = returns[REFERENCE]
    weekly = _weekly(returns)
    rows: list[dict] = []

    for col in returns.columns:
        r = returns[col]
        daily_var = float(r.var())
        weekly_var = float(weekly[col].var())
        # Lo-MacKinlay variance ratio. VR == 1 under no autocorrelation;
        # VR < 1 is the bid-ask-bounce signature (daily vol OVERSTATES),
        # VR > 1 is partial adjustment / stale prices.
        variance_ratio = (weekly_var / 5.0) / daily_var if daily_var > 0 else float("nan")

        rows.append({
            "asset": col,
            "is_non_us": col in NON_US_ASSETS,
            "corr_same_day": round(float(r.corr(ref)), 4),
            # Asset today vs reference YESTERDAY -- the asynchrony term.
            "corr_lag1": round(float(r.corr(ref.shift(1))), 4),
            # Asset today vs reference TOMORROW -- should be ~0 for everyone;
            # a large value here would mean something worse than asynchrony.
            "corr_lead1": round(float(r.corr(ref.shift(-1))), 4),
            "ar1": round(float(r.autocorr(1)), 4),
            "pct_zero_days": round(float((r.abs() < 1e-12).mean()), 4),
            "vol_daily_annualised": round(float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), 4),
            "vol_weekly_scaled": round(
                float(weekly[col].std() * np.sqrt(WEEKS_PER_YEAR)), 4),
            "variance_ratio_5d": round(float(variance_ratio), 4),
        })
    return rows


def summarise_diagnostics(rows: list[dict]) -> dict:
    """Collapse Stage A into the one number the outcome rule turns on."""
    def _mean(subset: list[dict], key: str) -> float:
        vals = [r[key] for r in subset if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    # The reference cannot be lead/lagged against itself meaningfully.
    others = [r for r in rows if r["asset"] != REFERENCE]
    non_us = [r for r in others if r["is_non_us"]]
    us = [r for r in others if not r["is_non_us"]]

    def _block(subset: list[dict]) -> dict | None:
        if not subset:
            return None
        same = _mean(subset, "corr_same_day")
        lag = _mean(subset, "corr_lag1")
        return {
            "n_assets": len(subset),
            "mean_corr_same_day": round(same, 4),
            "mean_corr_lag1": round(lag, 4),
            # The headline: >1 means yesterday's reference explains today's
            # return better than today's does.
            "lag1_over_same_day": round(lag / same, 3) if abs(same) > 1e-9 else None,
            "mean_ar1": round(_mean(subset, "ar1"), 4),
            "mean_pct_zero_days": round(_mean(subset, "pct_zero_days"), 4),
            "mean_variance_ratio_5d": round(_mean(subset, "variance_ratio_5d"), 4),
        }

    return {"non_us": _block(non_us), "us_listed": _block(us)}


# ---------------------------------------------------- Stage B: estimators


def _nearest_psd(cov: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Clip negative eigenvalues to zero, symmetrising first.

    Adding a lead-lag term to a shrunk covariance is not guaranteed to leave a
    positive semi-definite matrix, and SLSQP on an indefinite quadratic can
    return a direction of unbounded descent rather than a portfolio. The
    projection is reported, never silent: `psd_projected` travels into the
    result artifact so a reader can see how often the correction needed
    rescuing.
    """
    sym = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    if (eigvals >= -1e-12).all():
        return sym, False
    clipped = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(clipped) @ eigvecs.T, True


def cov_daily_lw(train: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Production's estimator: Ledoit-Wolf on daily returns."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(train.to_numpy())
    return lw.covariance_ * TRADING_DAYS_PER_YEAR, False


def cov_weekly_lw(train: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """
    Ledoit-Wolf on weekly returns, annualised by 52.

    A week is long enough that Casablanca and New York have both traded on the
    same information, so the synchronisation bias largely disappears. The cost
    is one fifth the observations -- see the module docstring; this arm is a
    diagnostic, not a proposed replacement estimator.
    """
    from sklearn.covariance import LedoitWolf

    weekly = _weekly(train).dropna(how="all")
    if len(weekly) < max(8, train.shape[1] + 1):
        # Too thin to say anything; fall back rather than emit a rank-deficient
        # matrix, and let the caller record that it happened.
        return cov_daily_lw(train)[0], True
    lw = LedoitWolf().fit(weekly.to_numpy())
    return lw.covariance_ * WEEKS_PER_YEAR, False


def cov_dimson_lw(train: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """
    Daily Ledoit-Wolf plus the first-order lead-lag term.

    The matrix analogue of Dimson (1979) aggregated coefficients: with stale
    prices, part of the true co-movement sits in the cross-autocovariance
    rather than the contemporaneous term, so

        Sigma_adj = Sigma_0 + Sigma_1 + Sigma_1'

    recovers it while keeping the daily sample size. Identical in spirit to a
    Newey-West estimator truncated at one lag, which is the right truncation
    here: the hypothesis is a one-session offset between two exchanges, not
    long-memory dependence.
    """
    from sklearn.covariance import LedoitWolf

    x = train.to_numpy()
    lw = LedoitWolf().fit(x)
    sigma0 = lw.covariance_

    centred = x - x.mean(axis=0, keepdims=True)
    # Sigma_1[i, j] = Cov(r_i,t , r_j,t-1)
    sigma1 = (centred[1:].T @ centred[:-1]) / (len(centred) - 1)

    adjusted, projected = _nearest_psd(sigma0 + sigma1 + sigma1.T)
    return adjusted * TRADING_DAYS_PER_YEAR, projected


ESTIMATORS = {
    "daily_lw": cov_daily_lw,
    "weekly_lw": cov_weekly_lw,
    "dimson_lw": cov_dimson_lw,
}


class _MinVarianceWith(Strategy):
    """
    Minimum variance under a swappable covariance estimator.

    Addresses: P1 — the ONLY thing that varies across the three arms is the
    covariance function. Objective, cap, optimizer, engine, cost model and
    rebalance calendar are the production ones, so any difference in the
    result is attributable to the estimator and nothing else.

    Deliberately defined here rather than in `src/`: this is a measurement
    instrument for one experiment, not a strategy anyone should be able to
    select in production.
    """

    def __init__(self, estimator: str, max_weight: float = 0.25) -> None:
        if estimator not in ESTIMATORS:
            raise ValueError(f"Unknown estimator {estimator!r}; expected {list(ESTIMATORS)}.")
        self.estimator = estimator
        self.max_weight = max_weight
        self.name = f"min_variance_{estimator}"
        self.psd_projections = 0
        self.degenerate_windows = 0

    def fit(
        self,
        train_returns: pd.DataFrame,
        extras: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.Series:
        cov, flagged = ESTIMATORS[self.estimator](train_returns)
        if flagged:
            if self.estimator == "dimson_lw":
                self.psd_projections += 1
            else:
                self.degenerate_windows += 1
        return _optimize_weights(
            lambda w: float(w @ cov @ w), train_returns.columns,
            self.max_weight, self.name, n_training_rows=len(train_returns),
        )


# ---------------------------------------------------------------- driver


def allocation_distance(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """
    How far apart are two allocation paths, in units a manager would feel?

    Sharpe can be unchanged while the portfolio is entirely different, and the
    reverse. Reporting only the Sharpe would hide exactly the sensitivity this
    experiment exists to measure.
    """
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"n_rebalances": 0}
    diff = (a.loc[common] - b.loc[common]).abs()
    # L1/2 is the fraction of the portfolio that would have to be traded to
    # move from one allocation to the other.
    per_date = diff.sum(axis=1) / 2.0
    return {
        "n_rebalances": int(len(common)),
        "mean_turnover_to_switch": round(float(per_date.mean()), 4),
        "max_turnover_to_switch": round(float(per_date.max()), 4),
        "max_single_weight_change": round(float(diff.to_numpy().max()), 4),
        "pct_rebalances_materially_different": round(
            float((per_date > WEIGHT_EPS).mean()), 4),
    }


def run_universe(universe: str, returns_path: str, params: dict) -> dict:
    bt = params["backtest"]
    boot = params["phase5"]["bootstrap"]
    returns = pd.read_parquet(ROOT / returns_path)

    log.info("")
    log.info("=" * 72)
    log.info("%s: %s -> %s, %d rows, %d assets",
             universe, returns.index.min().date(), returns.index.max().date(),
             len(returns), returns.shape[1])
    log.info("=" * 72)

    # --- Stage A -------------------------------------------------------
    diagnostics = asynchrony_diagnostics(returns, universe)
    summary = summarise_diagnostics(diagnostics)

    log.info("")
    log.info("STAGE A — asynchrony signature vs %s", REFERENCE)
    log.info("  %-9s %9s %9s %9s %8s %8s %8s",
             "asset", "corr_t", "corr_t-1", "corr_t+1", "AR(1)", "%zero", "VR(5d)")
    for row in diagnostics:
        if row["asset"] == REFERENCE:
            continue
        log.info("  %-9s %9.4f %9.4f %9.4f %8.4f %7.1f%% %8.3f",
                 row["asset"], row["corr_same_day"], row["corr_lag1"],
                 row["corr_lead1"], row["ar1"], 100 * row["pct_zero_days"],
                 row["variance_ratio_5d"])
    for group, block in summary.items():
        if block:
            log.info("  [%s] mean corr t=%.4f  t-1=%.4f  ratio=%s  VR=%.3f",
                     group, block["mean_corr_same_day"], block["mean_corr_lag1"],
                     block["lag1_over_same_day"], block["mean_variance_ratio_5d"])

    # --- Stage B -------------------------------------------------------
    cost_vector = build_cost_vector(
        list(returns.columns),
        etf_cost_bps=bt["costs_bps"]["etf"], bvc_cost_bps=bt["costs_bps"]["bvc"],
    )

    log.info("")
    log.info("STAGE B — same engine, same cap, three covariance estimators")
    arms: dict[str, dict] = {}
    weight_paths: dict[str, pd.DataFrame] = {}
    net_paths: dict[str, pd.Series] = {}

    for name in ESTIMATORS:
        strat = _MinVarianceWith(name, max_weight=bt["max_weight"])
        result = run_backtest(
            returns, strat, rebalance_freq=bt["rebalance_freq"],
            min_train_days=bt["min_train_days"], cost_bps=cost_vector,
            universe_name=f"nonsync_{universe}", max_weight=bt["max_weight"],
        )
        net = result.net_returns
        point, lo, hi = block_bootstrap_sharpe_ci(
            net, block_len=boot["block_len"], n_boot=boot["n_boot"],
            alpha=boot["alpha"], seed=boot["seed"])
        weight_paths[name] = result.target_weights
        net_paths[name] = net
        arms[name] = {
            "sharpe_net": round(float(annualized_sharpe(net, bt["risk_free_annual"])), 4),
            "sharpe_gross": round(
                float(annualized_sharpe(result.gross_returns, bt["risk_free_annual"])), 4),
            "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4),
            "max_drawdown": round(float(max_drawdown(net)), 4),
            "avg_turnover": round(float(result.turnover.mean()), 4),
            "n_rebalances": int(len(result.target_weights)),
            "psd_projections": strat.psd_projections,
            "degenerate_windows": strat.degenerate_windows,
        }
        log.info("  %-22s net %.4f  CI [%+.2f, %+.2f]  turnover %.3f  "
                 "psd-proj %d  thin %d",
                 strat.name, arms[name]["sharpe_net"], lo, hi,
                 arms[name]["avg_turnover"], strat.psd_projections,
                 strat.degenerate_windows)

    # --- how far did the ALLOCATIONS move? -----------------------------
    log.info("")
    log.info("  allocation distance from the production estimator (daily_lw):")
    distances = {}
    for name in ("weekly_lw", "dimson_lw"):
        distances[name] = allocation_distance(weight_paths["daily_lw"], weight_paths[name])
        d = distances[name]
        log.info("    %-12s mean %.4f  max %.4f  max single weight %.4f  "
                 "%.0f%% of rebalances differ",
                 name, d["mean_turnover_to_switch"], d["max_turnover_to_switch"],
                 d["max_single_weight_change"],
                 100 * d["pct_rebalances_materially_different"])

    # --- Stage C: is the DIFFERENCE testable, not just visible? --------
    # Marginal CIs on three arms would only let this experiment say "the point
    # estimates differ and the intervals overlap", which is the unlicensed
    # phrasing AGENTS.md 5.2 rules out in both directions. The arms share every
    # market day, so the paired test is both the correct instrument and a much
    # more powerful one.
    log.info("")
    log.info("STAGE C — paired block bootstrap of the DIFFERENCE vs daily_lw")
    paired: dict[str, dict] = {}
    for name in ("weekly_lw", "dimson_lw"):
        cmp_ = paired_block_bootstrap(
            candidate=net_paths[name], benchmark=net_paths["daily_lw"],
            block_len=boot["block_len"], n_boot=boot["n_boot"],
            alpha=boot["alpha"], risk_free_annual=bt["risk_free_annual"],
            seed=boot["seed"],
        )
        paired[name] = cmp_
        lo, hi = cmp_["sharpe_diff_ci"]
        log.info("  %-12s d_Sharpe %+.4f  CI [%+.4f, %+.4f]  p=%.4f  P(diff>0)=%.3f",
                 name, cmp_["sharpe_diff"], lo, hi,
                 cmp_["p_value_no_outperformance"],
                 cmp_["prob_sharpe_diff_positive"])

    return {
        "universe": universe,
        "stage_a_per_asset": diagnostics,
        "stage_a_summary": summary,
        "stage_b_arms": arms,
        "allocation_distance_vs_daily_lw": distances,
        "stage_c_paired_vs_daily_lw": paired,
    }


def verdict(target: dict, control: dict) -> dict:
    """
    Apply the pre-registered outcome rule. Written before the numbers existed.

    The control does the work here: a lead/lag signature on five US-listed ETFs
    sharing one calendar would mean the effect is not a calendar effect, and
    outcome D fires regardless of how interesting the target looks.
    """
    t_non_us = target["stage_a_summary"]["non_us"]
    c_us = control["stage_a_summary"]["us_listed"]

    signature_on_target = bool(
        t_non_us and t_non_us["lag1_over_same_day"] is not None
        and t_non_us["lag1_over_same_day"] > 1.0
    )
    signature_on_control = bool(
        c_us and c_us["lag1_over_same_day"] is not None
        and c_us["lag1_over_same_day"] > 1.0
    )

    arms = target["stage_b_arms"]
    sharpes = [a["sharpe_net"] for a in arms.values()]
    sharpe_spread = round(max(sharpes) - min(sharpes), 4)
    alloc_moved = any(
        d.get("mean_turnover_to_switch", 0.0) > WEIGHT_EPS
        for d in target["allocation_distance_vs_daily_lw"].values()
    )
    # "Material" for Sharpe reuses the 0.05 threshold the regime-cap experiment
    # adopted, so two experiments in this project do not use two different bars.
    sharpe_moved = sharpe_spread > 0.05

    # Whether any DIFFERENCE is established is a separate question from whether
    # the point estimates moved, and only the paired test can answer it. Kept
    # apart deliberately: outcome A turns on sensitivity of the input, which is
    # demonstrated by the allocations regardless of how the test lands.
    paired = target.get("stage_c_paired_vs_daily_lw", {})
    established = sorted(
        name for name, r in paired.items()
        if r.get("p_value_no_outperformance") is not None
        and r["p_value_no_outperformance"] < 0.05
    )

    if signature_on_control:
        outcome, reading = "D", (
            "The lead/lag signature appears on the single-calendar control too, "
            "so it is not a market-calendar effect. The premise is refuted."
        )
    elif not signature_on_target:
        outcome, reading = "REFUTED", (
            "No lead/lag signature on the non-US assets. The asynchrony premise "
            "does not hold on this data."
        )
    elif alloc_moved and sharpe_moved:
        established_clause = (
            f"The paired test establishes a Sharpe difference for: "
            f"{', '.join(established)}."
            if established else
            "No paired test establishes a Sharpe DIFFERENCE, so the point spread "
            "must not be read as one estimator outperforming another."
        )
        outcome, reading = "A", (
            "The signature is present on the target and absent on the control, "
            "and the covariance choice moves the ALLOCATION on every rebalance. "
            "The covariance input is a first-order sensitivity, so conclusions "
            "drawn from the covariance-model ladder are conditioned on a "
            "measurement choice nobody made deliberately. "
            + established_clause
        )
    elif alloc_moved:
        outcome, reading = "B", (
            "The signature is present and the allocation moves, but net Sharpe "
            "does not clear the materiality bar. The bias is real and the "
            "constraint absorbs it — consistent with the finding that the 25% "
            "cap out-regularises every covariance model tried."
        )
    else:
        outcome, reading = "C", (
            "The signature is present in the correlations but does not "
            "propagate to the allocation."
        )

    return {
        "outcome": outcome,
        "reading": reading,
        "signature_on_target": signature_on_target,
        "signature_on_control": signature_on_control,
        "allocation_moved": bool(alloc_moved),
        "sharpe_spread_across_estimators": sharpe_spread,
        "sharpe_material_threshold": 0.05,
        # Reported alongside, never merged into, the outcome letter.
        "paired_difference_established_vs_daily_lw": established,
        "paired_test_note": (
            "Estimators listed above reject the null of no difference against "
            "the production daily estimator at the 5% level under a paired "
            "moving-block bootstrap. An EMPTY list means no difference was "
            "established — which is not evidence the estimators are "
            "equivalent, and does not affect the outcome letter: outcome A "
            "turns on the sensitivity of the covariance input and the "
            "allocation, both of which are measured directly."
        ),
        # Said once, in the artifact, so it travels with the numbers.
        "scope": (
            "Measures the SENSITIVITY of the covariance input, and of the "
            "allocation built on it, to market-calendar and liquidity effects. "
            "This limits what can be inferred from the covariance-model ladder. "
            "It is NOT a bound on achievable performance and no such quantity "
            "is estimated here."
        ),
    }


def main() -> None:
    params = load_params()

    target = run_universe("full_2021", "data/gold/log_returns.parquet", params)
    control = run_universe("etf_2017", "data/gold/log_returns_etf.parquet", params)
    v = verdict(target, control)

    log.info("")
    log.info("=" * 72)
    log.info("PRE-REGISTERED OUTCOME %s", v["outcome"])
    log.info("  %s", v["reading"])
    log.info("=" * 72)

    returns = pd.read_parquet(ROOT / "data/gold/log_returns.parquet")
    out = ROOT / "data/gold/nonsync_covariance.json"
    out.write_text(json.dumps({
        "provenance": build_provenance(
            universe="full_2021",
            returns=returns,
            source_artifacts=[
                params["backtest"]["universes"]["full_2021"],
                params["backtest"]["universes"]["etf_2017"],
                "data/gold/currency_manifest.json",
                "params.yaml",
                "experiments/nonsync_covariance.py",
            ],
        ),
        "reference_asset": REFERENCE,
        "non_us_assets": list(NON_US_ASSETS),
        "target": target,
        "control": control,
        "verdict": v,
    }, indent=2))
    log.info("")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
