"""
run_fit_reports.py — persist what each model ACTUALLY did, as a Gold artifact.

Addresses: P4 — `BacktestResult.fit_reports` records every fallback, but until
now it died with the object that held it. A table without a saved source
artifact is neither auditable nor reproducible, so the per-rebalance records
are written here as versioned Gold and every downstream table reads from them.

Scope: the four published strategies that HAVE a fallback path. A strategy
with no failure mode (equal_weight, max_sharpe) cannot mislabel itself, so
including it would pad the artifact without adding a fact.

    dcc_garch           -> Ledoit-Wolf shrinkage, on GARCH/DCC non-convergence
    rf_signal           -> naive sample mean, on thin panel / NaN row / fit error
    xgb_signal          -> naive sample mean, same paths
    regime_conditional  -> defensive bear sub-strategy, on HMM non-convergence

REPORTING RULES, and each exists because its absence would mislead:

* Fallback is counted in REBALANCES and in DAYS. A rebalance governs the whole
  holding period until the next one, so a 1-in-40 rebalance fallback can be a
  1-in-40 *month*, not a single day.
* Full-period performance is labelled as the HYBRID it is, never as the
  requested model alone.
* Excluding-fallback performance is reported ONLY with enough active days.
  Below that floor it is `not_estimable` with a reason — never 0, never blank,
  and never a plausible-looking number a reader would take for the real thing.
  A strategy that fell back on every rebalance has no non-fallback performance
  to report, and saying so is the entire point of the exercise.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

import metrics
import telemetry
from backtest import build_cost_vector, run_backtest
from strategies import (
    DCCGarchStrategy,
    MaxSharpe,
    MinVarianceLW,
    RandomForestSignalStrategy,
    RegimeConditionalStrategy,
    XGBoostSignalStrategy,
)

log = logging.getLogger("run_fit_reports")

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
REPORTS_PATH = GOLD / "fit_reports.parquet"
SUMMARY_PATH = GOLD / "fit_report_summary.json"

UNIVERSES = {
    "full_2021": ("log_returns.parquet", "ml_features_full.parquet"),
    "etf_2017": ("log_returns_etf.parquet", "ml_features_etf.parquet"),
}

# Below this many non-fallback OOS days, an excluding-fallback Sharpe is not
# reported. One year is not a statistical threshold — it is the point below
# which this project has repeatedly shown its own intervals to be uninformative
# (see the full_2021 test-window caveat), so quoting a number under it would
# invite exactly the over-reading Phase 2 spent its time removing.
MIN_ACTIVE_DAYS_FOR_SHARPE = 252


def _build_strategies(params: dict) -> dict[str, object]:
    """Construct each strategy EXACTLY as its published runner does.

    Addresses: P4 — a fallback rate measured on a differently-configured
    strategy answers a different question. `min_train_rows` in particular is
    the parameter that drives the ML signal's most common fallback path, so
    reading it from `params.yaml` rather than accepting a constructor default
    is the difference between measuring the published strategy and measuring a
    lookalike. An earlier draft of this function omitted the whole
    `ml_signals` block and produced an rf_signal that scored 0.146 against the
    published 1.123 — a different strategy wearing the same name, which is the
    very confusion this module exists to prevent.

    Mirrors `run_phase4.py` (regime, dcc) and `run_phase4b.py` (signals).
    """
    bt, rg, sig = params["backtest"], params["regime"], params["ml_signals"]
    cap, rf_rate = bt["max_weight"], bt["risk_free_annual"]

    # The signal strategies take their regime settings from their own
    # defaults in run_phase4b — passing the `regime` block here would
    # configure them differently from the published run.
    signal_kwargs = dict(
        max_weight=cap,
        risk_free_annual=rf_rate,
        min_train_rows=sig["min_train_rows"],
        short_window=sig["short_window"],
        long_window=sig["long_window"],
        momentum_windows=sig["momentum_windows"],
        condition_on_regime=sig["condition_on_regime"],
    )
    return {
        "dcc_garch": DCCGarchStrategy(max_weight=cap),
        "regime_conditional": RegimeConditionalStrategy(
            bull_strategy=MaxSharpe(max_weight=cap, risk_free_annual=rf_rate),
            bear_strategy=MinVarianceLW(max_weight=cap),
            n_states=rg["n_states"], n_restarts=rg["n_restarts"],
            random_state_base=rg["random_state_base"],
            covariance_type=rg["covariance_type"],
            min_regime_train_days=rg["min_regime_train_days"],
        ),
        "rf_signal": RandomForestSignalStrategy(
            model_params=sig["random_forest"], **signal_kwargs
        ),
        "xgb_signal": XGBoostSignalStrategy(
            model_params=sig["xgboost"], **signal_kwargs
        ),
    }


def _performance_split(result, risk_free_annual: float) -> dict:
    """Full-period, fallback-period and active-period performance.

    The three are reported side by side because the comparison is the finding:
    a strategy whose fallback periods carry its performance is not the strategy
    its label claims.
    """
    net = result.net_returns
    mask = result.fallback_mask()
    active, degraded = net[~mask], net[mask]

    def block(series: pd.Series, label: str) -> dict:
        if series.empty:
            return {"label": label, "n_days": 0, "status": "not_estimable",
                    "reason": "zero observations in this period",
                    "net_sharpe": None, "annualized_return": None, "max_drawdown": None}
        if label == "excluding_fallback" and len(series) < MIN_ACTIVE_DAYS_FOR_SHARPE:
            # Reported as unavailable WITH the count, not silently omitted:
            # a reader must be able to see how far short it fell.
            return {
                "label": label, "n_days": int(len(series)), "status": "not_estimable",
                "reason": (
                    f"only {len(series)} non-fallback days "
                    f"(< {MIN_ACTIVE_DAYS_FOR_SHARPE} required)"
                ),
                "net_sharpe": None, "annualized_return": None, "max_drawdown": None,
            }
        return {
            "label": label, "n_days": int(len(series)), "status": "estimated", "reason": None,
            "net_sharpe": round(float(metrics.annualized_sharpe(series, risk_free_annual)), 4),
            "annualized_return": round(float(metrics.annualized_return(series)), 4),
            "max_drawdown": round(float(metrics.max_drawdown(series)), 4),
        }

    return {
        "full_period_hybrid": block(net, "full_period_hybrid"),
        "fallback_periods": block(degraded, "fallback_periods"),
        "excluding_fallback": block(active, "excluding_fallback"),
    }


COST_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0)


def _cost_sensitivity(result, risk_free_annual: float) -> dict:
    """Net performance under 0.5x / 1x / 1.5x / 2x transaction costs.

    Addresses: P4 — a net Sharpe is a statement about an assumed cost model as
    much as about a strategy, and this project's cost assumptions (10 bps ETF,
    30 bps BVC) are judgement calls. A result that survives 2x costs is a
    different claim from one that does not.

    Derived EXACTLY, not re-simulated. The engine deducts the cost drag from
    gross on the day after each rebalance, so `gross - net` is precisely the
    1x drag per day and it scales linearly in the cost multiplier:

        net(m) = gross - m * (gross - net)

    Re-running the backtest per multiplier would produce identical weights —
    costs never enter the optimizer for these strategies — at four times the
    compute, and any tiny difference would be a bug rather than a finding.
    """
    gross, net = result.gross_returns, result.net_returns
    drag = gross - net

    scenarios = {}
    for multiplier in COST_MULTIPLIERS:
        scaled = gross - multiplier * drag
        scenarios[f"{multiplier:g}x"] = {
            "cost_multiplier": multiplier,
            "net_sharpe": round(float(metrics.annualized_sharpe(scaled, risk_free_annual)), 4),
            "annualized_return": round(float(metrics.annualized_return(scaled)), 4),
            "total_cost_drag_annualized": round(
                float(metrics.annualized_return(gross) - metrics.annualized_return(scaled)), 4
            ),
            # Reported per scenario BECAUSE it must not be inferred. See the note.
            "fallback_rate_rebalances": round(float(result.fallback_rate), 4),
        }

    # Where does the strategy stop paying for itself? Scanned rather than
    # solved: annualized return is not linear in the multiplier even though the
    # drag is, so a closed form would be wrong in the third decimal.
    break_even = None
    for step in range(1, 401):
        multiplier = step * 0.05
        if metrics.annualized_return(gross - multiplier * drag) <= 0:
            break_even = round(multiplier, 2)
            break

    return {
        "scenarios": scenarios,
        "break_even_cost_multiplier": break_even,
        "break_even_note": (
            "Multiplier at which the annualized NET return reaches zero. None means "
            "it survives 20x the assumed costs."
        ),
        "fallback_invariance_note": (
            "The fallback rate is identical across every cost scenario, and this is "
            "true BY CONSTRUCTION rather than by measurement: transaction costs enter "
            "only the net-return accounting, never the optimizer, so scaling them "
            "cannot change a weight, a turnover, or whether an estimator converged. "
            "It is reported per scenario so a reader can see that it did not move, "
            "instead of having to deduce it. A turnover-PENALIZED strategy (Phase 4C, "
            "lambda in objective units) would NOT have this invariance, because there "
            "the cost of trading does reach the objective."
        ),
    }


def _execution_profile(result, max_weight: float, rebalance_freq: str) -> dict:
    """Turnover, concentration, cap binding and holding period.

    Addresses: P1/P3 realism — the levers a manager would actually argue about.
    A strategy whose edge needs monthly full-portfolio replacement is a
    different proposition from one that holds for a quarter, even at identical
    Sharpe.
    """
    weights = result.target_weights
    turnover = result.turnover
    at_cap = (weights >= max_weight - 1e-6)
    oos_days = len(result.net_returns)
    n_rebalances = max(len(result.rebalance_dates), 1)

    return {
        "rebalance_frequency": rebalance_freq,
        "avg_holding_days": round(oos_days / n_rebalances, 1),
        "avg_turnover": round(float(turnover.mean()), 4),
        "median_turnover": round(float(turnover.median()), 4),
        "max_turnover": round(float(turnover.max()), 4),
        "avg_cost_drag_bps": round(float(result.costs.mean()) * 10_000, 2),
        "max_allocation_observed": round(float(weights.to_numpy().max()), 4),
        "max_allocation_permitted": float(max_weight),
        "cap_binding_position_rate": round(float(at_cap.to_numpy().mean()), 4),
        "cap_binding_date_rate": round(float(at_cap.any(axis=1).mean()), 4),
        "liquidity_caveat": (
            "Costs are modelled as a fixed per-asset rate on realised turnover, with "
            "no market-impact or capacity term. The BVC names carry standing "
            "illiquidity flags and CIH.CS/BCP.CS show multi-day zero-return runs, so "
            "at a size where market impact matters these figures are optimistic. No "
            "capacity limit has been estimated."
        ),
    }


def _summarize(result, universe: str, strategy: str, risk_free_annual: float,
               max_weight: float, rebalance_freq: str) -> dict:
    reports = result.fit_reports
    mask = result.fallback_mask()
    fallback_rows = reports[reports["fit_status"] == telemetry.STATUS_FALLBACK]
    reasons = (
        fallback_rows["fallback_reason"].dropna().value_counts().to_dict()
        if not fallback_rows.empty else {}
    )
    effective = reports["model_effective"].value_counts().to_dict()

    return {
        "universe": universe,
        "strategy_requested": strategy,
        "models_effective": {str(k): int(v) for k, v in effective.items()},
        "rebalances": int(len(reports)),
        "fallback_rebalances": int(len(fallback_rows)),
        "fallback_rate_rebalances": round(float(result.fallback_rate), 4),
        "oos_days": int(len(mask)),
        "fallback_days": int(mask.sum()),
        "active_days": int((~mask).sum()),
        "fallback_rate_days": round(float(mask.mean()), 4) if len(mask) else 0.0,
        "fallback_reasons": {str(k): int(v) for k, v in reasons.items()},
        "min_training_rows": int(reports["n_training_rows"].min()) if len(reports) else 0,
        "performance": _performance_split(result, risk_free_annual),
        "cost_sensitivity": _cost_sensitivity(result, risk_free_annual),
        "execution_profile": _execution_profile(result, max_weight, rebalance_freq),
    }


def run(universes: list[str] | None = None) -> tuple[Path, Path]:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    bt = params["backtest"]
    selected = universes or list(UNIVERSES)

    long_rows: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for universe in selected:
        returns_file, features_file = UNIVERSES[universe]
        returns = pd.read_parquet(GOLD / returns_file)
        features = pd.read_parquet(GOLD / features_file)
        costs = build_cost_vector(
            returns.columns, bt["costs_bps"]["etf"], bt["costs_bps"]["bvc"]
        )

        for name, strategy in _build_strategies(params).items():
            started = time.time()
            result = run_backtest(
                returns, strategy, cost_bps=costs, extras={"features": features},
                min_train_days=bt["min_train_days"],
                rebalance_freq=bt["rebalance_freq"],
                max_weight=bt["max_weight"], universe_name=universe,
            )
            frame = result.fit_reports.copy()
            frame.insert(0, "strategy", name)
            frame.insert(0, "universe", universe)
            frame.index.name = "Date"
            long_rows.append(frame.reset_index())

            summary = _summarize(
                result, universe, name, bt["risk_free_annual"],
                bt["max_weight"], bt["rebalance_freq"],
            )
            summary["runtime_seconds"] = round(time.time() - started, 1)
            summaries.append(summary)
            log.info(
                "%s/%s: %d/%d rebalances fell back (%.1f%%), %d/%d OOS days (%.1f%%) [%.0fs]",
                universe, name, summary["fallback_rebalances"], summary["rebalances"],
                100 * summary["fallback_rate_rebalances"], summary["fallback_days"],
                summary["oos_days"], 100 * summary["fallback_rate_days"],
                summary["runtime_seconds"],
            )

    reports = pd.concat(long_rows, ignore_index=True)
    REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    reports.to_parquet(REPORTS_PATH, index=False)

    payload = {
        "generated_from_committed_gold": True,
        "scope_note": (
            "Only strategies with a fallback path are covered; a strategy that "
            "cannot degrade cannot mislabel itself."
        ),
        "reporting_rules": {
            "full_period_label": (
                "Full-period performance is the HYBRID of requested model and "
                "fallback, never the requested model alone."
            ),
            "excluding_fallback": (
                f"Reported only with >= {MIN_ACTIVE_DAYS_FOR_SHARPE} non-fallback OOS "
                "days; otherwise 'not_estimable' with the count. Never 0 and never blank."
            ),
            "days_vs_rebalances": (
                "A rebalance governs every day until the next one, so the day-based "
                "rate is the economically meaningful one."
            ),
        },
        "min_active_days_for_sharpe": MIN_ACTIVE_DAYS_FOR_SHARPE,
        "results": summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                            encoding="utf-8")
    return REPORTS_PATH, SUMMARY_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for path in run():
        print(path.relative_to(ROOT))
