"""
run_backtest.py — Phase 2 entry point: baselines × universes, MLflow-tracked.

Runs every baseline strategy on both universes (dual-universe design: the
ETF-only matrix restores the COVID-2020 crisis window that the BVC data gap
removes from the 9-asset matrix) and logs parameters, metrics, and artifacts
to MLflow so "how many strategies did we try?" is auditable — the input the
Deflated Sharpe Ratio needs to stay honest.

Addresses: P4 — every configuration tried is a logged trial, and all
headline metrics are out-of-sample and net of transaction costs.
Addresses: P1 — these baselines set the net-of-cost hurdle that Phase 4's
ML models (regime-aware, dynamic covariance) must beat to justify existing.

Usage:
    python src/run_backtest.py
"""

import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

import json

from backtest import BacktestResult, build_cost_vector, run_backtest
from metrics import annualized_sharpe, summarize
from strategies import EqualWeight, MaxSharpe, MinVariance, MinVarianceLW, Strategy
from utils import load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_backtest")

ROOT = Path(__file__).resolve().parents[1]


def load_universe(path: str | Path) -> pd.DataFrame:
    """Load a gold-layer log-returns matrix with a proper DatetimeIndex."""
    full_path = ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {full_path}. Run `python src/clean.py && "
            f"python src/features.py` to produce both gold matrices."
        )
    returns = pd.read_parquet(full_path)
    returns.index = pd.to_datetime(returns.index)
    return returns


def run_phase2() -> dict[str, list[BacktestResult]]:
    """
    Execute the full Phase 2 baseline comparison.

    Structure: one parent MLflow run holding shared parameters, one nested
    run per strategy × universe holding that combination's metrics and
    artifacts. `n_trials` (the DSR's N) counts the strategies compared per
    universe in THIS run — the honest-N limitation is documented in
    metrics.deflated_sharpe_ratio.

    N-honesty rule: ANY additional configuration backtested on the same
    universe during a comparison (including diagnostic runs in notebooks,
    e.g. an uncapped variant) must be added to that universe's trial pool.
    Selection bias doesn't care whether a run was labeled "diagnostic".
    """
    params = load_params()["backtest"]
    max_weight = params["max_weight"]
    rf = params["risk_free_annual"]

    strategies: list[Strategy] = [
        EqualWeight(),
        MinVariance(max_weight=max_weight),
        MinVarianceLW(max_weight=max_weight),   # covariance ablation ladder, rung 1
        MaxSharpe(max_weight=max_weight, risk_free_annual=rf),
    ]

    mlflow.set_experiment("phase2_backtest")
    all_results: dict[str, list[BacktestResult]] = {}

    with mlflow.start_run(run_name="phase2_baselines"):
        mlflow.log_params({
            "rebalance_freq":   params["rebalance_freq"],
            "min_train_days":   params["min_train_days"],
            "max_weight":       max_weight,
            "risk_free_annual": rf,
            "cost_bps_etf":     params["costs_bps"]["etf"],
            "cost_bps_bvc":     params["costs_bps"]["bvc"],
            "n_strategies":     len(strategies),
        })

        for universe_name, path in params["universes"].items():
            returns = load_universe(path)
            cost_vector = build_cost_vector(
                returns.columns,
                etf_cost_bps=params["costs_bps"]["etf"],
                bvc_cost_bps=params["costs_bps"]["bvc"],
            )
            log.info(
                "=== Universe %s: %d assets, %s → %s (%d days) ===",
                universe_name, returns.shape[1],
                returns.index.min().date(), returns.index.max().date(), len(returns),
            )

            results = [
                run_backtest(
                    returns,
                    strategy,
                    rebalance_freq=params["rebalance_freq"],
                    min_train_days=params["min_train_days"],
                    cost_bps=cost_vector,
                    universe_name=universe_name,
                    max_weight=max_weight,   # engine-enforced, not just promised
                )
                for strategy in strategies
            ]
            all_results[universe_name] = results

            # DSR inputs: per-period (daily, non-annualized) net Sharpe of every
            # strategy tried on this universe — including the one being scored.
            trial_sharpes = [
                float(r.net_returns.mean() / r.net_returns.std()) for r in results
            ]
            # Benchmark for IR: the equal-weight net series (DeMiguel hurdle)
            ew_net = next(r for r in results if r.strategy_name == "equal_weight").net_returns

            for result in results:
                with mlflow.start_run(
                    run_name=f"{result.strategy_name}__{universe_name}", nested=True
                ):
                    metrics_panel = summarize(
                        net_returns=result.net_returns,
                        gross_returns=result.gross_returns,
                        turnover=result.turnover,
                        benchmark_net=ew_net if result.strategy_name != "equal_weight" else None,
                        trial_sharpes=trial_sharpes,
                        risk_free_annual=rf,
                    )
                    mlflow.log_params({
                        "strategy": result.strategy_name,
                        "universe": universe_name,
                        "n_rebalances": len(result.rebalance_dates),
                    })
                    mlflow.log_metrics(
                        {k: v for k, v in metrics_panel.items() if pd.notna(v)}
                    )

                    artifact_dir = ROOT / "mlruns" / ".tmp_artifacts"
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    weights_csv = artifact_dir / "target_weights.csv"
                    returns_csv = artifact_dir / "net_returns.csv"
                    result.target_weights.to_csv(weights_csv)
                    result.net_returns.to_csv(returns_csv)
                    mlflow.log_artifact(str(weights_csv))
                    mlflow.log_artifact(str(returns_csv))

                    log.info(
                        "%s / %s: sharpe_net=%.3f mdd=%.1f%% turnover=%.3f dsr=%.3f",
                        result.strategy_name, universe_name,
                        metrics_panel["sharpe_net"],
                        metrics_panel["max_drawdown_net"] * 100,
                        metrics_panel["avg_turnover"],
                        metrics_panel.get("dsr_net", float("nan")),
                    )

        # Machine-readable hurdle: the number Phase 4 must beat, as data.
        # Phase 4's comparison can assert against this file programmatically
        # instead of trusting a hand-copied figure from a notebook.
        hurdle = {}
        for universe_name, results in all_results.items():
            best = max(results, key=lambda r: annualized_sharpe(r.net_returns, rf))
            hurdle[universe_name] = {
                "strategy": best.strategy_name,
                "sharpe_net": round(annualized_sharpe(best.net_returns, rf), 4),
                "n_trials": len(results),
                "rebalance_freq": params["rebalance_freq"],
                "max_weight": max_weight,
                "cost_bps": params["costs_bps"],
                "oos_start": str(best.net_returns.index.min().date()),
                "oos_end": str(best.net_returns.index.max().date()),
            }
        hurdle_path = ROOT / "data" / "gold" / "phase2_hurdle.json"
        hurdle_path.write_text(json.dumps(hurdle, indent=2))
        mlflow.log_artifact(str(hurdle_path))
        log.info("Phase 4 hurdle written → %s : %s", hurdle_path,
                 {k: v["sharpe_net"] for k, v in hurdle.items()})

        log.info("=== Phase 2 baseline comparison complete. `mlflow ui` to inspect. ===")

    return all_results


if __name__ == "__main__":
    try:
        run_phase2()
    except Exception as exc:
        log.error("Phase 2 run failed: %s", exc, exc_info=True)
        sys.exit(1)
