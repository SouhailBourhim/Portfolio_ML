"""
run_phase4b.py — Phase 4B / F7 entry point: adaptive ML signal models vs.
Phase 4 results, MLflow-tracked.

Reruns all 7 existing strategies (4 Phase 2 baselines + Phase 4's
MinVarianceEWMA/DCCGarchStrategy/RegimeConditionalStrategy) ALONGSIDE the
2 new F7 strategies (RandomForestSignalStrategy, XGBoostSignalStrategy) in
the SAME run, on the SAME live Gold snapshot — the same N-honesty rule
`run_phase4.py` already follows for `phase2_hurdle.json`, now one level up:
the DSR `trial_sharpes` pool stays honest, and this run self-detects data
drift against the stored `phase4_results.json` in the same pass instead of
comparing a fresh Phase 4B Sharpe against a possibly-stale stored number.

Addresses: P1, P2, P3 — RandomForest/XGBoost return-prediction signals,
regime-conditioned via one pooled input feature, feeding the same Sharpe-
maximization objective every prior addition has reused, compared against
their honest Phase 4 floor.
Addresses: P4 — every configuration tried (existing 7 + the 2 new F7
strategies) is one logged trial pool per universe; DSR stays honest.

LSTM (a third F7 model family) was built and tested in isolation
(`src/lstm_signal.py`... since removed) but dropped from this comparison:
torch and xgboost loaded together in the same process segfaulted on the
development machine (a native/OpenMP library conflict, not a code defect
in either library). Deferred to a run on more capable hardware — see
`CLAUDE.md`'s Phase 4B notes. `build_strategies` below returns 9 strategies,
not 10.

Usage:
    python src/run_phase4b.py
"""

import json
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

from backtest import BacktestResult, build_cost_vector, run_backtest
from metrics import annualized_sharpe, summarize
from run_phase4 import build_strategies as build_phase4_strategies
from run_phase4 import load_features, load_universe
from strategies import RandomForestSignalStrategy, Strategy, XGBoostSignalStrategy
from utils import load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_phase4b")

ROOT = Path(__file__).resolve().parents[1]

# Names as set on each Strategy.name — used to split "F7 models" from
# everything that came before, when logging and when computing the
# beats-the-hurdle comparison below.
PHASE4B_STRATEGY_NAMES = {"rf_signal", "xgb_signal"}


def build_strategies(params: dict) -> list[Strategy]:
    """
    The existing 7 strategies (Phase 2 + Phase 4) + the 2 new F7 strategies
    — freshly instantiated every call, same reasoning as
    `run_phase4.build_strategies` (RegimeConditionalStrategy's `regime_log`
    must not mix universes).
    """
    backtest_params = params["backtest"]
    max_weight = backtest_params["max_weight"]
    rf = backtest_params["risk_free_annual"]
    signal_params = params["ml_signals"]

    return build_phase4_strategies(params) + [
        RandomForestSignalStrategy(
            max_weight=max_weight,
            risk_free_annual=rf,
            model_params=signal_params["random_forest"],
            min_train_rows=signal_params["min_train_rows"],
            short_window=signal_params["short_window"],
            long_window=signal_params["long_window"],
            momentum_windows=signal_params["momentum_windows"],
            condition_on_regime=signal_params["condition_on_regime"],
        ),
        XGBoostSignalStrategy(
            max_weight=max_weight,
            risk_free_annual=rf,
            model_params=signal_params["xgboost"],
            min_train_rows=signal_params["min_train_rows"],
            short_window=signal_params["short_window"],
            long_window=signal_params["long_window"],
            momentum_windows=signal_params["momentum_windows"],
            condition_on_regime=signal_params["condition_on_regime"],
        ),
    ]


def run_phase4b() -> dict[str, list[BacktestResult]]:
    """
    Execute the full Phase 4B comparison: 9 strategies, both universes, one
    shared MLflow run and DSR trial pool.

    Structure mirrors `run_phase4.run_phase4()`. Writes
    `data/gold/phase4b_results.json` — same shape as `phase4_results.json`
    plus `beats_phase4_hurdle` and `is_phase4b_strategy`.
    """
    params = load_params()
    backtest_params = params["backtest"]
    max_weight = backtest_params["max_weight"]
    rf = backtest_params["risk_free_annual"]
    results_path = ROOT / params["phase4b"]["results_path"]
    hurdle_path = ROOT / "data" / "gold" / "phase4_results.json"

    stored_hurdle: dict = {}
    if hurdle_path.exists():
        stored_hurdle = json.loads(hurdle_path.read_text())
    else:
        log.warning(
            "No stored phase4_results.json at %s — run src/run_phase4.py first for a "
            "beats_phase4_hurdle comparison. Continuing without it.",
            hurdle_path,
        )

    n_strategies = len(build_strategies(params))

    mlflow.set_experiment("phase4b_adaptive_ml_signals")
    all_results: dict[str, list[BacktestResult]] = {}
    phase4b_output: dict[str, dict] = {}

    with mlflow.start_run(run_name="phase4b_adaptive_ml_signals"):
        mlflow.log_params({
            "rebalance_freq":   backtest_params["rebalance_freq"],
            "min_train_days":   backtest_params["min_train_days"],
            "max_weight":       max_weight,
            "risk_free_annual": rf,
            "cost_bps_etf":     backtest_params["costs_bps"]["etf"],
            "cost_bps_bvc":     backtest_params["costs_bps"]["bvc"],
            "n_strategies":     n_strategies,
        })

        for universe_name, path in backtest_params["universes"].items():
            returns = load_universe(path)
            features = load_features(params["ml_features"]["outputs"][universe_name])
            cost_vector = build_cost_vector(
                returns.columns,
                etf_cost_bps=backtest_params["costs_bps"]["etf"],
                bvc_cost_bps=backtest_params["costs_bps"]["bvc"],
            )
            log.info(
                "=== Universe %s: %d assets, %s → %s (%d days) ===",
                universe_name, returns.shape[1],
                returns.index.min().date(), returns.index.max().date(), len(returns),
            )

            strategies = build_strategies(params)
            results = [
                run_backtest(
                    returns,
                    strategy,
                    rebalance_freq=backtest_params["rebalance_freq"],
                    min_train_days=backtest_params["min_train_days"],
                    cost_bps=cost_vector,
                    extras={"features": features},
                    universe_name=universe_name,
                    max_weight=max_weight,
                )
                for strategy in strategies
            ]
            all_results[universe_name] = results

            # DSR inputs: per-period (daily, non-annualized) net Sharpe of
            # EVERY strategy tried on this universe — the N-honesty rule.
            trial_sharpes = [
                float(r.net_returns.mean() / r.net_returns.std())
                if r.net_returns.std() > 0
                else 0.0
                for r in results
            ]
            ew_net = next(r for r in results if r.strategy_name == "equal_weight").net_returns

            best = max(results, key=lambda r: annualized_sharpe(r.net_returns, rf))
            best_sharpe = round(annualized_sharpe(best.net_returns, rf), 4)

            universe_hurdle = stored_hurdle.get(universe_name)
            beats_stored_hurdle = (
                best_sharpe > universe_hurdle["sharpe_net"] if universe_hurdle else None
            )
            if universe_hurdle is not None:
                recomputed_prior = [
                    r for r in results if r.strategy_name not in PHASE4B_STRATEGY_NAMES
                ]
                recomputed_best = max(
                    recomputed_prior, key=lambda r: annualized_sharpe(r.net_returns, rf)
                )
                recomputed_sharpe = round(annualized_sharpe(recomputed_best.net_returns, rf), 4)
                if abs(recomputed_sharpe - universe_hurdle["sharpe_net"]) > 0.02:
                    log.warning(
                        "%s: recomputed prior-best Sharpe (%.4f, %s) differs materially "
                        "from stored phase4_results.json (%.4f, %s) — Gold data likely moved "
                        "since the hurdle was last regenerated. Rerun src/run_phase4.py.",
                        universe_name, recomputed_sharpe, recomputed_best.strategy_name,
                        universe_hurdle["sharpe_net"], universe_hurdle["strategy"],
                    )

            phase4b_output[universe_name] = {
                "strategy": best.strategy_name,
                "sharpe_net": best_sharpe,
                "n_trials": len(results),
                "is_phase4b_strategy": best.strategy_name in PHASE4B_STRATEGY_NAMES,
                "beats_phase4_hurdle": beats_stored_hurdle,
                "rebalance_freq": backtest_params["rebalance_freq"],
                "max_weight": max_weight,
                "cost_bps": backtest_params["costs_bps"],
                "oos_start": str(best.net_returns.index.min().date()),
                "oos_end": str(best.net_returns.index.max().date()),
            }

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
                        "is_phase4b_strategy": result.strategy_name in PHASE4B_STRATEGY_NAMES,
                    })
                    mlflow.log_metrics(
                        {k: v for k, v in metrics_panel.items() if pd.notna(v)}
                    )

                    artifact_dir = ROOT / "mlruns" / ".tmp_artifacts_phase4b"
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

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(phase4b_output, indent=2))
        mlflow.log_artifact(str(results_path))
        log.info(
            "Phase 4B results written → %s : %s",
            results_path,
            {
                k: (v["strategy"], v["sharpe_net"], v["beats_phase4_hurdle"])
                for k, v in phase4b_output.items()
            },
        )

        log.info("=== Phase 4B comparison complete. `mlflow ui` to inspect. ===")

    return all_results


if __name__ == "__main__":
    try:
        run_phase4b()
    except Exception as exc:
        log.error("Phase 4B run failed: %s", exc, exc_info=True)
        sys.exit(1)
