"""
run_phase4.py — Phase 4 entry point: HMM regime + dynamic covariance vs.
Phase 2 baselines, MLflow-tracked.

Reruns all 4 Phase 2 baselines ALONGSIDE the 3 new Phase 4 strategies
(MinVarianceEWMA, DCCGarchStrategy, RegimeConditionalStrategy) in the SAME
run, on the SAME live Gold snapshot — this is what makes the DSR
`trial_sharpes` pool honest (run_backtest.py's own N-honesty rule: "any
additional configuration backtested on the same universe must be added to
that universe's trial pool") and self-detects data drift against the stored
`phase2_hurdle.json` in the same pass, instead of comparing a fresh Phase 4
Sharpe against a possibly-stale stored number.

Addresses: P1, P2, P3 — HMM regime detection and the covariance ablation
ladder's final two rungs (EWMA, DCC-GARCH) are compared against their honest
Phase 2 floor, out-of-sample and net of cost, exactly like Phase 2 was
compared against 1/N.
Addresses: P4 — every configuration tried (Phase 2 baselines AND Phase 4
models) is one logged trial pool per universe; DSR stays honest.

N-honesty rule, inherited from run_backtest.py, restated for Phase 4: a
non-backtested diagnostic (e.g. a 2-vs-3-state HMM BIC/log-likelihood
comparison that never calls `run_backtest`) is exempt from the trial pool
by definition — `trial_sharpes` are Sharpe ratios of BACKTESTED return
series. If such a diagnostic is ever added (a notebook cell, say), it must
NOT be folded into this file's trial count.

Runtime note: the DCC-GARCH strategy dominates wall-clock — expect roughly
10-15 minutes for a full run against real Gold data (both universes), the
other 6 strategies are fast (seconds). This script is never run by CI.

Usage:
    python src/run_phase4.py
"""

import json
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

from backtest import BacktestResult, build_cost_vector, run_backtest
from metrics import annualized_sharpe, summarize
from strategies import (
    DCCGarchStrategy,
    EqualWeight,
    MaxSharpe,
    MinVariance,
    MinVarianceEWMA,
    MinVarianceLW,
    RegimeConditionalStrategy,
    Strategy,
)
from utils import configure_mlflow, load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_phase4")

ROOT = Path(__file__).resolve().parents[1]

# Names as set on each Strategy.name — used to split "Phase 4 models" from
# "Phase 2 baselines" when recomputing the stored-hurdle sanity check below.
PHASE4_STRATEGY_NAMES = {"min_variance_ewma", "dcc_garch", "regime_conditional"}

_BASELINE_FACTORY = {
    "equal_weight": lambda max_weight, rf: EqualWeight(),
    "min_variance": lambda max_weight, rf: MinVariance(max_weight=max_weight),
    "min_variance_lw": lambda max_weight, rf: MinVarianceLW(max_weight=max_weight),
    "max_sharpe": lambda max_weight, rf: MaxSharpe(max_weight=max_weight, risk_free_annual=rf),
}


def _validate_regime_strategy_names(regime_params: dict) -> None:
    """
    Fail fast with a clear error if `params.yaml: regime.bull_strategy` /
    `bear_strategy` name something outside `_BASELINE_FACTORY` — a typo here
    would otherwise surface as an opaque KeyError deep inside
    `build_strategies`, on whichever universe happens to run first.
    """
    allowed = ", ".join(sorted(_BASELINE_FACTORY))
    for key in ("bull_strategy", "bear_strategy"):
        name = regime_params.get(key)
        if name not in _BASELINE_FACTORY:
            raise ValueError(
                f"params.yaml: regime.{key}='{name}' is not a recognized baseline strategy. "
                f"Allowed values: {allowed}."
            )


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


def load_features(path: str | Path) -> pd.DataFrame:
    """Load a gold-layer Phase 3 ML feature matrix with a proper DatetimeIndex."""
    full_path = ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {full_path}. Run `python src/ml_features.py` first."
        )
    features = pd.read_parquet(full_path)
    features.index = pd.to_datetime(features.index)
    return features


def build_strategies(params: dict) -> list[Strategy]:
    """
    4 Phase 2 baselines + 3 Phase 4 strategies, freshly instantiated —
    NEVER reused across universes, since RegimeConditionalStrategy carries
    a per-instance `regime_log` that must not mix universes.
    """
    backtest_params = params["backtest"]
    max_weight = backtest_params["max_weight"]
    rf = backtest_params["risk_free_annual"]
    ewma_params = params["covariance_ewma"]
    dcc_params = params["covariance_dcc_garch"]
    regime_params = params["regime"]
    _validate_regime_strategy_names(regime_params)

    return [
        EqualWeight(),
        MinVariance(max_weight=max_weight),
        MinVarianceLW(max_weight=max_weight),
        MaxSharpe(max_weight=max_weight, risk_free_annual=rf),
        MinVarianceEWMA(max_weight=max_weight, halflife_days=ewma_params["halflife_days"]),
        DCCGarchStrategy(
            max_weight=max_weight,
            garch_p=dcc_params["garch_p"],
            garch_q=dcc_params["garch_q"],
            dcc_a_init=dcc_params["dcc_a_init"],
            dcc_b_init=dcc_params["dcc_b_init"],
            rescale_factor=dcc_params["rescale_factor"],
        ),
        RegimeConditionalStrategy(
            bull_strategy=_BASELINE_FACTORY[regime_params["bull_strategy"]](max_weight, rf),
            bear_strategy=_BASELINE_FACTORY[regime_params["bear_strategy"]](max_weight, rf),
            n_states=regime_params["n_states"],
            n_restarts=regime_params["n_restarts"],
            random_state_base=regime_params["random_state_base"],
            covariance_type=regime_params["covariance_type"],
            min_regime_train_days=regime_params["min_regime_train_days"],
            features=regime_params.get("features"),
        ),
    ]


def run_phase4() -> dict[str, list[BacktestResult]]:
    """
    Execute the full Phase 4 comparison: 4 Phase 2 baselines + 3 Phase 4
    strategies, both universes, one shared MLflow run and DSR trial pool.

    Structure mirrors run_backtest.run_phase2(): one parent MLflow run
    holding shared parameters, one nested run per strategy × universe.
    Writes `data/gold/phase4_results.json` — same shape as
    `phase2_hurdle.json` plus `beats_phase2_hurdle` and `is_phase4_strategy`.
    """
    params = load_params()
    backtest_params = params["backtest"]
    max_weight = backtest_params["max_weight"]
    rf = backtest_params["risk_free_annual"]
    results_path = ROOT / params["phase4"]["results_path"]
    hurdle_path = ROOT / "data" / "gold" / "phase2_hurdle.json"

    stored_hurdle: dict = {}
    if hurdle_path.exists():
        stored_hurdle = json.loads(hurdle_path.read_text())
    else:
        log.warning(
            "No stored phase2_hurdle.json at %s — run src/run_backtest.py first for a "
            "beats_phase2_hurdle comparison. Continuing without it.",
            hurdle_path,
        )

    n_strategies = len(build_strategies(params))

    configure_mlflow()
    mlflow.set_experiment("phase4_regime_covariance")
    all_results: dict[str, list[BacktestResult]] = {}
    phase4_output: dict[str, dict] = {}

    with mlflow.start_run(run_name="phase4_regime_covariance"):
        mlflow.log_params({
            "rebalance_freq":     backtest_params["rebalance_freq"],
            "min_train_days":     backtest_params["min_train_days"],
            "max_weight":         max_weight,
            "risk_free_annual":   rf,
            "cost_bps_etf":       backtest_params["costs_bps"]["etf"],
            "cost_bps_bvc":       backtest_params["costs_bps"]["bvc"],
            "n_strategies":       n_strategies,
            "n_states":           params["regime"]["n_states"],
            "ewma_halflife_days": params["covariance_ewma"]["halflife_days"],
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
            # EVERY strategy tried on this universe, baselines and Phase 4
            # models alike — the N-honesty rule from the module docstring.
            # A flat (zero-variance) net-return series would otherwise divide
            # by zero here; 0.0 is the honest Sharpe for "no realized risk or
            # reward", not a crash or a silently-dropped trial.
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
                recomputed_baselines = [
                    r for r in results if r.strategy_name not in PHASE4_STRATEGY_NAMES
                ]
                recomputed_best = max(
                    recomputed_baselines, key=lambda r: annualized_sharpe(r.net_returns, rf)
                )
                recomputed_sharpe = round(annualized_sharpe(recomputed_best.net_returns, rf), 4)
                if abs(recomputed_sharpe - universe_hurdle["sharpe_net"]) > 0.02:
                    log.warning(
                        "%s: recomputed Phase 2 baseline Sharpe (%.4f, %s) differs materially "
                        "from stored phase2_hurdle.json (%.4f, %s) — Gold data likely moved "
                        "since the hurdle was last regenerated. Rerun src/run_backtest.py.",
                        universe_name, recomputed_sharpe, recomputed_best.strategy_name,
                        universe_hurdle["sharpe_net"], universe_hurdle["strategy"],
                    )

            phase4_output[universe_name] = {
                "strategy": best.strategy_name,
                "sharpe_net": best_sharpe,
                "n_trials": len(results),
                "is_phase4_strategy": best.strategy_name in PHASE4_STRATEGY_NAMES,
                "beats_phase2_hurdle": beats_stored_hurdle,
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
                        "is_phase4_strategy": result.strategy_name in PHASE4_STRATEGY_NAMES,
                    })
                    mlflow.log_metrics(
                        {k: v for k, v in metrics_panel.items() if pd.notna(v)}
                    )

                    artifact_dir = ROOT / "mlruns" / ".tmp_artifacts_phase4"
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
        results_path.write_text(json.dumps(phase4_output, indent=2))
        mlflow.log_artifact(str(results_path))
        log.info(
            "Phase 4 results written → %s : %s",
            results_path,
            {
                k: (v["strategy"], v["sharpe_net"], v["beats_phase2_hurdle"])
                for k, v in phase4_output.items()
            },
        )

        log.info("=== Phase 4 comparison complete. `mlflow ui` to inspect. ===")

    return all_results


if __name__ == "__main__":
    try:
        run_phase4()
    except Exception as exc:
        log.error("Phase 4 run failed: %s", exc, exc_info=True)
        sys.exit(1)
