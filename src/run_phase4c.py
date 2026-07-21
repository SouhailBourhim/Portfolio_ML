"""
run_phase4c.py — Phase 4C entry point: cost-aware optimization and `mu`
regularization vs. the Phase 4 hurdle, MLflow-tracked.

WHY THIS PHASE EXISTS. Phase 4B returned an honest negative result — neither
F7 signal strategy beat its universe's hurdle — but the failure had a
specific, diagnosable shape rather than "the model doesn't predict anything":

    full_2021   regime_conditional  gross 1.204 -> net 1.122  (turnover 0.113)
    full_2021   rf_signal           gross 1.240 -> net 1.062  (turnover 0.885)

`rf_signal` produced the HIGHEST GROSS SHARPE of any strategy in the entire
comparison and then handed 0.178 of it back in transaction costs. The signal
was informative; acting on every revision of it was not affordable. That is a
portfolio-construction failure, not a prediction failure, and it has textbook
remedies this phase implements as a clean ablation:

  1. `*_cost`   — turnover-penalized objective. Price the cost of REACHING a
                  portfolio, not just the merit of holding it.
  2. `*_shrunk` — shrink the predicted `mu` toward the naive sample mean.
                  Chopra & Ziemba (1993): estimation error in expected
                  returns damages a mean-variance optimizer roughly an order
                  of magnitude more than equivalent covariance error — which
                  also explains why Phase 4 (better covariance) beat its
                  hurdle while Phase 4B (better mu) did not.
  3. `*_rank`   — keep only the model's cross-sectional ORDERING, borrowing
                  level and dispersion from the naive estimate. The strongest
                  form of "trust the ranking, not the magnitudes".
  4. `*_cost_dcc` — best predicted `mu` + best covariance rung (DCC-GARCH),
                  the pairing explicitly deferred in Phase 4B.

Each variant changes exactly ONE thing relative to `rf_signal` (except the
deliberate #4 combination), so a win is attributable rather than mysterious.

Addresses: P1 — both levers target estimation-error amplification: the
turnover penalty stops a noisy `mu` from being expressed as violent weight
swings; the `mu` transforms damp the noise before it reaches the optimizer.
Addresses: P4 — every configuration tried lands in ONE shared DSR trial pool
per universe. This phase ADDS trials to an already-searched space, which
deflates the Sharpe of everything in it; that is the honest accounting and
the reason `n_trials` is reported alongside every result.

HONESTY NOTE ON HYPERPARAMETERS: `turnover_penalty` and `shrinkage_weight`
are chosen, not fitted. Selecting them by looking at this comparison's own
out-of-sample result would be exactly the backtest overfitting (P4) this
project exists to avoid. Phase 5's purged K-Fold CV is where they get
selected honestly; until then, results here are "what this configuration
does", not "what the best configuration does".

Usage:
    python src/run_phase4c.py
"""

import json
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

from backtest import BacktestResult, build_cost_vector, run_backtest
from metrics import annualized_sharpe, summarize
from run_phase4 import load_features, load_universe
from run_phase4b import build_strategies as build_phase4b_strategies
from strategies import RandomForestSignalStrategy, Strategy, XGBoostSignalStrategy
from utils import load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("run_phase4c")

ROOT = Path(__file__).resolve().parents[1]

PHASE4C_STRATEGY_NAMES = {
    "rf_signal_cost",
    "rf_signal_shrunk",
    "rf_signal_rank",
    "rf_signal_cost_dcc",
    "xgb_signal_cost",
}


def build_strategies(params: dict) -> list[Strategy]:
    """
    The 9 strategies Phase 4B compared, plus the 5 Phase 4C variants.

    Freshly instantiated every call, same reasoning as
    `run_phase4.build_strategies` (a strategy holding per-run diagnostic
    state must not mix universes).
    """
    backtest_params = params["backtest"]
    signal_params = params["ml_signals"]
    phase4c_params = params["phase4c"]
    dcc_params = params["covariance_dcc_garch"]
    regime_params = params["regime"]

    shared = dict(
        max_weight=backtest_params["max_weight"],
        risk_free_annual=backtest_params["risk_free_annual"],
        min_train_rows=signal_params["min_train_rows"],
        short_window=signal_params["short_window"],
        long_window=signal_params["long_window"],
        momentum_windows=signal_params["momentum_windows"],
        condition_on_regime=signal_params["condition_on_regime"],
        n_states=regime_params["n_states"],
        n_restarts=regime_params["n_restarts"],
        random_state_base=regime_params["random_state_base"],
        covariance_type=regime_params["covariance_type"],
        min_regime_train_days=regime_params["min_regime_train_days"],
    )
    penalty = phase4c_params["turnover_penalty"]

    return build_phase4b_strategies(params) + [
        # 1. Turnover penalty alone — isolates the cost-of-trading fix.
        RandomForestSignalStrategy(
            name="rf_signal_cost", model_params=signal_params["random_forest"],
            turnover_penalty=penalty, **shared,
        ),
        # 2. mu shrinkage alone — isolates the Chopra-Ziemba fix.
        RandomForestSignalStrategy(
            name="rf_signal_shrunk", model_params=signal_params["random_forest"],
            mu_transform="shrink",
            shrinkage_weight=phase4c_params["shrinkage_weight"], **shared,
        ),
        # 3. Rank tilt alone — the strongest form of distrusting magnitudes.
        RandomForestSignalStrategy(
            name="rf_signal_rank", model_params=signal_params["random_forest"],
            mu_transform="rank", **shared,
        ),
        # 4. Best mu + best covariance, deliberately combined (not an
        #    isolation test — the "does stacking help" question).
        RandomForestSignalStrategy(
            name="rf_signal_cost_dcc", model_params=signal_params["random_forest"],
            turnover_penalty=penalty, cov_estimator="dcc_garch",
            garch_p=dcc_params["garch_p"], garch_q=dcc_params["garch_q"],
            **shared,
        ),
        # 5. The same penalty on Phase 4B's WORST offender (xgb_signal ran
        #    1.076 turnover on full_2021) — does the fix rescue it too, or
        #    was that model's signal genuinely weaker?
        XGBoostSignalStrategy(
            name="xgb_signal_cost", model_params=signal_params["xgboost"],
            turnover_penalty=penalty, **shared,
        ),
    ]


def run_phase4c() -> dict[str, list[BacktestResult]]:
    """
    Execute the full Phase 4C comparison: 14 strategies, both universes,
    one shared MLflow run and DSR trial pool per universe.

    Structure mirrors `run_phase4b.run_phase4c`'s predecessor. Writes
    `data/gold/phase4c_results.json` — same shape as `phase4b_results.json`
    plus `beats_phase4_hurdle` and `is_phase4c_strategy`.
    """
    params = load_params()
    backtest_params = params["backtest"]
    max_weight = backtest_params["max_weight"]
    rf = backtest_params["risk_free_annual"]
    results_path = ROOT / params["phase4c"]["results_path"]
    hurdle_path = ROOT / "data" / "gold" / "phase4_results.json"

    stored_hurdle: dict = {}
    if hurdle_path.exists():
        stored_hurdle = json.loads(hurdle_path.read_text())
    else:
        log.warning(
            "No stored phase4_results.json at %s — run src/run_phase4.py first "
            "for a beats_phase4_hurdle comparison. Continuing without it.",
            hurdle_path,
        )

    n_strategies = len(build_strategies(params))

    mlflow.set_experiment("phase4c_cost_aware")
    all_results: dict[str, list[BacktestResult]] = {}
    phase4c_output: dict[str, dict] = {}

    with mlflow.start_run(run_name="phase4c_cost_aware"):
        mlflow.log_params({
            "rebalance_freq":    backtest_params["rebalance_freq"],
            "min_train_days":    backtest_params["min_train_days"],
            "max_weight":        max_weight,
            "risk_free_annual":  rf,
            "cost_bps_etf":      backtest_params["costs_bps"]["etf"],
            "cost_bps_bvc":      backtest_params["costs_bps"]["bvc"],
            "n_strategies":      n_strategies,
            "turnover_penalty":  params["phase4c"]["turnover_penalty"],
            "shrinkage_weight":  params["phase4c"]["shrinkage_weight"],
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
                    returns, strategy,
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

            trial_sharpes = [
                float(r.net_returns.mean() / r.net_returns.std())
                if r.net_returns.std() > 0 else 0.0
                for r in results
            ]
            ew_net = next(r for r in results if r.strategy_name == "equal_weight").net_returns

            best = max(results, key=lambda r: annualized_sharpe(r.net_returns, rf))
            best_sharpe = round(annualized_sharpe(best.net_returns, rf), 4)

            universe_hurdle = stored_hurdle.get(universe_name)
            beats_stored_hurdle = (
                best_sharpe > universe_hurdle["sharpe_net"] if universe_hurdle else None
            )

            phase4c_output[universe_name] = {
                "strategy": best.strategy_name,
                "sharpe_net": best_sharpe,
                "n_trials": len(results),
                "is_phase4c_strategy": best.strategy_name in PHASE4C_STRATEGY_NAMES,
                "beats_phase4_hurdle": beats_stored_hurdle,
                "phase4_hurdle": (
                    {"strategy": universe_hurdle["strategy"],
                     "sharpe_net": universe_hurdle["sharpe_net"]}
                    if universe_hurdle else None
                ),
                # The whole point of the phase: turnover and the gross→net
                # gap, per strategy, so the diagnosis is in the artifact and
                # not only in a notebook someone has to rerun.
                "per_strategy": {
                    r.strategy_name: {
                        "sharpe_gross": round(annualized_sharpe(r.gross_returns, rf), 4),
                        "sharpe_net": round(annualized_sharpe(r.net_returns, rf), 4),
                        "avg_turnover": round(float(r.turnover.mean()), 4),
                    }
                    for r in results
                },
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
                        "is_phase4c_strategy": result.strategy_name in PHASE4C_STRATEGY_NAMES,
                    })
                    mlflow.log_metrics(
                        {k: v for k, v in metrics_panel.items() if pd.notna(v)}
                    )
                    log.info(
                        "%s / %s: gross=%.3f net=%.3f turnover=%.3f dsr=%.3f",
                        result.strategy_name, universe_name,
                        annualized_sharpe(result.gross_returns, rf),
                        metrics_panel["sharpe_net"],
                        metrics_panel["avg_turnover"],
                        metrics_panel.get("dsr_net", float("nan")),
                    )

        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(phase4c_output, indent=2))
        mlflow.log_artifact(str(results_path))
        log.info(
            "Phase 4C results written → %s : %s",
            results_path,
            {
                k: (v["strategy"], v["sharpe_net"], v["beats_phase4_hurdle"])
                for k, v in phase4c_output.items()
            },
        )
        log.info("=== Phase 4C comparison complete. `mlflow ui` to inspect. ===")

    return all_results


if __name__ == "__main__":
    try:
        run_phase4c()
    except Exception as exc:
        log.error("Phase 4C run failed: %s", exc, exc_info=True)
        sys.exit(1)
