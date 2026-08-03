"""
run_explainability.py — produce `data/gold/model_explanations.json`.

Addresses: P4 — turns the explainability primitives into a committed,
versioned artifact so a reviewer reads the same explanation the authors did,
rather than one regenerated on their machine from whatever data they happen to
hold.

Scope discipline: this runner FITS NOTHING NEW in the research sense. It
re-fits the already-selected configurations at one rebalance date in order to
inspect them. No hyperparameter is chosen here, no model is compared, and no
result in any other artifact can move because of it.

The rebalance date explained is the LAST month-end of each universe's history
— the decision a reviewer would ask about first ("what would this system hold
today, and why"). It is derived from the data, never hardcoded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from explainability import (
    explain_expected_return_ranking,
    feature_distribution_summary,
    local_contributions,
    permutation_importance,
    trace_regime_decision,
)

log = logging.getLogger("run_explainability")

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
ARTIFACT = GOLD / "model_explanations.json"

UNIVERSES = {
    "etf_2017": ("log_returns_etf.parquet", "ml_features_etf.parquet"),
    "full_2021": ("log_returns.parquet", "ml_features_full.parquet"),
}
CHALLENGERS = {"rf_signal_tuned": "random_forest", "xgb_signal_tuned": "xgboost"}


def _params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))


def _last_rebalance_date(returns: pd.DataFrame, freq: str) -> pd.Timestamp:
    """The final rebalance boundary in the data — derived, not chosen."""
    return returns.resample(freq).last().index[-1]


def explain_universe(universe: str, params: dict) -> dict:
    """Every explanation for one universe at its last rebalance date."""
    returns_file, features_file = UNIVERSES[universe]
    returns = pd.read_parquet(GOLD / returns_file)
    features = pd.read_parquet(GOLD / features_file)

    backtest, regime = params["backtest"], params["regime"]
    tau = min(_last_rebalance_date(returns, backtest["rebalance_freq"]), returns.index[-1])
    train_returns = returns.loc[:tau]
    train_features = features.loc[:tau]

    log.info("%s: explaining decision at %s (%d rows)", universe, tau.date(), len(train_returns))

    result = {
        "decision_date": str(tau.date()),
        "primary": trace_regime_decision(
            train_returns,
            train_features,
            n_states=regime["n_states"],
            n_restarts=regime["n_restarts"],
            random_state_base=regime["random_state_base"],
            covariance_type=regime["covariance_type"],
            min_regime_train_days=regime["min_regime_train_days"],
            regime_features=regime["features"],
            bull_strategy=regime["bull_strategy"],
            bear_strategy=regime["bear_strategy"],
            max_weight=backtest["max_weight"],
            risk_free_annual=backtest["risk_free_annual"],
        ),
        "challengers": {},
    }

    for name, model_type in CHALLENGERS.items():
        try:
            result["challengers"][name] = explain_challenger(
                name, model_type, universe, train_returns, train_features, params
            )
        except Exception as exc:  # noqa: BLE001 — recorded, never silent
            log.warning("%s: %s explanation unavailable: %s", universe, name, exc)
            result["challengers"][name] = {"unavailable": str(exc)}
    return result


def explain_challenger(
    name: str,
    model_type: str,
    universe: str,
    train_returns: pd.DataFrame,
    train_features: pd.DataFrame,
    params: dict,
) -> dict:
    """Global importance, local contributions and the ranking→weights link."""
    from ml_signals import (
        attach_regime_feature,
        build_asset_features,
        build_supervised_dataset,
        melt_to_panel,
    )
    from strategies import RandomForestSignalStrategy, XGBoostSignalStrategy

    regime, backtest = params["regime"], params["backtest"]
    selected = json.loads((GOLD / "phase5_results.json").read_text())[universe]
    model_params = dict(selected["tuned"][name]["selected_ml_params"])
    levers = dict(selected["tuned"][name]["selected_levers"])

    wide = build_asset_features(train_returns)
    panel = melt_to_panel(wide, list(train_returns.columns))
    panel = attach_regime_feature(
        panel,
        train_features,
        n_states=regime["n_states"],
        n_restarts=regime["n_restarts"],
        random_state_base=regime["random_state_base"],
        covariance_type=regime["covariance_type"],
        min_regime_train_days=regime["min_regime_train_days"],
        features=regime["features"],
    )
    X, y, X_predict = build_supervised_dataset(panel, train_returns)
    if X.empty or X_predict.empty:
        raise ValueError("insufficient panel rows to fit an explainable model")

    if model_type == "xgboost":
        from xgboost import XGBRegressor

        # n_jobs=1 is a standing policy in this project, not a performance
        # choice: multi-threaded xgboost after a scikit-learn workload has
        # segfaulted on this platform.
        model = XGBRegressor(random_state=0, n_jobs=1, **model_params)
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=0, n_jobs=1, **model_params)
    model.fit(X.to_numpy(), y.to_numpy())

    predicted = pd.Series(model.predict(X_predict.to_numpy()), index=X_predict.index)
    predicted.index = [asset for _, asset in predicted.index]

    # The weights must be THIS challenger's own, produced from its predicted mu
    # under its selected levers. Using a sample-mean optimizer here would
    # compare the prediction against a portfolio that never saw it — which is
    # precisely the link this section exists to expose.
    strategy_class = (
        XGBoostSignalStrategy if model_type == "xgboost" else RandomForestSignalStrategy
    )
    weights = strategy_class(
        max_weight=backtest["max_weight"],
        risk_free_annual=backtest["risk_free_annual"],
        model_params=model_params,
        n_states=regime["n_states"],
        n_restarts=regime["n_restarts"],
        random_state_base=regime["random_state_base"],
        covariance_type=regime["covariance_type"],
        min_regime_train_days=regime["min_regime_train_days"],
        mu_transform="shrink" if levers.get("shrinkage_weight") else "none",
        shrinkage_weight=levers.get("shrinkage_weight", 0.5),
    ).fit(train_returns, {"features": train_features})

    # One representative row: the first asset of the scored row, alphabetically,
    # so the choice is reproducible rather than incidental to dict ordering.
    explained_asset = sorted(predicted.index)[0]
    row_position = list(predicted.index).index(explained_asset)

    return {
        "model_type": model_type,
        "selected_parameters": model_params,
        "training_rows": int(len(X)),
        "feature_names": [str(c) for c in X.columns],
        # A ranked LIST, not a dict: the artifact is written with sort_keys=True
        # for stable diffs, which would re-alphabetize a dict and silently
        # destroy the very ordering this section exists to convey.
        "global_importance_permutation": [
            {"feature": name, "mse_increase": value, "rank": rank}
            for rank, (name, value) in enumerate(
                permutation_importance(model, X, y, seed=0).items(), start=1
            )
        ],
        "local_contributions": {
            "asset": explained_asset,
            **local_contributions(model, X_predict.iloc[row_position], model_type),
        },
        "feature_distribution": feature_distribution_summary(X),
        "ranking_to_weights": explain_expected_return_ranking(
            predicted, weights, backtest["max_weight"]
        ),
        "attribution_caveat": (
            "Attribution describes what the fitted function does with its inputs. "
            "It does not identify a cause of any market return."
        ),
    }


def main() -> Path:
    params = _params()
    payload = {
        "generated_from_committed_gold": True,
        "explanation_policy": {
            "primary_system": "regime_conditional",
            "primary_method": "deterministic decision trace (fits no predictive function)",
            "challenger_method": "exact additive tree contributions + seeded permutation importance",
            "shap_package_used": False,
            "shap_rationale": (
                "xgboost ships TreeSHAP natively and scikit-learn's tree API supports "
                "the exact path decomposition, so exact attribution needs no extra "
                "dependency; `shap` would add a numba/llvmlite native stack to a "
                "project that already withdrew a model over a native-library conflict."
            ),
            "causality": "No explanation in this artifact supports a causal claim.",
        },
        "universes": {u: explain_universe(u, params) for u in UNIVERSES},
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ARTIFACT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(main().relative_to(ROOT))
