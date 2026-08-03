"""
explainability.py — reconstruct why a portfolio holds what it holds.

Addresses: P4 — a result that cannot be reconstructed cannot be reviewed. This
module produces, for one rebalance date, the evidence a risk reviewer needs to
follow an allocation from raw inputs to final weights.

TWO SYSTEMS, TWO DIFFERENT EXPLANATIONS — and the asymmetry is the point
---------------------------------------------------------------------------
`regime_conditional` is the project's reference system and it fits NO
predictive function over features: it classifies a market regime and hands the
decision to an already-validated Markowitz optimizer. Feature attribution
would therefore explain nothing about it. Its explanation is a deterministic
DECISION TRACE — posterior, chosen regime, chosen sub-optimizer, the moment
inputs that optimizer received, which constraints were BINDING, and the
resulting weights. Every step is a function of data available at τ.

The F7 challengers (`rf_signal`, `xgb_signal`) do fit a predictive function, so
attribution is the right tool there, and it is applied to them alone.

WHY NOT SHAP THE PACKAGE
------------------------
`shap` is not installed, and installing it was rejected. Two reasons, in order
of weight:

1. Exact tree attribution is ALREADY available without it. XGBoost ships
   TreeSHAP natively (`Booster.predict(..., pred_contribs=True)`), and for a
   scikit-learn forest the exact path decomposition (Saabas) is ~30 lines
   against the public `tree_` API. Both are exactly additive — contributions
   plus bias reproduce the prediction to float precision — which is what makes
   `test_explainability.py`'s consistency assertion a real check rather than a
   tolerance-tuning exercise.
2. `shap` pulls in numba/llvmlite. This project has already lost a model to a
   native-library conflict: `LSTMSignalStrategy` was fully built and tested,
   then withdrawn after `torch` and `xgboost` segfaulted in one process. Adding
   another native stack to buy a capability we already have is a bad trade.

The trade-off being accepted, stated plainly: the two models' local
contributions come from different exact algorithms, so their MAGNITUDES are not
directly comparable across models. Rankings within a model are. Global
importance uses one seeded permutation procedure for both, so that IS
comparable.

ATTRIBUTION IS NOT CAUSATION. Everything here describes what a fitted function
does with its inputs. None of it identifies a cause of a market return.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("explainability")

# Weights within this of the cap are reported as resting ON it. SLSQP returns a
# numerically-converged optimum, not an exact boundary value, so an equality
# test would report "no constraint was binding" for a portfolio pinned to the
# cap on four of five assets.
CAP_TOLERANCE = 1e-6


# ── Primary system: deterministic decision trace ─────────────────────────────
def trace_regime_decision(
    train_returns: pd.DataFrame,
    features: pd.DataFrame,
    *,
    n_states: int = 2,
    n_restarts: int = 5,
    random_state_base: int = 0,
    covariance_type: str = "diag",
    min_regime_train_days: int = 252,
    regime_features: list[str] | None = None,
    bull_strategy: str = "max_sharpe",
    bear_strategy: str = "min_variance_lw",
    max_weight: float = 0.25,
    risk_free_annual: float = 0.0,
) -> dict:
    """Reconstruct one `regime_conditional` allocation decision, end to end.

    Addresses: P2, P3, P4 — the explanation for the reference system. Mirrors
    `RegimeConditionalStrategy.fit`'s logic exactly, including its fallback
    policy, so the trace describes the decision that WOULD be taken rather
    than a plausible reconstruction of it.

    Args:
        train_returns: Log-returns sliced to `:τ`, as the engine supplies them.
        features: Phase 3 market features, also sliced to `:τ`.
        Remaining arguments mirror `RegimeConditionalStrategy`'s constructor.

    Returns:
        A JSON-serialisable trace: inputs at τ, posterior, selected regime and
        sub-optimizer, the moment inputs, binding constraints, final weights,
        and any fallback with its reason.
    """
    from regime import REGIME_FEATURES, fit_hmm, predict_regime_posterior
    from strategies import MaxSharpe, MinVarianceLW, estimate_covariance

    regime_features = list(regime_features or REGIME_FEATURES)
    decision_date = train_returns.index[-1]

    # Step 1 — the inputs the HMM actually sees at τ.
    hmm_inputs = {
        name: (None if pd.isna(v) else float(v))
        for name, v in features[regime_features].iloc[-1].items()
    }

    # Step 2 — posterior, and whether the fit is trustworthy.
    hmm_fit = fit_hmm(
        features,
        n_states=n_states,
        n_restarts=n_restarts,
        random_state_base=random_state_base,
        covariance_type=covariance_type,
        min_regime_train_days=min_regime_train_days,
        features=regime_features,
    )
    posterior = predict_regime_posterior(hmm_fit, features, features=regime_features)

    # Step 3 — the regime, and the fallback policy when the fit did not converge.
    fallback_used = not hmm_fit.converged
    if fallback_used:
        regime_label = "bear"
        fallback_reason = (
            f"HMM did not converge (or fewer than {min_regime_train_days} usable "
            f"training days). The neutral 50/50 posterior resolves to the DEFENSIVE "
            f"sub-strategy rather than an arbitrary tie-break: when the regime read "
            f"is uncertain, do not guess bullish."
        )
    else:
        regime_label = max(posterior, key=posterior.get)
        fallback_reason = None

    # Step 4 — which optimizer received the decision, and on what moments.
    selected = bull_strategy if regime_label == "bull" else bear_strategy
    covariance_estimator = "ledoit_wolf"
    cov = estimate_covariance(train_returns, estimator=covariance_estimator)
    strategy = (
        MaxSharpe(max_weight=max_weight, risk_free_annual=risk_free_annual)
        if selected == "max_sharpe"
        else MinVarianceLW(max_weight=max_weight)
    )
    weights = strategy.fit(train_returns, {"features": features})

    # Step 5 — the moments, summarised. Full matrices belong in the run, not
    # in an audit artifact a human is expected to read.
    mu_annual = train_returns.mean() * 252
    vol_annual = pd.Series(np.sqrt(np.diag(cov)), index=train_returns.columns)

    # Step 6 — which constraints BIND. On a 5-asset universe at a 25% cap this
    # is very nearly the whole allocation decision, so it is reported as a
    # first-class part of the trace rather than inferred by the reader.
    at_cap = [a for a, w in weights.items() if w >= max_weight - CAP_TOLERANCE]
    at_zero = [a for a, w in weights.items() if w <= CAP_TOLERANCE]
    n_assets = train_returns.shape[1]

    return {
        "system": "regime_conditional",
        "explanation_method": "deterministic decision trace",
        "decision_date": str(pd.Timestamp(decision_date).date()),
        "n_assets": int(n_assets),
        "hmm": {
            "features": regime_features,
            "inputs_at_decision_date": hmm_inputs,
            "n_states": int(n_states),
            "converged": bool(hmm_fit.converged),
            "log_likelihood": float(hmm_fit.log_likelihood),
            "seed_used": hmm_fit.seed_used,
            "state_label_map": {str(k): v for k, v in (hmm_fit.label_map or {}).items()},
            "posterior": {k: float(v) for k, v in posterior.items()},
            "label_mapping_note": (
                "hmmlearn does not guarantee stable state ordering across fits. "
                "States are mapped to bull/bear by ranking each state's fitted "
                "MARKET_RETURN mean, recomputed at every refit."
            ),
        },
        "decision": {
            "selected_regime": regime_label,
            "selected_sub_optimizer": selected,
            "bull_sub_optimizer": bull_strategy,
            "bear_sub_optimizer": bear_strategy,
        },
        "moment_inputs": {
            "covariance_estimator": covariance_estimator,
            "annualized_mean_return": {a: float(v) for a, v in mu_annual.items()},
            "annualized_volatility": {a: float(v) for a, v in vol_annual.items()},
            "training_rows": int(len(train_returns)),
            "training_start": str(train_returns.index[0].date()),
        },
        "constraints": {
            "max_weight": float(max_weight),
            "long_only": True,
            "assets_at_cap": at_cap,
            "assets_at_zero": at_zero,
            "n_binding_at_cap": len(at_cap),
            "cap_is_near_determining": bool(n_assets * max_weight < 1.5),
            "binding_note": (
                f"{n_assets} assets x {max_weight:.0%} cap = {n_assets * max_weight:.2f}. "
                "When this product is close to 1, every feasible long-only portfolio "
                "must hold most assets at the cap, so the CONSTRAINT rather than the "
                "covariance model determines the allocation."
            ),
        },
        "weights": {a: float(w) for a, w in weights.items()},
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
    }


# ── Challengers: exact additive local contributions ──────────────────────────
def _rf_path_contributions(model, x_row: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact per-feature contributions for one row of a scikit-learn forest.

    Walks each tree's decision path and credits the change in node value to
    the feature that was split on, averaged over trees (Saabas decomposition).
    Exactly additive by construction: `bias + contributions.sum()` reproduces
    `model.predict(x_row)`, which `test_explainability.py` asserts.
    """
    n_features = x_row.shape[0]
    total = np.zeros(n_features, dtype=float)
    bias = 0.0
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value.reshape(-1)
        node, bias = 0, bias + float(values[0])
        while tree.children_left[node] != -1:
            feature = tree.feature[node]
            child = (
                tree.children_left[node]
                if x_row[feature] <= tree.threshold[node]
                else tree.children_right[node]
            )
            total[feature] += float(values[child]) - float(values[node])
            node = child
    n_trees = len(model.estimators_)
    return total / n_trees, bias / n_trees


def local_contributions(model, x_row: pd.Series, model_type: str) -> dict:
    """Exact additive contributions of each feature to one prediction.

    Addresses: P4 — `xgboost` supplies TreeSHAP natively and scikit-learn's
    tree API supports the exact path decomposition, so neither path needs the
    `shap` package (see module docstring). Both are additive, so the returned
    `reconstruction_error` is a genuine self-check rather than decoration.
    """
    x = x_row.to_numpy(dtype=float).reshape(1, -1)
    if model_type == "xgboost":
        import xgboost as xgb

        contribs = model.get_booster().predict(
            xgb.DMatrix(x, feature_names=list(x_row.index)), pred_contribs=True
        )[0]
        values, bias = contribs[:-1], float(contribs[-1])
        method = "TreeSHAP (exact, xgboost native)"
    else:
        values, bias = _rf_path_contributions(model, x[0])
        method = "decision-path decomposition (exact, Saabas)"

    prediction = float(model.predict(x)[0])
    return {
        "method": method,
        "bias": bias,
        "prediction": prediction,
        "reconstruction_error": abs(bias + float(values.sum()) - prediction),
        "contributions": {
            name: float(v) for name, v in zip(x_row.index, values)
        },
    }


def permutation_importance(
    model, X: pd.DataFrame, y: pd.Series, *, n_repeats: int = 5, seed: int = 0
) -> dict[str, float]:
    """Deterministic permutation importance: mean MSE increase per feature.

    Addresses: P4 — one procedure for both model families, so global rankings
    ARE comparable across them (unlike the local contributions, which come
    from two different exact algorithms). Seeded explicitly: an unseeded
    explanation that changes between runs cannot be audited.
    """
    rng = np.random.default_rng(seed)
    baseline = float(np.mean((model.predict(X.to_numpy()) - y.to_numpy()) ** 2))
    scores: dict[str, float] = {}
    for column in X.columns:
        losses = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            losses.append(
                float(np.mean((model.predict(shuffled.to_numpy()) - y.to_numpy()) ** 2))
            )
        scores[column] = float(np.mean(losses) - baseline)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def feature_distribution_summary(X: pd.DataFrame) -> dict:
    """Per-feature location, spread, tails and missingness of the training matrix."""
    described = X.describe().T
    return {
        str(name): {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "min": float(row["min"]),
            "p25": float(row["25%"]),
            "median": float(row["50%"]),
            "p75": float(row["75%"]),
            "max": float(row["max"]),
            "missing": int(X[name].isna().sum()),
        }
        for name, row in described.iterrows()
    }


def explain_expected_return_ranking(
    predicted_mu: pd.Series, weights: pd.Series, max_weight: float
) -> dict:
    """Link the predicted ranking to the weights it produced.

    Addresses: P4 — the step most often left implicit. A model can rank assets
    well and still produce weights that ignore the ranking, because the cap and
    the covariance term both intervene; showing rank alongside weight rank makes
    that visible instead of assumed.
    """
    mu_rank = predicted_mu.rank(ascending=False, method="min")
    weight_rank = weights.rank(ascending=False, method="min")
    rows = {
        asset: {
            "predicted_expected_return": float(predicted_mu[asset]),
            "predicted_rank": int(mu_rank[asset]),
            "weight": float(weights[asset]),
            "weight_rank": int(weight_rank[asset]),
            "at_cap": bool(weights[asset] >= max_weight - CAP_TOLERANCE),
        }
        for asset in predicted_mu.index
    }
    # Undefined, not zero, when either ranking is constant. That is not a
    # corner case here: a binding cap can put EVERY asset at the same weight
    # (5 assets x 25% leaves only 4 free), and reporting NaN — or worse, 0.0 —
    # would read as "the weights ignore the ranking" when the truth is that the
    # constraint left no ranking to express.
    degenerate = mu_rank.nunique() < 2 or weight_rank.nunique() < 2
    concordance = (
        None if degenerate
        else float(pd.Series(mu_rank).corr(pd.Series(weight_rank), method="spearman"))
    )
    return {
        "per_asset": rows,
        "rank_concordance_spearman": concordance,
        "concordance_undefined_reason": (
            "Every asset shares one rank on at least one side — with a binding cap "
            "the allocation carries no ordering to compare against."
            if degenerate else None
        ),
        "note": (
            "Concordance below 1 does not indicate a defect. The optimizer trades "
            "predicted return against covariance and is bounded by the weight cap, "
            "so weights are not a monotone function of predicted return."
        ),
    }
