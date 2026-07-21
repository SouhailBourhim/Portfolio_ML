"""
model_selection.py — Leak-free hyperparameter selection (Phase 5).

Addresses: P4 — Phase 4C left `shrinkage_weight` and `turnover_penalty`
CHOSEN, not tuned, and proved (via the rf-vs-xgb asymmetry) that a single
global penalty is wrong. Selecting them by eyeballing the test set would be
the exact backtest overfitting the project exists to avoid. This module
selects hyperparameters using ONLY training data, with the right tool for
each objective — the methodology, not just the code, is the deliverable.

Two objectives, two tools (a deliberate methodology call):

  1. ML PREDICTION hyperparameters (RF/XGB depth, leaf size, …) are an
     IID-ish labeled-sample problem → `select_ml_hyperparameters` uses
     `PurgedKFold` (leakage-free CV), scored by INFORMATION COEFFICIENT
     (Spearman rank-corr of predicted vs. realized next-period returns —
     the right metric for a *signal*, where the ordering matters more than
     the magnitude). This is López de Prado's textbook use of purged CV.

  2. PORTFOLIO-CONSTRUCTION levers (`shrinkage_weight`, a per-model
     `turnover_penalty`) are a *net-Sharpe over a return time series*
     objective, not a labeled-sample one → `select_portfolio_levers` scores
     each grid point by the net Sharpe of a real walk-forward backtest over
     a VALIDATION segment (the out-of-sample tail of the training window).
     Using K-Fold here would be a category error — portfolio returns are a
     dependent time series, not exchangeable samples.

The frozen TEST segment (chosen in `run_phase5.py`) is never touched by
either selector; both see only train+validation data.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger("model_selection")


def information_coefficient(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Spearman rank correlation between realized and predicted returns.

    Addresses: P4 — the signal-quality score for `select_ml_hyperparameters`.
    Rank (not linear) correlation because a return-prediction signal is used
    for cross-sectional ORDERING (who to overweight), so its ranking is what
    must generalize; the predicted magnitudes are far less reliable (the same
    reason Phase 4C's `rank` mu-transform exists). Returns 0.0 for a
    degenerate (constant) input rather than NaN, so a dead grid point scores
    as useless, not as missing.
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return 0.0
    rho, _ = spearmanr(a[mask], b[mask])
    return float(rho) if np.isfinite(rho) else 0.0


def _build_panel_xy(
    train_returns: pd.DataFrame,
    features: pd.DataFrame | None,
    short_window: int,
    long_window: int,
    momentum_windows: Sequence[int],
    condition_on_regime: bool,
    regime_kwargs: Mapping,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the pooled (X, y) supervised panel, reusing ml_signals unchanged."""
    from ml_signals import (
        attach_regime_feature,
        build_asset_features,
        build_supervised_dataset,
        melt_to_panel,
    )

    wide = build_asset_features(
        train_returns,
        short_window=short_window,
        long_window=long_window,
        momentum_windows=momentum_windows,
    )
    panel = melt_to_panel(wide, list(train_returns.columns))
    if condition_on_regime and features is not None:
        panel = attach_regime_feature(panel, features, **regime_kwargs)
    X, y, _ = build_supervised_dataset(panel, train_returns)
    return X, y


def _instantiate(model_type: str, params: Mapping):
    resolved = {"random_state": 0, **dict(params)}
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**resolved)
    if model_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**resolved)
    raise ValueError(f"Unknown model_type: {model_type!r}.")


def _grid_points(grid: Mapping[str, Sequence]) -> list[dict]:
    """Cartesian product of a {param: [values]} grid → list of param dicts."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def select_ml_hyperparameters(
    train_returns: pd.DataFrame,
    features: pd.DataFrame | None,
    grid: Mapping[str, Sequence],
    model_type: str = "random_forest",
    n_splits: int = 5,
    embargo_frac: float = 0.01,
    short_window: int = 21,
    long_window: int = 63,
    momentum_windows: Sequence[int] = (5, 21, 63),
    condition_on_regime: bool = True,
    regime_kwargs: Mapping | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Purged-CV selection of RF/XGB hyperparameters, scored by mean fold IC.

    Addresses: P4 — the leakage-free ML selection. Builds the pooled panel
    once, then for each grid point runs `PurgedKFold`, fits on each train
    fold and predicts on the purged test fold, and scores the fold by the
    information coefficient of predicted vs. realized returns. The grid point
    with the highest mean IC wins.

    Args:
        train_returns: Log-returns of the TRAIN+VALIDATION window only (never
            the frozen test segment).
        features: Phase 3 market-level features for regime conditioning.
        grid: `{param: [values]}` searched as a Cartesian product.
        model_type, n_splits, embargo_frac, short/long/momentum windows,
            condition_on_regime, regime_kwargs: see module / `ml_signals`.

    Returns:
        (best_params, cv_table) — the winning param dict and a DataFrame of
        every grid point with its mean/std IC across folds (for the notebook).
    """
    from purged_kfold import PurgedKFold

    X, y = _build_panel_xy(
        train_returns, features, short_window, long_window, momentum_windows,
        condition_on_regime, regime_kwargs or {},
    )
    cv = PurgedKFold(n_splits=n_splits, embargo_frac=embargo_frac, label_horizon=1)
    splits = list(cv.split(X))

    rows = []
    for params in _grid_points(grid):
        fold_ics = []
        for train_idx, test_idx in splits:
            model = _instantiate(model_type, params)
            model.fit(X.iloc[train_idx].to_numpy(), y.iloc[train_idx].to_numpy())
            pred = model.predict(X.iloc[test_idx].to_numpy())
            fold_ics.append(information_coefficient(y.iloc[test_idx].to_numpy(), pred))
        rows.append({
            **params,
            "mean_ic": float(np.mean(fold_ics)),
            "std_ic": float(np.std(fold_ics)),
            "n_folds": len(fold_ics),
        })

    cv_table = pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)
    best = {k: cv_table.iloc[0][k] for k in grid}
    # Restore int types the grid declared (DataFrame upcasts to float/object).
    best = {k: (int(v) if isinstance(list(grid[k])[0], int) else v) for k, v in best.items()}
    log.info("select_ml_hyperparameters(%s): best %s (IC %.4f)",
             model_type, best, cv_table.iloc[0]["mean_ic"])
    return best, cv_table


def select_portfolio_levers(
    train_returns: pd.DataFrame,
    features: pd.DataFrame,
    model_type: str,
    ml_params: Mapping,
    shrink_grid: Sequence[float],
    penalty_grid: Sequence[float],
    backtest_kwargs: Mapping,
    ledger=None,
    universe: str = "",
) -> tuple[dict, pd.DataFrame]:
    """
    Validation-segment selection of (shrinkage_weight, turnover_penalty).

    Addresses: P4 — the portfolio levers are a net-Sharpe objective, selected
    by a REAL walk-forward backtest over the training window's out-of-sample
    tail (the validation segment the engine produces after its warm-up), not
    by K-Fold. Per-model, honoring Phase 4C's finding that the right penalty
    differs between RF and XGB. Every grid point's net-return series is
    recorded to the DSR ledger (if given), so the final deflation counts this
    search honestly.

    Args:
        train_returns: TRAIN+VALIDATION log-returns (never the test segment).
        features: Phase 3 features passed as `extras={"features": ...}`.
        model_type: "random_forest" or "xgboost".
        ml_params: the hyperparameters chosen by `select_ml_hyperparameters`.
        shrink_grid, penalty_grid: candidate lever values.
        backtest_kwargs: forwarded to `run_backtest` (rebalance_freq,
            min_train_days, cost_bps, max_weight, universe_name) and used to
            build the strategy (max_weight, risk_free_annual, min_train_rows,
            feature-window params).
        ledger: optional `metrics.DSRTrialLedger`.
        universe: ledger key.

    Returns:
        (best_levers, table) — `{"shrinkage_weight", "turnover_penalty"}` and
        a DataFrame of every grid point's validation net Sharpe.
    """
    from backtest import run_backtest
    from metrics import annualized_sharpe
    from strategies import RandomForestSignalStrategy, XGBoostSignalStrategy

    cls = RandomForestSignalStrategy if model_type == "random_forest" else XGBoostSignalStrategy
    bt = dict(backtest_kwargs)
    strat_common = dict(
        max_weight=bt["max_weight"],
        risk_free_annual=bt.get("risk_free_annual", 0.0),
        model_params=dict(ml_params),
        min_train_rows=bt.get("min_train_rows", 504),
        short_window=bt.get("short_window", 21),
        long_window=bt.get("long_window", 63),
        momentum_windows=bt.get("momentum_windows", (5, 21, 63)),
        condition_on_regime=bt.get("condition_on_regime", True),
    )

    rows = []
    for shrink, penalty in itertools.product(shrink_grid, penalty_grid):
        strategy = cls(
            name=f"{cls.name}__val",
            mu_transform="shrink", shrinkage_weight=float(shrink),
            turnover_penalty=float(penalty), **strat_common,
        )
        result = run_backtest(
            train_returns, strategy,
            rebalance_freq=bt["rebalance_freq"],
            min_train_days=bt["min_train_days"],
            cost_bps=bt["cost_bps"],
            extras={"features": features},
            universe_name=bt.get("universe_name", ""),
            max_weight=bt["max_weight"],
        )
        val_sharpe = annualized_sharpe(result.net_returns, bt.get("risk_free_annual", 0.0))
        rows.append({
            "shrinkage_weight": float(shrink),
            "turnover_penalty": float(penalty),
            "val_sharpe_net": round(float(val_sharpe), 4),
            "avg_turnover": round(float(result.turnover.mean()), 4),
        })
        if ledger is not None:
            ledger.record(universe, result.net_returns)

    table = pd.DataFrame(rows).sort_values("val_sharpe_net", ascending=False).reset_index(drop=True)
    best = {
        "shrinkage_weight": float(table.iloc[0]["shrinkage_weight"]),
        "turnover_penalty": float(table.iloc[0]["turnover_penalty"]),
    }
    log.info("select_portfolio_levers(%s): best %s (val Sharpe %.4f)",
             model_type, best, table.iloc[0]["val_sharpe_net"])
    return best, table
