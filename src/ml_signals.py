"""
ml_signals.py — Phase 4B / F7: per-asset feature panel + supervised return
prediction (RandomForest, XGBoost; LSTM lives in lstm_signal.py).

Addresses: P1, P2, P3 — F4's HMM regime detection switches between existing
Markowitz optimizers; it does not predict returns. F7 is the missing piece:
a supervised model that learns to forecast each asset's next-period return,
conditioned on the detected regime, feeding a better `mu` into the same
Sharpe-maximization objective every prior Phase 4 covariance rung already
reused (P1 — replacing the noisy sample mean with a learned estimate; P2 —
the model conditions on regime and rolling features rather than assuming a
static return distribution; P3 — the pooled panel sees every asset's
behavior around the same regime shifts).

No per-asset feature matrix existed anywhere in this codebase before this
module — `ml_features.py`'s features are all market-level cross-sectional
aggregates (one number per day for the whole universe). This module builds
per-asset features from scratch, deliberately small (a few trailing-return
windows, realized volatility, price-relative-to-moving-average) rather than
a kitchen sink, matching this project's MVP-first convention.

Causality convention — identical to `ml_features.build_return_features`:
features at date t use only observations up to and including t (all known
at t's close), and are NOT lagged, because the engine fits at the close of
rebalance date τ and only earns returns from τ+1 — feature[τ] legitimately
drives a τ+1 decision. This holds trivially for the rolling constructs below
(a pandas `.rolling()` window at position i is a function of rows ≤ i only,
by definition), so precomputing over the FULL history once and later
slicing the *output* to `:τ` (exactly what `backtest.run_backtest`'s
`extras` mechanism already does for every other feature frame) is provably
identical to recomputing on a `:τ`-sliced input at every rebalance — locked
in by `test_future_returns_do_not_change_past_asset_features`.

Label construction (`build_supervised_dataset`) is the single highest-risk
correctness point in this module: a rebalance date τ's own next-period
return is not yet realized, so it is structurally excluded from the
training set, not merely dropped by a NaN filter that could be bypassed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("ml_signals")

ROOT = Path(__file__).resolve().parents[1]
TRADING_DAYS_PER_YEAR = 252


def _validate_log_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Minimal defensive validation — mirrors ml_features._validate_returns."""
    if not isinstance(log_returns, pd.DataFrame) or log_returns.empty:
        raise ValueError("log_returns must be a non-empty DataFrame.")
    if not isinstance(log_returns.index, pd.DatetimeIndex):
        raise TypeError("log_returns must use a DatetimeIndex.")
    if not log_returns.index.is_monotonic_increasing:
        raise ValueError("log_returns index must be sorted ascending.")
    if log_returns.index.has_duplicates:
        raise ValueError("log_returns index contains duplicate dates.")
    validated = log_returns.astype(float).copy()
    validated.index.name = "Date"
    return validated


def build_asset_features(
    log_returns: pd.DataFrame,
    short_window: int = 21,
    long_window: int = 63,
    momentum_windows: list[int] = (5, 21, 63),
) -> pd.DataFrame:
    """Build a wide, per-asset causal feature panel from a return matrix.

    Addresses: P1, P2 — per-asset trailing return, realized volatility, and
    price-relative-to-moving-average describe each instrument's own recent
    behavior, the raw material a return-prediction model needs; nothing here
    is a cross-sectional aggregate (that's `ml_features.py`'s job).

    One column set per asset: `{TICKER}__RET_{W}D` for each window in
    `momentum_windows` (trailing W-day cumulative log return — short windows
    read as reversal, longer windows as momentum); `{TICKER}__VOL_{W}D` for
    `short_window`/`long_window` (annualized realized volatility, same
    windows `ml_features.py` uses for the market-level equivalent, for
    consistency); `{TICKER}__PRICE_REL_MA_{long_window}D` (cumulative
    wealth relative to its own trailing moving average — a mean-reversion
    signal, computed entirely from returns since `Strategy.fit()` never
    receives raw prices).

    Args:
        log_returns: Clean log-return matrix, dates × assets.
        short_window: Trailing window for short realized volatility.
        long_window: Trailing window for long realized volatility and the
            price-relative-to-moving-average signal.
        momentum_windows: Trailing windows for cumulative-return features.

    Returns:
        DataFrame aligned to log_returns.index. Leading rows are NaN until
        each rolling window has enough history — never dropped here (the
        caller decides warm-up policy, same division of responsibility as
        `ml_features.build_return_features` vs. `build_ml_feature_set`).
    """
    returns = _validate_log_returns(log_returns)
    momentum_windows = list(momentum_windows)
    for name, value in {"short_window": short_window, "long_window": long_window}.items():
        if value < 2:
            raise ValueError(f"{name} must be >= 2.")
    for window in momentum_windows:
        if window < 2:
            raise ValueError("Every momentum window must be >= 2.")

    assets = list(returns.columns)
    columns: dict[str, pd.Series] = {}

    for asset in assets:
        series = returns[asset]
        wealth = np.exp(series.cumsum())

        for window in momentum_windows:
            columns[f"{asset}__RET_{window}D"] = series.rolling(
                window, min_periods=window
            ).sum()

        for window in (short_window, long_window):
            columns[f"{asset}__VOL_{window}D"] = series.rolling(
                window, min_periods=window
            ).std() * np.sqrt(252.0)

        moving_avg = wealth.rolling(long_window, min_periods=long_window).mean()
        columns[f"{asset}__PRICE_REL_MA_{long_window}D"] = wealth / moving_avg - 1.0

    features = pd.DataFrame(columns, index=returns.index)
    features = features.replace([np.inf, -np.inf], np.nan)
    features.index.name = "Date"
    return features


def melt_to_panel(wide_features: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Reshape a wide per-asset feature frame into a long (Date, ASSET) panel.

    Addresses: P1 — the pooled cross-sectional training format: every
    asset's rows are stacked together (asset identity carried by the index,
    not a `{TICKER}__` column prefix), multiplying effective training rows
    by the asset count instead of fitting one starved model per instrument.

    Pure reshape — no new rolling-window computation, so it is safe to call
    on an already-`:τ`-sliced frame inside `Strategy.fit()` at every
    rebalance without redoing any causal-window logic.

    Args:
        wide_features: Output of `build_asset_features` (or an already-
            `:τ`-sliced subset of it) — columns named `{TICKER}__FEATURE`.
        assets: The asset universe, in the order the caller cares about
            (typically `train_returns.columns`).

    Returns:
        Long DataFrame with a `(Date, ASSET)` MultiIndex and generic feature
        columns (the `{TICKER}__` prefix stripped). Assets absent from
        `wide_features` (no matching columns) are silently skipped — not an
        error, since a universe can shrink between panel-build time and a
        later rebalance only in ways the engine already governs.
    """
    frames = []
    for asset in assets:
        prefix = f"{asset}__"
        asset_columns = [c for c in wide_features.columns if c.startswith(prefix)]
        if not asset_columns:
            continue
        sub = wide_features[asset_columns].copy()
        sub.columns = [c[len(prefix) :] for c in asset_columns]
        sub["ASSET"] = asset
        frames.append(sub)

    if not frames:
        raise ValueError(
            f"None of {assets} have matching '{{TICKER}}__' columns in wide_features."
        )

    panel = pd.concat(frames, axis=0)
    panel.index.name = "Date"
    panel = panel.reset_index().set_index(["Date", "ASSET"]).sort_index()
    return panel


def attach_regime_feature(
    panel: pd.DataFrame,
    market_features: pd.DataFrame | None,
    n_states: int = 2,
    n_restarts: int = 5,
    random_state_base: int = 0,
    covariance_type: str = "diag",
    min_regime_train_days: int = 252,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Broadcast the HMM's bull-probability onto every asset row for that date.

    Addresses: P2, P3 — lets a pooled tree-based model learn regime-
    dependent splits natively (a data-efficient reading of "regime-
    conditional") instead of fragmenting the training set into separate
    per-regime models. Regime is market-wide, so a one-to-many join from
    one posterior-per-date onto every asset for that date is legitimate,
    not an approximation.

    Reuses `regime.fit_hmm` / `regime.predict_regime_posterior_series`
    exactly like `RegimeConditionalStrategy` reuses `fit_hmm` /
    `predict_regime_posterior` — no new HMM logic here. Parameters mirror
    `fit_hmm`'s own signature exactly (rather than accepting a generic
    kwargs dict) so a caller can't accidentally leak an unrelated key —
    e.g. `regime.bull_strategy` from `params.yaml` — into the HMM fit.

    Args:
        panel: Output of `melt_to_panel` — `(Date, ASSET)` MultiIndex.
        market_features: The Phase 3 market-level feature frame (same
            `extras["features"]` `RegimeConditionalStrategy` reads),
            already `:τ`-sliced by the engine.
        n_states, n_restarts, random_state_base, covariance_type,
            min_regime_train_days, features: Forwarded to `regime.fit_hmm`
            unchanged; see its docstring.

    Returns:
        `panel` with one new column, `REGIME_BULL_PROB`. Neutral `0.5` for
        every row if `market_features` is missing/empty or the HMM doesn't
        converge — the same defensive idiom every other Phase 4 addition
        uses, never a crash.
    """
    from regime import REGIME_FEATURES, fit_hmm, predict_regime_posterior_series

    dates = panel.index.get_level_values("Date")
    if market_features is None or market_features.empty:
        result = panel.copy()
        result["REGIME_BULL_PROB"] = 0.5
        return result

    regime_features = features or REGIME_FEATURES
    hmm_fit = fit_hmm(
        market_features,
        n_states=n_states,
        n_restarts=n_restarts,
        random_state_base=random_state_base,
        covariance_type=covariance_type,
        min_regime_train_days=min_regime_train_days,
        features=regime_features,
    )
    posterior = predict_regime_posterior_series(hmm_fit, market_features, regime_features)

    result = panel.copy()
    if posterior.empty or "bull" not in posterior.columns:
        result["REGIME_BULL_PROB"] = 0.5
        return result

    bull_prob_by_date = posterior["bull"].reindex(dates)
    result["REGIME_BULL_PROB"] = bull_prob_by_date.fillna(0.5).to_numpy()
    return result


def build_supervised_dataset(
    panel: pd.DataFrame, log_returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split a feature panel into a trainable (X, y) set and the row to score.

    Addresses: P4 — the single highest-risk correctness point in this
    module. `panel`'s last date τ has no REALIZED next-period return (τ+1
    hasn't happened yet from τ's point of view) — that date is excluded
    from training UNCONDITIONALLY, by construction, regardless of what
    `log_returns` happens to contain beyond it. This does not rely on the
    caller passing an exactly-`:τ`-sliced `log_returns` as an invariant to
    trust (`Strategy.fit()` receives `train_returns` already sliced that
    way, so in practice `log_returns` never extends past τ) — even if it
    did, τ's row would still never appear in the training set, because the
    exclusion is by DATE, not by whether a label happens to be available.
    A NaN-only filter could not offer this guarantee; an explicit boundary
    can.

    Args:
        panel: Output of `melt_to_panel` (optionally passed through
            `attach_regime_feature`) — `(Date, ASSET)` MultiIndex, feature
            columns only.
        log_returns: The SAME returns frame the panel's features were built
            from (`train_returns`, as received by `Strategy.fit()`) — used
            only to look up each `(date, asset)` row's REALIZED return at
            date+1 as the regression target.

    Returns:
        `(X, y, X_predict)`: `X`/`y` are the trainable rows (last date
        excluded; any row with a NaN feature or NaN target additionally
        dropped) — `xgboost` tolerates missing values but `sklearn`'s
        `RandomForestRegressor` does not, so this module drops rather than
        imputes, for both estimators. `X_predict` is the last date's row
        for every asset — the row `fit_predict_expected_returns` actually
        scores. `X_predict` may itself contain NaN (e.g. an asset still in
        its feature warm-up); the caller must check before predicting.
    """
    frames = []
    next_returns = log_returns.shift(-1)
    for asset in log_returns.columns:
        series = next_returns[asset].copy()
        series.index = pd.MultiIndex.from_product(
            [series.index, [asset]], names=["Date", "ASSET"]
        )
        frames.append(series)
    target_long = pd.concat(frames)

    y_full = target_long.reindex(panel.index)

    last_date = panel.index.get_level_values("Date").max()
    is_last_date = panel.index.get_level_values("Date") == last_date

    X_predict = panel.loc[is_last_date]

    trainable = panel.loc[~is_last_date]
    y_trainable = y_full.loc[~is_last_date]

    valid = y_trainable.notna() & trainable.notna().all(axis=1)
    X = trainable.loc[valid]
    y = y_trainable.loc[valid]

    return X, y, X_predict


def fit_predict_expected_returns(
    train_returns: pd.DataFrame,
    extras: Mapping[str, pd.DataFrame] | None = None,
    model_type: str = "random_forest",
    model_params: Mapping | None = None,
    min_train_rows: int = 504,
    short_window: int = 21,
    long_window: int = 63,
    momentum_windows: list[int] = (5, 21, 63),
    condition_on_regime: bool = True,
    n_states: int = 2,
    n_restarts: int = 5,
    random_state_base: int = 0,
    covariance_type: str = "diag",
    min_regime_train_days: int = 252,
) -> pd.Series:
    """Predict each asset's next-period expected return via a pooled panel model.

    Addresses: P1, P2, P3 — see module docstring. Orchestrates
    `build_asset_features` → `melt_to_panel` → `attach_regime_feature` →
    `build_supervised_dataset` → model fit/predict, for whichever
    `model_type` is requested (`"random_forest"` or `"xgboost"`; both
    sklearn-compatible, lazy-imported here the same way `strategies.
    MinVarianceLW` lazy-imports `LedoitWolf`).

    Failure policy, matching every existing estimator in this codebase —
    `strategies._optimize_weights`'s "loud degradation, never crash" rule
    applies here too: below `min_train_rows` pooled rows, if the row to
    score (`X_predict`) contains any NaN feature, or if the underlying
    model's fit/predict raises, log a `WARNING` and return
    `train_returns.mean() * 252` — exactly what `MaxSharpe` already
    computes as its naive `mu`, the nearest already-tested baseline.

    Args:
        train_returns: The exact frame `Strategy.fit()` received — already
            `:τ`-sliced by the engine. Used both to build per-asset features
            and, unchanged, as the label source for `build_supervised_
            dataset` (see that function's docstring for why this matters).
        extras: `Strategy.fit()`'s `extras` mapping; `extras["features"]`
            (the Phase 3 market-level frame) is used for regime-conditioning
            if `condition_on_regime` is true and the key is present.
        model_type: `"random_forest"` or `"xgboost"`.
        model_params: Forwarded to the estimator's constructor.
        min_train_rows, short_window, long_window, momentum_windows,
            condition_on_regime, n_states, n_restarts, random_state_base,
            covariance_type, min_regime_train_days: See `build_asset_
            features`/`attach_regime_feature`.

    Returns:
        Annualized (×252) expected-return `pd.Series`, indexed exactly by
        `train_returns.columns` — the contract `strategies._optimize_
        weights` needs for its `mu` argument.
    """
    fallback = train_returns.mean() * TRADING_DAYS_PER_YEAR

    wide = build_asset_features(
        train_returns,
        short_window=short_window,
        long_window=long_window,
        momentum_windows=momentum_windows,
    )
    panel = melt_to_panel(wide, list(train_returns.columns))

    if condition_on_regime:
        market_features = (extras or {}).get("features")
        panel = attach_regime_feature(
            panel,
            market_features,
            n_states=n_states,
            n_restarts=n_restarts,
            random_state_base=random_state_base,
            covariance_type=covariance_type,
            min_regime_train_days=min_regime_train_days,
        )

    X, y, X_predict = build_supervised_dataset(panel, train_returns)

    if len(X) < min_train_rows:
        log.warning(
            "ml_signals(%s): only %d pooled training rows (< min_train_rows=%d) — "
            "falling back to the naive sample mean for this rebalance.",
            model_type, len(X), min_train_rows,
        )
        return fallback

    X_predict_ordered = X_predict.reindex(columns=X.columns)
    if X_predict_ordered.isna().any().any():
        log.warning(
            "ml_signals(%s): NaN feature(s) in the row to score — "
            "falling back to the naive sample mean for this rebalance.",
            model_type,
        )
        return fallback

    # Both estimators are stochastic (bootstrap sampling / feature subsampling)
    # and default to an UNSEEDED random_state — two fit() calls on identical
    # inputs would otherwise produce different trees and different weights,
    # breaking this codebase's determinism convention (HMM's random_state_base,
    # SLSQP's fixed retry seed) and the "extras accepted and ignored" strategy
    # invariant every prior addition satisfies. random_state=0 is the default
    # here specifically so a caller doesn't have to know to set it themselves.
    resolved_params: dict = {"random_state": 0, **dict(model_params or {})}

    try:
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(**resolved_params)
        elif model_type == "xgboost":
            from xgboost import XGBRegressor

            model = XGBRegressor(**resolved_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        model.fit(X.to_numpy(), y.to_numpy())
        raw_predictions = model.predict(X_predict_ordered.to_numpy())
    except Exception as exc:  # noqa: BLE001 — third-party estimator, deliberately broad
        log.warning(
            "ml_signals(%s): model fit/predict failed (%s) — "
            "falling back to the naive sample mean for this rebalance.",
            model_type, exc,
        )
        return fallback

    predicted = pd.Series(
        raw_predictions, index=X_predict.index.get_level_values("ASSET")
    )
    result = predicted.reindex(train_returns.columns) * TRADING_DAYS_PER_YEAR

    if result.isna().any():
        log.warning(
            "ml_signals(%s): prediction missing for at least one asset — "
            "falling back to the naive sample mean for this rebalance.",
            model_type,
        )
        return fallback

    return result


def run_ml_signal_features(
    config: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate and persist the Phase 4B per-asset feature panel, for audit.

    Addresses: P1, P2, P3 — mirrors `ml_features.run_phase3()`'s Gold-
    persistence pattern for both universes.

    NOT what `RandomForestSignalStrategy`/`XGBoostSignalStrategy` read at
    runtime — those call `build_asset_features` fresh on `train_returns`
    inside `fit()` at every rebalance (already proven causal in isolation,
    and safe by construction since `train_returns` is already `:τ`-sliced
    by the engine before the strategy ever sees it). This Gold output
    exists purely so the actual per-asset feature panel can be inspected —
    e.g. in the Phase 4B hardening notebook — not as a second source of
    truth a strategy depends on.

    Args:
        config: Full project configuration. Defaults to params.yaml.
        project_root: Repository root. Primarily injectable for offline tests.

    Returns:
        Mapping from universe name to the feature DataFrame written to Gold.
    """
    from utils import load_params

    params = dict(config) if config is not None else load_params()
    root = Path(project_root) if project_root is not None else ROOT
    signal_config = params["ml_signals"]
    universes = params["backtest"]["universes"]
    outputs = signal_config["outputs"]

    results: dict[str, pd.DataFrame] = {}
    manifest: dict[str, Any] = {
        "pipeline": "phase4b_ml_signals",
        "note": (
            "Auditability artifact only — RandomForestSignalStrategy/"
            "XGBoostSignalStrategy recompute this panel fresh from "
            "train_returns at every rebalance; they do not read this file."
        ),
        "parameters": {
            key: value for key, value in signal_config.items()
            if key not in ("outputs", "manifest_path")
        },
        "universes": {},
    }

    for universe_name, input_relative in universes.items():
        if universe_name not in outputs:
            raise KeyError(f"No ml_signals output configured for universe {universe_name!r}.")
        input_path = root / input_relative
        if not input_path.exists():
            raise FileNotFoundError(
                f"Gold universe not found: {input_path}. Run Phase 1 and Phase 2 data preparation first."
            )

        returns = pd.read_parquet(input_path)
        returns.index = pd.to_datetime(returns.index)
        features = build_asset_features(
            returns,
            short_window=int(signal_config["short_window"]),
            long_window=int(signal_config["long_window"]),
            momentum_windows=list(signal_config["momentum_windows"]),
        )

        output_path = root / outputs[universe_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path)
        results[universe_name] = features

        manifest["universes"][universe_name] = {
            "input": str(input_relative),
            "output": str(outputs[universe_name]),
            "rows": int(len(features)),
            "columns": int(features.shape[1]),
            "start_date": str(features.index.min().date()),
            "end_date": str(features.index.max().date()),
        }
        log.info("Phase 4B %s features written → %s", universe_name, output_path)

    manifest_path = root / signal_config["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Phase 4B manifest written → %s", manifest_path)
    return results
