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

import telemetry
from memo import ContentCache, content_key

log = logging.getLogger("ml_signals")

ROOT = Path(__file__).resolve().parents[1]
TRADING_DAYS_PER_YEAR = 252

VALID_MU_TRANSFORMS = ("none", "shrink", "rank")

# Shared across every signal strategy in a run: five RandomForest variants
# differ only in post-prediction settings, so without this they fit the same
# forest five times per rebalance date. Content-addressed — see `memo`.
_PREDICTION_CACHE = ContentCache("ml_signal_predict")


def _validate_mu_transform(mu_transform: str, shrinkage_weight: float) -> None:
    """
    Single source of truth for mu-regularization config validity.

    Called both at the top of `fit_predict_expected_returns` (fail-fast,
    before an expensive model fit) and inside `apply_mu_transform` (which is
    also a public entry point) — one function, so the two call sites cannot
    drift if a new transform mode is added.

    A misspelled config value is a caller bug that must surface immediately,
    NOT degrade to the naive mean the way a runtime estimator failure does —
    the two are different failure classes and must not be conflated.
    """
    if mu_transform not in VALID_MU_TRANSFORMS:
        raise ValueError(
            f"Unknown mu_transform: {mu_transform!r} — expected one of {VALID_MU_TRANSFORMS}."
        )
    if mu_transform == "shrink" and not 0.0 <= shrinkage_weight <= 1.0:
        # Outside [0, 1] this stops being a convex blend and becomes
        # extrapolation (e.g. 1.5·predicted − 0.5·naive), which contradicts
        # the documented "damp toward the naive estimate" behavior and would
        # AMPLIFY the very estimation noise the shrink mode exists to temper.
        raise ValueError(
            f"shrinkage_weight must be within [0, 1] for a convex blend, "
            f"got {shrinkage_weight!r}."
        )


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
            ).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

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


def attach_fundamentals_features(
    panel: pd.DataFrame,
    fundamentals_panel: pd.DataFrame | None,
    fund_assets: list[str] | None = None,
) -> pd.DataFrame:
    """Merge per-asset fundamentals (F7 fundamentals experiment) onto the panel.

    Addresses: P4 (fundamentals experiment) — the deep-Morocco experiment
    showed that price-only ML tops out at the "prediction accuracy ≠
    portfolio performance" ceiling. This function is the seam through which
    a genuinely new data class (fundamentals, from `src/fundamentals.py`)
    reaches the F7 model.

    The fundamentals Gold panel is `(Date × TICKER__FUND_metric)` and covers
    only a subset of the universe's tickers (BVC equities have published
    fundamentals; ETFs do not — an ETF's "fundamentals" would be a holdings-
    weighted aggregate, a modelling problem out of scope). Two facts about
    that panel drive this function's design:

    1. For each `(date, asset)` row on the F7 panel, if `asset` has its own
       fundamentals column, we read it directly. This is genuine, causal,
       point-in-time information (§15.14 / §15.8-style: `fundamentals.py`'s
       `apply_publication_lag` already enforced the causal boundary
       upstream).
    2. For assets WITHOUT fundamentals (ETFs), we fill each fundamental with
       the CROSS-SECTIONAL MEDIAN of the tickers that DO have that
       fundamental on that date. A binary `HAS_FUND` column tells the tree
       model which rows carry a real signal. This preserves the training row
       — critical for a pooled cross-sectional model where dropping the ETFs
       would lose 5/9 of `full_2021`'s assets — while letting the tree
       discover the asset-type distinction itself, rather than us hard-
       coding it. The `HAS_FUND=0` rows do not contaminate the fit because
       the median-fill values are, by construction, the same for every ETF
       on a given date (a constant across the ETF subset per date), so any
       split on a `FUND_*` column stratifies mainly by `HAS_FUND`.

    Args:
        panel: Output of `melt_to_panel` (typically after
            `attach_regime_feature`) — `(Date, ASSET)` MultiIndex.
        fundamentals_panel: Gold-layer fundamentals panel (columns like
            `IAM__FUND_pe`); pass `None` to no-op, useful for the ablation
            control (F7 without fundamentals vs. F7 with fundamentals).
        fund_assets: Which assets in `panel` have fundamentals (typically
            the BVC subset). Inferred from column names if omitted.

    Returns:
        `panel` with one `FUND_{metric}` column per metric and a
        `HAS_FUND` indicator (0/1) — same idiom `attach_regime_feature`
        uses to add `REGIME_BULL_PROB`.
    """
    if fundamentals_panel is None or fundamentals_panel.empty:
        return panel

    # Infer per-asset fund columns: those matching TICKER__FUND_metric
    fund_cols = list(fundamentals_panel.columns)
    parsed = [c.split("__FUND_") for c in fund_cols]
    parsed = [p for p in parsed if len(p) == 2]
    if not parsed:
        return panel

    all_fund_assets = sorted({t for t, _ in parsed})
    metrics = sorted({m for _, m in parsed})
    if fund_assets is None:
        fund_assets = all_fund_assets

    # For each (date, asset), look up asset's fund columns if it has them,
    # else fill with the cross-sectional median of the fund_assets at that date.
    dates = panel.index.get_level_values("Date").unique()
    fund_view = fundamentals_panel.reindex(dates)  # forward-fill already applied upstream

    result = panel.copy()
    for m in metrics:
        cols = [f"{t}__FUND_{m}" for t in fund_assets if f"{t}__FUND_{m}" in fund_view.columns]
        if not cols:
            continue
        median_by_date = fund_view[cols].median(axis=1, skipna=True)

        panel_dates = result.index.get_level_values("Date")
        panel_assets = result.index.get_level_values("ASSET")

        # First, initialise everyone with the date's median (the ETF fill).
        vals = median_by_date.reindex(panel_dates).to_numpy()

        # Then overwrite with each asset's own value where present.
        for t in fund_assets:
            col_name = f"{t}__FUND_{m}"
            if col_name not in fund_view.columns:
                continue
            asset_series = fund_view[col_name].reindex(panel_dates).to_numpy()
            mask = (panel_assets == t)
            vals = np.where(mask, asset_series, vals)

        result[f"FUND_{m}"] = vals

    # HAS_FUND: 1 for assets with their own fundamentals column, 0 otherwise.
    panel_assets = result.index.get_level_values("ASSET")
    result["HAS_FUND"] = panel_assets.isin(fund_assets).astype(int)
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


def apply_mu_transform(
    predicted: pd.Series,
    naive: pd.Series,
    mu_transform: str = "none",
    shrinkage_weight: float = 0.5,
) -> pd.Series:
    """Regularize a predicted expected-return vector before it reaches the optimizer.

    Addresses: P1 — Chopra & Ziemba (1993) showed estimation error in
    EXPECTED RETURNS is roughly an order of magnitude more damaging to a
    mean-variance optimizer than equivalent error in the covariance. That
    result explains the entire Phase 4/4B pattern on this project's own
    data: Phase 4 improved the covariance input and beat its hurdle on
    `full_2021` (+15%), while Phase 4B improved the `mu` input and had its
    best-in-class GROSS Sharpe (1.240) destroyed by the weight swings a
    noisy `mu` provokes (0.885 turnover). These transforms attack that
    directly, by refusing to take the model's point estimates at face value.

    Modes:
      - `"none"`: pass the prediction through unchanged (the Phase 4B
        behavior, kept as the honest floor to compare against).
      - `"shrink"`: convex blend `w·predicted + (1−w)·naive`. The standard
        remedy — keep the signal, damp its magnitude toward the estimate
        that at least has no model risk. `shrinkage_weight=0` recovers the
        naive sample mean exactly, `=1` recovers the raw prediction.
      - `"rank"`: keep only the model's cross-sectional ORDERING and borrow
        the level and dispersion from `naive`. The strongest form of "trust
        the ranking, not the magnitudes" — standard in cross-sectional
        equity ML precisely because predicted return magnitudes are far
        less reliable than the ordering they imply.

    Args:
        predicted: Model output, indexed by asset (annualized).
        naive: The sample-mean estimate `MaxSharpe` already uses, same index
            and units — the shrinkage target and the dispersion donor.
        mu_transform: One of `"none"`, `"shrink"`, `"rank"`.
        shrinkage_weight: Weight on `predicted` when `mu_transform="shrink"`;
            must be within [0, 1] so the blend stays convex.

    Returns:
        Transformed expected returns, same index as `predicted`.

    Raises:
        ValueError: on an unknown `mu_transform`, or a `shrinkage_weight`
            outside [0, 1] under the shrink mode — unlike a runtime estimator
            failure (which degrades to the naive mean by design), a
            misspelled/out-of-range config value is a caller bug that should
            surface immediately rather than silently misbehaving. Validated
            via the shared `_validate_mu_transform`.
    """
    _validate_mu_transform(mu_transform, shrinkage_weight)

    if mu_transform == "none":
        return predicted
    if mu_transform == "shrink":
        return shrinkage_weight * predicted + (1.0 - shrinkage_weight) * naive
    # mu_transform == "rank" — the only remaining valid mode after validation.
    ranks = predicted.rank(method="average")
    spread = float(ranks.std(ddof=0))
    if spread < 1e-12:  # single asset, or a perfectly flat prediction
        return naive
    centered = (ranks - ranks.mean()) / spread
    return float(naive.mean()) + centered * float(naive.std(ddof=0))


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
    mu_transform: str = "none",
    shrinkage_weight: float = 0.5,
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
        mu_transform, shrinkage_weight: Regularization applied to the
            prediction before it is returned — see `apply_mu_transform`.
            Every fallback path below returns the naive mean directly and is
            therefore unaffected by these (a transform of the naive estimate
            toward itself is a no-op for `"shrink"`, and `"rank"` on an
            estimate we already decided not to trust would be noise).

    Returns:
        Annualized (×252) expected-return `pd.Series`, indexed exactly by
        `train_returns.columns` — the contract `strategies._optimize_
        weights` needs for its `mu` argument.
    """
    fallback = train_returns.mean() * TRADING_DAYS_PER_YEAR

    # Fail fast on a bad transform config BEFORE the expensive model fit,
    # using the same validator apply_mu_transform enforces at the end.
    _validate_mu_transform(mu_transform, shrinkage_weight)

    # The model fit does not depend on `mu_transform`/`shrinkage_weight` —
    # those are applied to the prediction AFTERWARDS — so the expensive part
    # is cached on everything EXCEPT them. That is what lets `rf_signal`,
    # `rf_signal_cost`, `rf_signal_shrunk`, `rf_signal_rank` and
    # `rf_signal_cost_dcc` share one fit per rebalance date instead of five.
    # See `memo` for why a content-addressed cache cannot serve a stale value.
    #
    # The key deliberately names the two `extras` frames this function reads
    # rather than digesting `extras` wholesale. That is not a shortcut: the
    # turnover-penalized variants also receive `backtest.CURRENT_WEIGHTS_KEY`
    # in `extras`, which is consumed by `strategies._extract_current_weights`
    # AFTER this call and never read here. Keying on the whole mapping would
    # therefore separate `rf_signal_cost` from `rf_signal` on an input that
    # provably cannot change this function's output — correct, but it would
    # forfeit most of the saving. Anything read below must appear here.
    key = content_key(
        "ml_signal_predict", train_returns,
        (extras or {}).get("features"), (extras or {}).get("fundamentals"),
        model_type, model_params, min_train_rows, short_window, long_window,
        tuple(momentum_windows), condition_on_regime, n_states, n_restarts,
        random_state_base, covariance_type, min_regime_train_days,
    )
    # The cache stores the FitRecord alongside the prediction, and it is
    # re-emitted on every hit. Without that, a fallback would be recorded once
    # and then vanish behind the cache, so the measured fallback RATE would
    # fall as the cache warmed — the one failure a fallback-rate metric must
    # not have. `tests/test_telemetry.py` pins warm == cold.
    predicted, used_fallback, fit_record = _PREDICTION_CACHE.get_or_compute(
        key,
        lambda: _predict_expected_returns_uncached(
            train_returns, extras, model_type, model_params, min_train_rows,
            short_window, long_window, momentum_windows, condition_on_regime,
            n_states, n_restarts, random_state_base, covariance_type,
            min_regime_train_days, fallback,
        ),
    )
    # Copy so a caller that mutates the returned Series in place — or takes a
    # zero-copy `.to_numpy()` view of it, as `_MLSignalStrategy.fit` does —
    # cannot corrupt the cached entry for every later strategy.
    predicted = predicted.copy()
    telemetry.record(fit_record)

    # Every fallback path returns the naive mean unchanged, exactly as before:
    # a transform of the naive estimate toward itself is a no-op for
    # "shrink", and ranking an estimate we already decided not to trust would
    # be noise. Preserving that branch is why the cache stores the flag too.
    if used_fallback:
        return predicted
    return apply_mu_transform(predicted, fallback, mu_transform, shrinkage_weight)


def _fallback_record(model_type: str, n_rows: int, reason: str) -> telemetry.FitRecord:
    """A record for a rebalance where the naive sample mean produced the number.

    Addresses: P4 — every early return in the predictor below is a rebalance
    on which an "ML signal" result is arithmetically a sample-mean result. The
    reason travels with it so a reader can tell a thin window apart from a
    crashed estimator; they call for different responses.
    """
    return telemetry.FitRecord(
        model_requested=model_type,
        model_effective="naive_sample_mean",
        fit_status=telemetry.STATUS_FALLBACK,
        n_training_rows=int(n_rows),
        fallback_reason=reason,
    )


def _predict_expected_returns_uncached(
    train_returns: pd.DataFrame,
    extras: Mapping[str, pd.DataFrame] | None,
    model_type: str,
    model_params: Mapping | None,
    min_train_rows: int,
    short_window: int,
    long_window: int,
    momentum_windows: list[int],
    condition_on_regime: bool,
    n_states: int,
    n_restarts: int,
    random_state_base: int,
    covariance_type: str,
    min_regime_train_days: int,
    fallback: pd.Series,
) -> tuple[pd.Series, bool, telemetry.FitRecord]:
    """The panel fit/predict itself, with no `mu` transform applied.

    Addresses: P1, P2, P3 — this is the body `fit_predict_expected_returns`
    used to inline; it was split out unchanged so the expensive half can be
    memoized on its own inputs (see `memo`). Returns `(prediction,
    used_fallback)`; the caller applies the transform only when the model
    genuinely produced an estimate, preserving the previous behaviour that
    fallbacks are never transformed.
    """
    if model_type not in ("random_forest", "xgboost"):
        log.warning(
            "ml_signals(%s): unknown model_type (expected 'random_forest' or "
            "'xgboost') — falling back to the naive sample mean for this rebalance.",
            model_type,
        )
        return fallback, True, _fallback_record(
            model_type, len(train_returns),
            f"unknown model_type {model_type!r} (expected 'random_forest' or 'xgboost')",
        )

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

    # Fundamentals experiment (opt-in via extras["fundamentals"]): if the
    # caller supplied a point-in-time fundamentals panel (already `:τ`-sliced
    # by the engine like any other extras frame), attach it as additional
    # per-asset feature columns. Absent key = classical F7, unchanged.
    fundamentals_panel = (extras or {}).get("fundamentals")
    if fundamentals_panel is not None and not fundamentals_panel.empty:
        panel = attach_fundamentals_features(panel, fundamentals_panel)

    X, y, X_predict = build_supervised_dataset(panel, train_returns)

    if len(X) < min_train_rows:
        log.warning(
            "ml_signals(%s): only %d pooled training rows (< min_train_rows=%d) — "
            "falling back to the naive sample mean for this rebalance.",
            model_type, len(X), min_train_rows,
        )
        return fallback, True, _fallback_record(
            model_type, len(X),
            f"only {len(X)} pooled training rows (< min_train_rows={min_train_rows})",
        )

    X_predict_ordered = X_predict.reindex(columns=X.columns)
    if X_predict_ordered.isna().any().any():
        log.warning(
            "ml_signals(%s): NaN feature(s) in the row to score — "
            "falling back to the naive sample mean for this rebalance.",
            model_type,
        )
        return fallback, True, _fallback_record(
            model_type, len(X), "NaN feature(s) in the row to score",
        )

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
        else:
            from xgboost import XGBRegressor

            # Match the CV path in model_selection: do not let XGBoost create
            # an uncontrolled native worker pool during repeated walk-forward
            # fits. An explicit model_params["n_jobs"] remains authoritative.
            resolved_params.setdefault("n_jobs", 1)
            model = XGBRegressor(**resolved_params)

        model.fit(X.to_numpy(), y.to_numpy())
        raw_predictions = model.predict(X_predict_ordered.to_numpy())
    except Exception as exc:  # noqa: BLE001 — third-party estimator, deliberately broad
        log.warning(
            "ml_signals(%s): model fit/predict failed (%s) — "
            "falling back to the naive sample mean for this rebalance.",
            model_type, exc,
        )
        return fallback, True, _fallback_record(
            model_type, len(X), f"{type(exc).__name__}: {exc}",
        )

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
        return fallback, True, _fallback_record(
            model_type, len(X), "prediction missing for at least one asset",
        )

    return result, False, telemetry.FitRecord(
        model_requested=model_type,
        model_effective=model_type,
        fit_status=telemetry.STATUS_OK,
        n_training_rows=len(X),
    )


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
