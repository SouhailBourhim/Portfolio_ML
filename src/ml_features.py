"""
ml_features.py — Phase 3 leakage-free feature engineering.

Builds causal, regime-relevant features for both project universes without
modifying the approved Phase 1 Gold artifacts. Macro signals are differenced
and lagged here but deliberately NOT standardized globally; Phase 4 models
must fit their scaler only on each walk-forward training window.

Addresses: P2 — rolling volatility, drawdown, and macro changes describe
non-stationary market conditions.
Addresses: P3 — rolling correlation and cross-sectional dispersion describe
when diversification is weakening.

Usage:
    python src/ml_features.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils import load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("ml_features")

ROOT = Path(__file__).resolve().parents[1]

ML_CORE_FEATURES = [
    "MARKET_RETURN",
    "MARKET_VOL_SHORT",
    "MARKET_VOL_LONG",
    "AVG_PAIRWISE_CORR",
    "CROSS_SECTIONAL_DISPERSION",
    "MARKET_DRAWDOWN",
]


def _validate_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Validate the returns matrix before causal feature construction."""
    if not isinstance(log_returns, pd.DataFrame) or log_returns.empty:
        raise ValueError("log_returns must be a non-empty DataFrame.")
    if not isinstance(log_returns.index, pd.DatetimeIndex):
        raise TypeError("log_returns must use a DatetimeIndex.")
    if not log_returns.index.is_monotonic_increasing:
        raise ValueError("log_returns index must be sorted ascending.")
    if log_returns.index.has_duplicates:
        raise ValueError("log_returns index contains duplicate dates.")
    if log_returns.shape[1] < 2:
        raise ValueError("At least two assets are required for correlation features.")
    if log_returns.isna().any().any():
        raise ValueError("log_returns contains NaN values; clean the Gold input first.")
    values = log_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("log_returns contains infinite or non-numeric values.")
    validated = log_returns.astype(float).copy()
    validated.index.name = "Date"
    return validated


def average_pairwise_rolling_correlation(
    log_returns: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Calculate the causal rolling mean of all off-diagonal correlations.

    Addresses: P3 — measures whether normally distinct assets begin moving
    together, which is the signature of diversification breakdown in stress.

    Args:
        log_returns: Clean log-return matrix, dates × assets.
        window: Maximum number of observations in each trailing window.
        min_periods: Minimum observations required before emitting a value.

    Returns:
        Series aligned to the input index. Early dates are NaN until enough
        historical observations exist.
    """
    returns = _validate_returns(log_returns)
    if window < 2:
        raise ValueError("window must be >= 2.")
    if min_periods < 2 or min_periods > window:
        raise ValueError("min_periods must satisfy 2 <= min_periods <= window.")

    result = pd.Series(np.nan, index=returns.index, dtype=float, name="AVG_PAIRWISE_CORR")
    upper = np.triu_indices(returns.shape[1], k=1)

    for end_pos in range(min_periods - 1, len(returns)):
        start_pos = max(0, end_pos - window + 1)
        trailing = returns.iloc[start_pos : end_pos + 1]
        if len(trailing) < min_periods:
            continue
        corr_values = trailing.corr().to_numpy()[upper]
        finite_values = corr_values[np.isfinite(corr_values)]
        if finite_values.size:
            result.iloc[end_pos] = float(finite_values.mean())

    return result


def build_return_features(
    log_returns: pd.DataFrame,
    short_window: int = 21,
    long_window: int = 63,
    correlation_window: int = 63,
    correlation_min_periods: int = 42,
) -> pd.DataFrame:
    """Build causal rolling features from a clean return matrix.

    Addresses: P2, P3 — represents volatility clustering, market losses,
    return dispersion, and dynamic correlation using only observations at or
    before each feature date.

    Args:
        log_returns: Clean Gold log-returns, dates × assets.
        short_window: Trailing window for short realized volatility.
        long_window: Trailing window for long realized volatility.
        correlation_window: Trailing window for pairwise correlation.
        correlation_min_periods: Minimum observations for correlation.

    Returns:
        DataFrame aligned to log_returns. Leading rows contain NaN where a
        rolling window does not yet have enough history.
    """
    returns = _validate_returns(log_returns)
    for name, value in {
        "short_window": short_window,
        "long_window": long_window,
        "correlation_window": correlation_window,
    }.items():
        if value < 2:
            raise ValueError(f"{name} must be >= 2.")
    if short_window > long_window:
        raise ValueError("short_window must be <= long_window.")

    market_return = returns.mean(axis=1)
    market_wealth = np.exp(market_return.cumsum())
    running_peak = market_wealth.cummax()

    features = pd.DataFrame(index=returns.index)
    features["MARKET_RETURN"] = market_return
    features["MARKET_VOL_SHORT"] = (
        market_return.rolling(short_window, min_periods=short_window).std()
        * np.sqrt(252.0)
    )
    features["MARKET_VOL_LONG"] = (
        market_return.rolling(long_window, min_periods=long_window).std()
        * np.sqrt(252.0)
    )
    features["AVG_PAIRWISE_CORR"] = average_pairwise_rolling_correlation(
        returns,
        window=correlation_window,
        min_periods=correlation_min_periods,
    )
    features["CROSS_SECTIONAL_DISPERSION"] = returns.std(axis=1, ddof=1)
    features["MARKET_DRAWDOWN"] = market_wealth / running_peak - 1.0
    features.index.name = "Date"
    return features


def build_lagged_macro_signals(
    raw_macro: pd.DataFrame,
    returns_index: pd.DatetimeIndex,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Build differenced and lagged macro signals without global scaling.

    Addresses: P2, P3 — macro changes can help characterize regime shifts,
    while the mandatory lag and the absence of full-sample standardization
    prevent lookahead leakage.

    Args:
        raw_macro: Bronze macro levels, dates × series.
        returns_index: Trading dates to which macro releases are aligned.
        lag_days: Trading-day lag. Must be at least one.

    Returns:
        Macro changes aligned to returns_index. A Phase 4 strategy must fit
        its scaler only on the training slice supplied by the backtest engine.
    """
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1 to prevent lookahead bias.")
    if not isinstance(raw_macro, pd.DataFrame) or raw_macro.empty:
        return pd.DataFrame(index=returns_index)
    if not isinstance(returns_index, pd.DatetimeIndex):
        raise TypeError("returns_index must be a DatetimeIndex.")

    macro = raw_macro.copy()
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()
    if macro.index.has_duplicates:
        raise ValueError("raw_macro index contains duplicate dates.")
    if macro.columns.duplicated().any():
        duplicates = macro.columns[macro.columns.duplicated()].tolist()
        raise ValueError(f"raw_macro contains duplicate columns: {duplicates}")

    aligned = macro.reindex(returns_index, method="ffill")
    lagged = aligned.diff().shift(lag_days)
    lagged = lagged.replace([np.inf, -np.inf], np.nan)

    all_nan_columns = lagged.columns[lagged.isna().all()].tolist()
    if all_nan_columns:
        log.warning("Dropping all-NaN macro signals: %s", all_nan_columns)
        lagged = lagged.drop(columns=all_nan_columns)

    lagged.columns = [f"{column}_DIFF_L{lag_days}" for column in lagged.columns]
    lagged.index.name = "Date"
    return lagged.astype(float)


def build_ml_feature_set(
    log_returns: pd.DataFrame,
    raw_macro: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Combine return and macro signals into one Phase 3 feature set.

    Addresses: P2, P3 — creates the causal observation matrix consumed by
    regime models while preserving optional macro columns with partial history.

    Args:
        log_returns: Gold return matrix for one universe.
        raw_macro: Combined Bronze macro levels, or None if unavailable.
        config: The `ml_features` section from params.yaml.

    Returns:
        Feature matrix beginning at the first date where every required return
        feature is available. Optional macro NaN values are retained explicitly.
    """
    return_features = build_return_features(
        log_returns,
        short_window=int(config["volatility_short_window"]),
        long_window=int(config["volatility_long_window"]),
        correlation_window=int(config["correlation_window"]),
        correlation_min_periods=int(config["correlation_min_periods"]),
    )

    if raw_macro is None or raw_macro.empty:
        log.warning("No macro data available; producing return-only Phase 3 features.")
        combined = return_features
    else:
        macro_signals = build_lagged_macro_signals(
            raw_macro,
            return_features.index,
            lag_days=int(config["macro_lag_days"]),
        )
        combined = return_features.join(macro_signals, how="left")

    combined = combined.replace([np.inf, -np.inf], np.nan)
    required_ready = combined[ML_CORE_FEATURES].notna().all(axis=1)
    dropped_rows = int((~required_ready).sum())
    combined = combined.loc[required_ready].copy()

    if combined.empty:
        raise ValueError("No rows remain after the required rolling feature warm-up.")
    if combined[ML_CORE_FEATURES].isna().any().any():
        raise ValueError("Required Phase 3 features contain NaN after warm-up filtering.")
    if not combined.index.is_monotonic_increasing or combined.index.has_duplicates:
        raise ValueError("Phase 3 feature index must be sorted and unique.")

    combined.index.name = "Date"
    log.info(
        "Phase 3 feature set built: %d rows × %d columns; %d leading rows removed.",
        combined.shape[0],
        combined.shape[1],
        dropped_rows,
    )
    return combined.astype(float)


def run_phase3(
    config: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate and persist Phase 3 features for both project universes.

    Addresses: P2, P3 — applies the same audited feature definitions to the
    ETF 2017+ and full 2021+ universes, keeping their crisis evidence and
    EURAFRIC relevance separate and traceable.

    Args:
        config: Full project configuration. Defaults to params.yaml.
        project_root: Repository root. Primarily injectable for offline tests.

    Returns:
        Mapping from universe name to the feature DataFrame written to Gold.
    """
    params = dict(config) if config is not None else load_params()
    root = Path(project_root) if project_root is not None else ROOT
    ml_config = params["ml_features"]
    universes = params["backtest"]["universes"]
    outputs = ml_config["outputs"]

    macro_frames: list[pd.DataFrame] = []
    for relative_path in (
        "data/bronze/raw_macro.parquet",
        "data/bronze/raw_bam_macro.parquet",
    ):
        path = root / relative_path
        if path.exists():
            frame = pd.read_parquet(path)
            frame.index = pd.to_datetime(frame.index)
            macro_frames.append(frame)
        else:
            log.warning("Optional macro input not found: %s", path)

    raw_macro: pd.DataFrame | None = None
    if macro_frames:
        raw_macro = pd.concat(macro_frames, axis=1).sort_index()
        if raw_macro.columns.duplicated().any():
            duplicates = raw_macro.columns[raw_macro.columns.duplicated()].tolist()
            raise ValueError(f"Combined macro inputs contain duplicate columns: {duplicates}")

    results: dict[str, pd.DataFrame] = {}
    manifest: dict[str, Any] = {
        "pipeline": "phase3_ml_features",
        "global_standardization": False,
        "standardization_policy": (
            "Fit scaling only inside each Phase 4 strategy.fit training window."
        ),
        "parameters": {
            key: value for key, value in ml_config.items() if key != "outputs"
        },
        "core_features": {
            "MARKET_RETURN": "Equal-weight cross-asset mean log-return.",
            "MARKET_VOL_SHORT": "Annualized trailing volatility of MARKET_RETURN.",
            "MARKET_VOL_LONG": "Annualized longer trailing volatility of MARKET_RETURN.",
            "AVG_PAIRWISE_CORR": "Mean off-diagonal trailing asset correlation.",
            "CROSS_SECTIONAL_DISPERSION": "Same-day standard deviation across asset returns.",
            "MARKET_DRAWDOWN": "Drawdown of cumulative equal-weight market wealth.",
        },
        "universes": {},
    }

    for universe_name, input_relative in universes.items():
        if universe_name not in outputs:
            raise KeyError(f"No ml_features output configured for universe {universe_name!r}.")
        input_path = root / input_relative
        if not input_path.exists():
            raise FileNotFoundError(
                f"Gold universe not found: {input_path}. Run Phase 1 and Phase 2 data preparation first."
            )

        returns = pd.read_parquet(input_path)
        returns.index = pd.to_datetime(returns.index)
        features = build_ml_feature_set(returns, raw_macro, ml_config)

        output_path = root / outputs[universe_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path)
        results[universe_name] = features

        manifest["universes"][universe_name] = {
            "input": str(input_relative),
            "output": str(outputs[universe_name]),
            "input_rows": int(len(returns)),
            "output_rows": int(len(features)),
            "dropped_leading_rows": int(len(returns) - len(features)),
            "start_date": str(features.index.min().date()),
            "end_date": str(features.index.max().date()),
            "columns": list(features.columns),
            "fully_complete_rows": int(features.notna().all(axis=1).sum()),
            "missing_values_by_column": {
                column: int(count) for column, count in features.isna().sum().items()
            },
        }
        log.info("Phase 3 %s written → %s", universe_name, output_path)

    manifest_path = root / ml_config["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Phase 3 manifest written → %s", manifest_path)
    return results


if __name__ == "__main__":
    run_phase3()
