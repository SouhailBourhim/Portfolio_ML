"""
run_monitoring_baseline.py — build (and re-check against) the reference window.

Addresses: P2, P4 — pins the reference distributions that any future drift
comparison is measured against, and stores them in a versioned artifact so the
comparison is repeatable by someone who does not hold the training data.

    python src/run_monitoring_baseline.py            # write the baseline
    python src/run_monitoring_baseline.py --evaluate # compare the held-out
                                                     # segment against it

THIS IS NOT LIVE MONITORING. There is no schedule, no alerting path, and no
trigger. `--evaluate` is a human running a command and reading warnings.
Restarting the Dagster schedule is gated on `docs/MODEL_GOVERNANCE.md` §8 and
nothing here does it.

REFERENCE WINDOW. Each universe's Phase 5 train+validation segment — the data
selection was allowed to see. The held-out test segment is deliberately NOT
part of the reference: a baseline that included it would be comparing the test
window against itself and would report stability by construction.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

import monitoring

log = logging.getLogger("run_monitoring_baseline")

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
ARTIFACT = GOLD / "monitoring_baseline.json"

UNIVERSES = {
    "etf_2017": {"returns": "log_returns_etf.parquet", "features": "ml_features_etf.parquet"},
    "full_2021": {"returns": "log_returns.parquet", "features": "ml_features_full.parquet"},
}
PRIMARY = "regime_conditional"


def _load(name: str) -> dict:
    return json.loads((GOLD / name).read_text(encoding="utf-8"))


def _params() -> dict:
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))


def _windows(universe: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Reference = Phase 5 train+validation; evaluation = the frozen test segment."""
    p5 = _load("phase5_results.json")[universe]
    return (
        pd.Timestamp(p5["train_val_start"]), pd.Timestamp(p5["train_val_end"]),
        pd.Timestamp(p5["test_start"]), pd.Timestamp(p5["test_end"]),
    )


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def _slice_long(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame["Date"] >= start) & (frame["Date"] <= end)]


def _universe_slices(universe: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Every input the metrics need, restricted to one window."""
    files = UNIVERSES[universe]
    features = pd.read_parquet(GOLD / files["features"])
    weights = pd.read_parquet(GOLD / "dashboard_weights.parquet")
    regime = pd.read_parquet(GOLD / "dashboard_regime.parquet")

    weights = weights[(weights["universe"] == universe) & (weights["strategy"] == PRIMARY)]
    regime = regime[regime["universe"] == universe]
    return {
        "features": _slice(features, start, end),
        "weights": _slice_long(weights, start, end),
        "regime": _slice_long(regime, start, end),
    }


CHALLENGERS = {"rf_signal_tuned": "random_forest", "xgb_signal_tuned": "xgboost"}


def _challenger_predictions(
    universe: str, params: dict, reference_end: pd.Timestamp
) -> dict[str, pd.Series]:
    """Predictions of ONE fixed model over the whole panel, keyed by challenger.

    Addresses: P2 — the question is whether the *inputs* have moved enough to
    change what the model says, so the model is held FIXED: fit once on rows up
    to `reference_end`, then scored over the entire panel. Refitting per window
    would confound input drift with the refit itself and make a shift
    uninterpretable.

    No lookahead concern arises: nothing here allocates, ranks, or feeds a
    decision. It is a distribution measurement over an already-published model.
    """
    from ml_signals import (
        attach_regime_feature,
        build_asset_features,
        build_supervised_dataset,
        melt_to_panel,
    )

    files = UNIVERSES[universe]
    returns = pd.read_parquet(GOLD / files["returns"])
    market_features = pd.read_parquet(GOLD / files["features"])
    regime = params["regime"]
    selected = _load("phase5_results.json")[universe]

    panel = melt_to_panel(build_asset_features(returns), list(returns.columns))
    panel = attach_regime_feature(
        panel, market_features,
        n_states=regime["n_states"], n_restarts=regime["n_restarts"],
        random_state_base=regime["random_state_base"],
        covariance_type=regime["covariance_type"],
        min_regime_train_days=regime["min_regime_train_days"],
        features=regime["features"],
    )
    X, y, _ = build_supervised_dataset(panel, returns)
    if X.empty:
        return {}

    dates = X.index.get_level_values(0)
    train_mask = dates <= reference_end
    if not train_mask.any():
        return {}

    predictions: dict[str, pd.Series] = {}
    for name, model_type in CHALLENGERS.items():
        model_params = dict(selected["tuned"][name]["selected_ml_params"])
        if model_type == "xgboost":
            from xgboost import XGBRegressor

            # n_jobs=1 is a standing policy here, not a performance choice.
            model = XGBRegressor(random_state=0, n_jobs=1, **model_params)
        else:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(random_state=0, n_jobs=1, **model_params)
        model.fit(X[train_mask].to_numpy(), y[train_mask].to_numpy())
        predictions[name] = pd.Series(model.predict(X.to_numpy()), index=X.index)
    return predictions


def _predictions_in_window(
    predictions: Mapping[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, np.ndarray]:
    out = {}
    for name, series in predictions.items():
        dates = series.index.get_level_values(0)
        out[name] = series[(dates >= start) & (dates <= end)].to_numpy()
    return out


def _metrics(slices: dict, max_weight: float) -> dict:
    """The eight tracked quantities, for one window."""
    features, weights, regime = slices["features"], slices["weights"], slices["regime"]
    latest = (
        weights[weights["Date"] == weights["Date"].max()].set_index("asset")["weight"]
        if not weights.empty else pd.Series(dtype=float)
    )
    return {
        "feature_missingness": monitoring.feature_missingness(features),
        "allocation_concentration": monitoring.allocation_concentration(latest),
        "cap_binding": monitoring.cap_binding_rate(weights, max_weight),
        "turnover": monitoring.turnover_summary(weights),
        "fallback": monitoring.fallback_rate(regime),
        "regime_share": (
            {k: float(v) for k, v in regime["regime"].value_counts(normalize=True).items()}
            if not regime.empty else {}
        ),
    }


def build_baseline() -> dict:
    """Reference distributions and health metrics, per universe."""
    params = _params()
    max_weight = params["backtest"]["max_weight"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "offline_on_demand_reference",
        "operational_note": (
            "Reference distributions for an OFFLINE, ON-DEMAND drift check. No live "
            "schedule is active; drift alerts, retraining triggers and incident "
            "response are not operational. Monitoring emits warnings and never "
            "alters model behaviour."
        ),
        "psi_bands": {
            "moderate": monitoring.PSI_MODERATE,
            "significant": monitoring.PSI_SIGNIFICANT,
            "note": (
                "Conventional credit-risk thresholds, not derived from this "
                "project's data and carrying no statistical guarantee here."
            ),
        },
        "max_weight": float(max_weight),
        "universes": {},
    }

    for universe in UNIVERSES:
        ref_start, ref_end, test_start, test_end = _windows(universe)
        slices = _universe_slices(universe, ref_start, ref_end)
        features = slices["features"]

        payload["universes"][universe] = {
            "reference_window": {"start": str(ref_start.date()), "end": str(ref_end.date())},
            "evaluation_window_available": {
                "start": str(test_start.date()), "end": str(test_end.date()),
                "note": "The frozen Phase 5 test segment — deliberately NOT in the reference.",
            },
            "reference_rows": int(len(features)),
            # Quantile EDGES, not observations: the artifact must be reviewable
            # and must not become a second copy of licence-restricted market data.
            "feature_reference": {
                str(column): monitoring.summarize_reference(features[column].dropna())
                for column in features.columns
            },
            "metrics": _metrics(slices, max_weight),
            "prediction_reference": {
                name: monitoring.summarize_reference(values)
                for name, values in _predictions_in_window(
                    _challenger_predictions(universe, params, ref_end),
                    ref_start, ref_end,
                ).items()
            },
        }
        log.info(
            "%s: reference %s -> %s (%d rows, %d features)",
            universe, ref_start.date(), ref_end.date(), len(features), features.shape[1],
        )
    return payload


def evaluate_against_baseline(baseline: dict | None = None) -> dict:
    """Compare each universe's frozen test segment against the stored reference.

    This is the shape a future deployment would use: same function, a different
    evaluation window. It is run by a human, and its output is a report.
    """
    baseline = baseline or _load(ARTIFACT.name)
    max_weight = baseline["max_weight"]
    report: dict[str, object] = {
        "status": "offline_on_demand_evaluation",
        "operational_note": baseline["operational_note"],
        "universes": {},
    }

    for universe, stored in baseline["universes"].items():
        window = stored["evaluation_window_available"]
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        slices = _universe_slices(universe, start, end)
        features = slices["features"]

        feature_psi = {}
        for column, summary in stored["feature_reference"].items():
            if column not in features.columns:
                continue
            psi = monitoring.psi_from_reference_summary(summary, features[column].dropna())
            feature_psi[column] = {"psi": psi, "interpretation": monitoring.interpret_psi(psi)}

        stored_features = set(stored["feature_reference"])
        current_features = set(map(str, features.columns))
        if stored_features != current_features:
            feature_psi["__schema_mismatch__"] = {
                "missing_from_evaluation": sorted(stored_features - current_features),
                "unexpected_in_evaluation": sorted(current_features - stored_features),
            }

        prediction_psi = {}
        stored_predictions = stored.get("prediction_reference") or {}
        if stored_predictions:
            predictions = _challenger_predictions(
                universe, _params(), pd.Timestamp(stored["reference_window"]["end"])
            )
            for name, values in _predictions_in_window(predictions, start, end).items():
                if name not in stored_predictions:
                    continue
                psi = monitoring.psi_from_reference_summary(stored_predictions[name], values)
                prediction_psi[name] = {
                    "psi": psi, "interpretation": monitoring.interpret_psi(psi)
                }

        regime_shift = monitoring.categorical_shift(
            _regime_labels(universe, stored["reference_window"]),
            slices["regime"]["regime"].tolist(),
        )

        block = {
            "evaluation_window": {"start": str(start.date()), "end": str(end.date())},
            "evaluation_rows": int(len(features)),
            "feature_psi": feature_psi,
            "prediction_psi": prediction_psi,
            "regime_shift": regime_shift,
            "metrics": _metrics(slices, max_weight),
            "reference_metrics": stored["metrics"],
        }
        block["warnings"] = monitoring.build_warnings(block)
        report["universes"][universe] = block
        log.info("%s: %d warning(s)", universe, len(block["warnings"]))
    return report


def _regime_labels(universe: str, window: dict) -> list[str]:
    frame = pd.read_parquet(GOLD / "dashboard_regime.parquet")
    frame = frame[frame["universe"] == universe]
    frame = _slice_long(frame, pd.Timestamp(window["start"]), pd.Timestamp(window["end"]))
    return frame["regime"].tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Compare the frozen test segment against the stored baseline and print warnings.",
    )
    args = parser.parse_args(argv)

    if args.evaluate:
        report = evaluate_against_baseline()
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        total = sum(len(u["warnings"]) for u in report["universes"].values())
        print(f"\n{total} warning(s). Offline check only — no model behaviour changed.")
        return 0

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(
        json.dumps(build_baseline(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(ARTIFACT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
