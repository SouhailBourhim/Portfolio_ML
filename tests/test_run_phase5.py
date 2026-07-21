"""
test_run_phase5.py — Smoke test for the Phase 5 OOS-evaluation runner.

Tiny synthetic Gold snapshot under tmp_path (offline, never the real mlruns/
or data/), matching tests/test_run_phase4c.py's pattern. Proves the pipeline
wires together and — critically — that the frozen test segment is disjoint
from the train+validation window the selectors saw.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_phase4
import run_phase5
from ml_features import build_ml_feature_set


def _write_gold_snapshot(tmp_path, n: int = 320, assets=("SPY", "QQQ", "GLD"), seed: int = 1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n, name="Date")
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(n, len(assets))), index=dates, columns=list(assets)
    )
    macro = pd.DataFrame({"VIX": 18 + np.cumsum(rng.normal(0, 0.3, n))}, index=dates)
    features = build_ml_feature_set(
        returns, macro,
        {"volatility_short_window": 10, "volatility_long_window": 20,
         "correlation_window": 20, "correlation_min_periods": 10, "macro_lag_days": 1},
    )
    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True)
    returns.to_parquet(gold / "log_returns_etf.parquet")
    returns.to_parquet(gold / "log_returns.parquet")
    features.to_parquet(gold / "ml_features_etf.parquet")
    features.to_parquet(gold / "ml_features_full.parquet")


def _params() -> dict:
    return {
        "backtest": {
            "rebalance_freq": "ME", "min_train_days": 120, "max_weight": 1.0,
            "risk_free_annual": 0.0, "costs_bps": {"etf": 10, "bvc": 30},
            "universes": {
                "etf_2017": "data/gold/log_returns_etf.parquet",
                "full_2021": "data/gold/log_returns.parquet",
            },
        },
        "ml_features": {"outputs": {
            "etf_2017": "data/gold/ml_features_etf.parquet",
            "full_2021": "data/gold/ml_features_full.parquet",
        }},
        "ml_signals": {
            "short_window": 10, "long_window": 20, "momentum_windows": [5, 10],
            "min_train_rows": 30, "condition_on_regime": False,
            "random_forest": {"n_estimators": 12, "max_depth": 3, "random_state": 0},
            "xgboost": {"n_estimators": 12, "max_depth": 3, "random_state": 0},
        },
        "covariance_ewma": {"halflife_days": 20},
        "covariance_dcc_garch": {
            "garch_p": 1, "garch_q": 1, "dcc_a_init": 0.02, "dcc_b_init": 0.95, "rescale_factor": 100,
        },
        "regime": {
            "n_states": 2, "covariance_type": "diag", "n_restarts": 2, "random_state_base": 0,
            "min_regime_train_days": 30, "features": ["MARKET_RETURN"],
            "bull_strategy": "max_sharpe", "bear_strategy": "min_variance_lw",
        },
        "purged_cv": {"n_splits": 3, "embargo_frac": 0.02},
        "phase5": {
            "results_path": "data/gold/phase5_results.json",
            "ledger_path": "data/gold/dsr_trial_ledger.json",
            "test_frac": 0.35,
            "rf_grid": {"max_depth": [2, 3], "n_estimators": [12]},
            "xgb_grid": {"max_depth": [2, 3], "n_estimators": [12]},
            "shrink_grid": [0.5, 1.0],
            "penalty_grid": [0.0, 1.0],
            "bootstrap": {"block_len": 10, "n_boot": 100, "alpha": 0.10, "seed": 0},
        },
    }


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(run_phase4, "ROOT", tmp_path)
    monkeypatch.setattr(run_phase5, "ROOT", tmp_path)
    monkeypatch.setattr(run_phase5, "load_params", lambda: _params())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow_test.db'}")


def test_run_phase5_writes_results_and_respects_the_test_split(tmp_path, monkeypatch):
    _write_gold_snapshot(tmp_path)
    _patch(monkeypatch, tmp_path)

    output = run_phase5.run_phase5()

    assert set(output) == {"etf_2017", "full_2021"}
    for uni, entry in output.items():
        # Both F7 families tuned, both baselines re-evaluated on the test window.
        assert set(entry["tuned"]) == {"rf_signal_tuned", "xgb_signal_tuned"}
        assert set(entry["baselines"]) == {"regime_conditional", "equal_weight"}
        # THE guarantee: the frozen test window starts strictly after the
        # train+validation window the selectors saw.
        assert entry["test_start"] > entry["train_val_end"]
        # Selected hyperparameters come from the declared grids.
        for tuned in entry["tuned"].values():
            assert tuned["selected_ml_params"]["max_depth"] in (2, 3)
            assert tuned["selected_levers"]["shrinkage_weight"] in (0.5, 1.0)
            assert tuned["selected_levers"]["turnover_penalty"] in (0.0, 1.0)
            # Every headline Sharpe carries a CI.
            assert len(tuned["test_sharpe_ci"]) == 2
        assert isinstance(entry["beats_hurdle_on_test"], bool)
        # DSR pool accumulated the whole search (validation grid + finals).
        assert entry["n_search_trials"] >= 4

    results_file = tmp_path / "data" / "gold" / "phase5_results.json"
    ledger_file = tmp_path / "data" / "gold" / "dsr_trial_ledger.json"
    assert results_file.exists() and ledger_file.exists()
    assert set(json.loads(results_file.read_text())) == {"etf_2017", "full_2021"}
