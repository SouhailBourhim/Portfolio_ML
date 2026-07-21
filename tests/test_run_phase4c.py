"""
test_run_phase4c.py — Smoke test for the Phase 4C comparison runner.

Uses a tiny synthetic Gold snapshot under tmp_path (never a real download,
never the real mlruns/ directory) so this stays offline and fast, matching
tests/test_run_phase4b.py's pattern for the prior runner.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_phase4
import run_phase4b
import run_phase4c
from ml_features import build_ml_feature_set


def _write_gold_snapshot(tmp_path, n: int = 200, assets=("SPY", "QQQ", "GLD"), seed: int = 1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n, name="Date")
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(n, len(assets))), index=dates, columns=list(assets)
    )
    macro = pd.DataFrame({"VIX": 18 + np.cumsum(rng.normal(0, 0.3, n))}, index=dates)
    features = build_ml_feature_set(
        returns, macro,
        {
            "volatility_short_window": 10,
            "volatility_long_window": 20,
            "correlation_window": 20,
            "correlation_min_periods": 10,
            "macro_lag_days": 1,
        },
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
            "rebalance_freq": "ME",
            "min_train_days": 100,
            "max_weight": 1.0,
            "risk_free_annual": 0.0,
            "costs_bps": {"etf": 10, "bvc": 30},
            "universes": {
                "etf_2017": "data/gold/log_returns_etf.parquet",
                "full_2021": "data/gold/log_returns.parquet",
            },
        },
        "ml_features": {
            "outputs": {
                "etf_2017": "data/gold/ml_features_etf.parquet",
                "full_2021": "data/gold/ml_features_full.parquet",
            }
        },
        "covariance_ewma": {"halflife_days": 20},
        "covariance_dcc_garch": {
            "garch_p": 1, "garch_q": 1,
            "dcc_a_init": 0.02, "dcc_b_init": 0.95, "rescale_factor": 100,
        },
        "regime": {
            "n_states": 2, "covariance_type": "diag", "n_restarts": 2,
            "random_state_base": 0, "min_regime_train_days": 30,
            "bull_strategy": "max_sharpe", "bear_strategy": "min_variance_lw",
        },
        "phase4": {"results_path": "data/gold/phase4_results.json"},
        "ml_signals": {
            "short_window": 10, "long_window": 20, "momentum_windows": [5, 10],
            "min_train_rows": 30, "condition_on_regime": False,
            "random_forest": {"n_estimators": 12, "max_depth": 3, "random_state": 0},
            "xgboost": {"n_estimators": 12, "max_depth": 3, "random_state": 0},
        },
        "phase4b": {"results_path": "data/gold/phase4b_results.json"},
        "phase4c": {
            "results_path": "data/gold/phase4c_results.json",
            "turnover_penalty": 1.0,
            "shrinkage_weight": 0.5,
        },
    }


def _patch(monkeypatch, tmp_path):
    # run_phase4c reuses load_universe/load_features from run_phase4 and
    # build_strategies from run_phase4b, each of which resolves paths via its
    # OWN module-level ROOT — all three must be redirected.
    monkeypatch.setattr(run_phase4, "ROOT", tmp_path)
    monkeypatch.setattr(run_phase4b, "ROOT", tmp_path)
    monkeypatch.setattr(run_phase4c, "ROOT", tmp_path)
    monkeypatch.setattr(run_phase4c, "load_params", lambda: _params())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow_test.db'}")


def test_run_phase4c_writes_results_with_full_trial_pool(tmp_path, monkeypatch):
    _write_gold_snapshot(tmp_path)
    _patch(monkeypatch, tmp_path)

    all_results = run_phase4c.run_phase4c()

    expected_names = {
        "equal_weight", "min_variance", "min_variance_lw", "max_sharpe",
        "min_variance_ewma", "dcc_garch", "regime_conditional",
        "rf_signal", "xgb_signal",
        "rf_signal_cost", "rf_signal_shrunk", "rf_signal_rank",
        "rf_signal_cost_dcc", "xgb_signal_cost",
    }
    assert set(all_results) == {"etf_2017", "full_2021"}
    for universe_results in all_results.values():
        # 9 carried forward from Phase 4B + 5 new Phase 4C variants.
        assert len(universe_results) == 14
        assert {r.strategy_name for r in universe_results} == expected_names

    output = json.loads((tmp_path / "data" / "gold" / "phase4c_results.json").read_text())
    for entry in output.values():
        assert entry["n_trials"] == 14
        # The diagnosis (gross, net, turnover per strategy) must live in the
        # artifact, not only in a notebook someone has to rerun.
        assert set(entry["per_strategy"]) == expected_names
        for row in entry["per_strategy"].values():
            assert {"sharpe_gross", "sharpe_net", "avg_turnover"} <= set(row)
        # No stored phase4_results.json under tmp_path -> comparison is None.
        assert entry["beats_phase4_hurdle"] is None


def test_turnover_penalty_lowers_turnover_on_the_full_runner_path(tmp_path, monkeypatch):
    """The Phase 4C thesis, checked on the runner's own wiring rather than
    only in the unit tests: `rf_signal_cost` must trade less than `rf_signal`.

    Guards against the variant being constructed without its penalty actually
    reaching the optimizer — a wiring bug that would silently produce five
    identical strategies under five different names.
    """
    _write_gold_snapshot(tmp_path)
    _patch(monkeypatch, tmp_path)

    all_results = run_phase4c.run_phase4c()

    for universe_results in all_results.values():
        by_name = {r.strategy_name: r for r in universe_results}
        assert by_name["rf_signal_cost"].turnover.mean() < by_name["rf_signal"].turnover.mean()
