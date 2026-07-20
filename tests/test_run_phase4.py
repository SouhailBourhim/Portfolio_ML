"""
test_run_phase4.py — Smoke test for the Phase 4 comparison runner.

Uses a tiny synthetic Gold snapshot under tmp_path (never a real download,
never the real mlruns/ directory) so this stays offline, fast, and doesn't
pollute the project's actual experiment history. Not exhaustive — just
verifies the runner's output shape and trial-pool honesty, since
run_backtest.py's Phase 2 runner has no equivalent pytest coverage either.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_phase4
from ml_features import build_ml_feature_set


def _write_gold_snapshot(tmp_path, n: int = 200, assets=("SPY", "QQQ", "GLD"), seed: int = 1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n, name="Date")
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(n, len(assets))), index=dates, columns=list(assets)
    )
    macro = pd.DataFrame({"VIX": 18 + np.cumsum(rng.normal(0, 0.3, n))}, index=dates)
    ml_config = {
        "volatility_short_window": 10,
        "volatility_long_window": 20,
        "correlation_window": 20,
        "correlation_min_periods": 10,
        "macro_lag_days": 1,
    }
    features = build_ml_feature_set(returns, macro, ml_config)

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
            "garch_p": 1,
            "garch_q": 1,
            "dcc_a_init": 0.02,
            "dcc_b_init": 0.95,
            "rescale_factor": 100,
        },
        "regime": {
            "n_states": 2,
            "covariance_type": "diag",
            "n_restarts": 2,
            "random_state_base": 0,
            "min_regime_train_days": 30,
            "bull_strategy": "max_sharpe",
            "bear_strategy": "min_variance_lw",
        },
        "phase4": {"results_path": "data/gold/phase4_results.json"},
    }


def test_run_phase4_writes_results_with_full_trial_pool(tmp_path, monkeypatch):
    _write_gold_snapshot(tmp_path)
    monkeypatch.setattr(run_phase4, "ROOT", tmp_path)
    monkeypatch.setattr(run_phase4, "load_params", lambda: _params())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow_test.db'}")

    all_results = run_phase4.run_phase4()

    assert set(all_results) == {"etf_2017", "full_2021"}
    for universe_results in all_results.values():
        # 4 Phase 2 baselines + 3 Phase 4 strategies — the shared trial pool.
        assert len(universe_results) == 7
        assert {r.strategy_name for r in universe_results} == {
            "equal_weight",
            "min_variance",
            "min_variance_lw",
            "max_sharpe",
            "min_variance_ewma",
            "dcc_garch",
            "regime_conditional",
        }

    results_path = tmp_path / "data" / "gold" / "phase4_results.json"
    assert results_path.exists()
    output = json.loads(results_path.read_text())
    assert set(output) == {"etf_2017", "full_2021"}
    for entry in output.values():
        assert entry["n_trials"] == 7
        assert "beats_phase2_hurdle" in entry
        assert "is_phase4_strategy" in entry
        # No stored phase2_hurdle.json under tmp_path -> comparison is None, not a crash.
        assert entry["beats_phase2_hurdle"] is None


def test_run_phase4_compares_against_a_stored_hurdle(tmp_path, monkeypatch):
    _write_gold_snapshot(tmp_path)
    hurdle = {
        "etf_2017": {"strategy": "equal_weight", "sharpe_net": -999.0},
        "full_2021": {"strategy": "equal_weight", "sharpe_net": -999.0},
    }
    (tmp_path / "data" / "gold" / "phase2_hurdle.json").write_text(json.dumps(hurdle))

    monkeypatch.setattr(run_phase4, "ROOT", tmp_path)
    monkeypatch.setattr(run_phase4, "load_params", lambda: _params())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow_test.db'}")

    run_phase4.run_phase4()

    output = json.loads((tmp_path / "data" / "gold" / "phase4_results.json").read_text())
    for entry in output.values():
        # An absurdly low stored hurdle (-999) must be beaten by anything real.
        assert entry["beats_phase2_hurdle"] is True
