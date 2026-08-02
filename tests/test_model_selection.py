"""
test_model_selection.py — Phase 5 leak-free hyperparameter selection.

Offline/synthetic. Proves the two selectors pick sensibly and honestly, and
that the IC scorer behaves at its extremes. The purge itself is proven by
tests/test_purged_kfold.py — this suite trusts that and tests the selection
built on top of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics import DSRTrialLedger
from model_selection import (
    _instantiate,
    information_coefficient,
    select_ml_hyperparameters,
    select_portfolio_levers,
)


class TestInformationCoefficient:
    def test_perfectly_ordered_prediction_scores_near_one(self):
        y = np.array([0.01, -0.02, 0.03, 0.00, 0.05, -0.01])
        assert information_coefficient(y, 2 * y + 0.001) == pytest.approx(1.0)

    def test_reversed_prediction_scores_near_minus_one(self):
        y = np.array([0.01, -0.02, 0.03, 0.00, 0.05, -0.01])
        assert information_coefficient(y, -y) == pytest.approx(-1.0)

    def test_pure_noise_scores_near_zero(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=2000)
        pred = rng.normal(size=2000)
        assert abs(information_coefficient(y, pred)) < 0.1

    def test_constant_prediction_scores_zero_not_nan(self):
        y = np.array([0.01, -0.02, 0.03, 0.00])
        assert information_coefficient(y, np.zeros_like(y)) == 0.0


def _returns_with_signal(n_dates=260, assets=("A", "B", "C", "D"), seed=0):
    """Returns where each asset's next return is partly predictable from its
    own trailing return — so a deeper tree should earn a higher IC."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates, name="Date")
    base = rng.normal(0.0003, 0.01, size=(n_dates, len(assets)))
    # inject mild momentum: tomorrow ~ 0.15 * today + noise
    for j in range(len(assets)):
        for t in range(1, n_dates):
            base[t, j] += 0.15 * base[t - 1, j]
    return pd.DataFrame(base, index=dates, columns=list(assets))


class TestSelectMLHyperparameters:
    def test_xgboost_cv_uses_a_single_native_worker_by_default(self):
        """Regression guard for Phase 5's native -11 crash on macOS."""
        model = _instantiate("xgboost", {"n_estimators": 5, "max_depth": 2})
        assert model.get_params()["n_jobs"] == 1

    def test_xgboost_cv_allows_an_explicit_reviewed_worker_override(self):
        model = _instantiate("xgboost", {"n_estimators": 5, "max_depth": 2, "n_jobs": 2})
        assert model.get_params()["n_jobs"] == 2

    def test_returns_best_params_from_the_grid_and_a_full_table(self):
        returns = _returns_with_signal()
        grid = {"max_depth": [2, 4], "n_estimators": [40]}
        best, table = select_ml_hyperparameters(
            returns, features=None, grid=grid, model_type="random_forest",
            n_splits=4, embargo_frac=0.02, condition_on_regime=False,
        )
        assert set(best) == {"max_depth", "n_estimators"}
        assert best["max_depth"] in (2, 4)
        assert isinstance(best["max_depth"], int)          # int type preserved
        assert len(table) == 2                              # 2 grid points
        assert {"mean_ic", "std_ic", "n_folds"} <= set(table.columns)
        assert table["mean_ic"].iloc[0] >= table["mean_ic"].iloc[1]   # sorted desc

    @pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
    def test_runs_for_both_model_types(self, model_type):
        returns = _returns_with_signal(seed=2)
        grid = {"max_depth": [3], "n_estimators": [30]}
        best, table = select_ml_hyperparameters(
            returns, features=None, grid=grid, model_type=model_type,
            n_splits=3, embargo_frac=0.0, condition_on_regime=False,
        )
        assert table["n_folds"].iloc[0] == 3
        assert np.isfinite(table["mean_ic"].iloc[0])


class TestSelectPortfolioLevers:
    def _backtest_kwargs(self):
        return {
            "rebalance_freq": "ME",
            "min_train_days": 120,
            "cost_bps": 10.0,
            "max_weight": 0.5,
            "risk_free_annual": 0.0,
            "min_train_rows": 40,
            "condition_on_regime": False,
            "short_window": 10,
            "long_window": 20,
            "momentum_windows": (5, 10),
            "universe_name": "test",
        }

    def test_selects_a_grid_point_and_records_every_trial(self):
        returns = _returns_with_signal(n_dates=320, seed=1)
        features = pd.DataFrame(
            {"MARKET_RETURN": returns.mean(axis=1)}, index=returns.index
        )
        ledger = DSRTrialLedger()
        shrink_grid = [0.0, 0.5]
        penalty_grid = [0.0, 1.0]
        best, table = select_portfolio_levers(
            returns, features, model_type="random_forest",
            ml_params={"n_estimators": 20, "max_depth": 3},
            shrink_grid=shrink_grid, penalty_grid=penalty_grid,
            backtest_kwargs=self._backtest_kwargs(),
            ledger=ledger, universe="test",
        )
        assert set(best) == {"shrinkage_weight", "turnover_penalty"}
        assert best["shrinkage_weight"] in shrink_grid
        assert best["turnover_penalty"] in penalty_grid
        assert len(table) == 4                              # 2 x 2 grid
        # Every backtested grid point recorded for the DSR pool.
        assert ledger.n_trials("test") == 4
        # Winner has the highest validation net Sharpe.
        assert table["val_sharpe_net"].iloc[0] == table["val_sharpe_net"].max()
