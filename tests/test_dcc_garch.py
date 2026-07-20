"""
test_dcc_garch.py — Tests for the DCC-GARCH covariance estimator.

Uses tiny synthetic windows throughout (never real Gold data) so the suite
stays fast and offline; a full walk-forward run against real data is a
separate, deliberately slow manual step (see src/run_phase4.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

import dcc_garch
from dcc_garch import DCCGarchNonConvergence, dcc_covariance


def _synthetic_returns(n: int = 200, assets: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-04", periods=n)
    data = rng.normal(0.0003, 0.01, size=(n, len(assets)))
    return pd.DataFrame(data, index=dates, columns=list(assets))


class TestCovarianceValidity:
    def test_returns_symmetric_positive_semidefinite_matrix(self):
        returns = _synthetic_returns()
        cov = dcc_covariance(returns)

        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() > -1e-8

    def test_implied_correlations_are_bounded(self):
        returns = _synthetic_returns()
        cov = dcc_covariance(returns)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        off_diag = corr[~np.eye(3, dtype=bool)]
        assert np.all(off_diag >= -1.0 - 1e-8)
        assert np.all(off_diag <= 1.0 + 1e-8)

    def test_diagonal_is_positive_variance(self):
        returns = _synthetic_returns()
        cov = dcc_covariance(returns)
        assert np.all(np.diag(cov) > 0)


class TestReactsToCorrelationShift:
    def test_dcc_correlation_exceeds_static_lw_after_a_planted_shift(self):
        # First half: A and B move independently (corr ~ 0). Second half: B is
        # A plus small noise (corr ~ 1). A flat-window estimator (Ledoit-Wolf)
        # blends both regimes into one moderate correlation; DCC's recursion
        # weights recent co-movement more heavily and should read the shift.
        # This is the load-bearing proof DCC-GARCH is genuinely "dynamic" at
        # the CORRELATION level (P3), not just at the per-asset volatility
        # level EWMA already covers.
        rng = np.random.default_rng(7)
        n = 100
        dates = pd.bdate_range("2021-01-04", periods=2 * n)

        a1 = rng.normal(0.0, 0.01, n)
        b1 = rng.normal(0.0, 0.01, n)
        a2 = rng.normal(0.0, 0.01, n)
        b2 = a2 + rng.normal(0.0, 0.001, n)

        returns = pd.DataFrame(
            {"A": np.concatenate([a1, a2]), "B": np.concatenate([b1, b2])}, index=dates
        )

        cov_dcc = dcc_covariance(returns)
        d_dcc = np.sqrt(np.diag(cov_dcc))
        corr_dcc = cov_dcc[0, 1] / (d_dcc[0] * d_dcc[1])

        lw = LedoitWolf().fit(returns.to_numpy())
        d_lw = np.sqrt(np.diag(lw.covariance_))
        corr_lw = lw.covariance_[0, 1] / (d_lw[0] * d_lw[1])

        assert corr_dcc > corr_lw


class TestFallbackOnNonConvergence:
    def test_falls_back_to_ledoit_wolf_and_warns_on_garch_failure(self, monkeypatch, caplog):
        def _raise(*args, **kwargs):
            raise DCCGarchNonConvergence("forced failure for test")

        monkeypatch.setattr(dcc_garch, "_fit_univariate_garch", _raise)

        returns = _synthetic_returns()
        with caplog.at_level("WARNING", logger="dcc_garch"):
            cov = dcc_covariance(returns)

        expected = LedoitWolf().fit(returns.to_numpy()).covariance_ * dcc_garch.TRADING_DAYS_PER_YEAR
        np.testing.assert_allclose(cov, expected)
        assert any("falling back to Ledoit-Wolf" in record.message for record in caplog.records)

    def test_falls_back_when_dcc_optimization_fails(self, monkeypatch, caplog):
        def _raise(*args, **kwargs):
            raise DCCGarchNonConvergence("forced DCC failure for test")

        monkeypatch.setattr(dcc_garch, "_fit_dcc", _raise)

        returns = _synthetic_returns()
        with caplog.at_level("WARNING", logger="dcc_garch"):
            cov = dcc_covariance(returns)

        expected = LedoitWolf().fit(returns.to_numpy()).covariance_ * dcc_garch.TRADING_DAYS_PER_YEAR
        np.testing.assert_allclose(cov, expected)
        assert any("falling back to Ledoit-Wolf" in record.message for record in caplog.records)


class TestDCCRecursionMechanics:
    def test_final_q_diagonal_reflects_persistence(self):
        # A sanity check on the recursion helper itself, independent of GARCH:
        # with a=0, b close to 1, Q_t should barely move from Q_bar (pure
        # persistence, shocks ignored) — a hand-checkable degenerate case.
        rng = np.random.default_rng(1)
        std_resid = rng.normal(0.0, 1.0, size=(50, 2))
        q_bar = np.cov(std_resid, rowvar=False)

        q_final = dcc_garch._dcc_recursion_final_q(std_resid, a=0.0, b=0.999, q_bar=q_bar)
        np.testing.assert_allclose(q_final, q_bar, atol=0.5)
