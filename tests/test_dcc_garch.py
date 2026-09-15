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


class TestTheLimit4Corrections:
    """Both Limit #4 defects are fixed as of 2026-09-15: the covariance uses the
    one-step FORECAST rather than the lagged conditional volatility, and averages
    the 1..H step forecasts over the holding period rather than repeating the
    one-day value. These pin that the corrections are actually applied, since the
    difference is a few percent and would not be obvious in any output."""

    @staticmethod
    def _panel(n=400, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2021-01-01", periods=n, freq="B")
        vol = 0.01 * (1 + 0.5 * np.sin(np.linspace(0, 8, n)))
        return pd.DataFrame(
            {c: rng.normal(0.0003, vol) for c in ("A", "B", "C")}, index=idx
        )

    def test_the_correction_changes_the_matrix(self):
        panel = self._panel()
        corrected = dcc_garch.dcc_covariance(panel)
        legacy = dcc_garch.dcc_covariance(panel, holding_days=0)
        assert not np.allclose(corrected, legacy), (
            "holding_days made no difference — the Limit #4 corrections are not "
            "reaching the returned covariance"
        )

    def test_the_result_is_still_a_valid_covariance_matrix(self):
        cov = dcc_garch.dcc_covariance(self._panel())
        assert np.allclose(cov, cov.T), "covariance is not symmetric"
        assert np.all(np.linalg.eigvalsh(cov) > -1e-10), "covariance is not PSD"
        assert np.all(np.isfinite(cov))

    def test_holding_days_one_is_the_forecast_fix_only(self):
        """H = 1 needs no aggregation, so it isolates the off-by-one: it must
        differ from the legacy path but by less than the full correction."""
        panel = self._panel()
        legacy = np.diag(dcc_garch.dcc_covariance(panel, holding_days=0))
        one = np.diag(dcc_garch.dcc_covariance(panel, holding_days=1))
        full = np.diag(dcc_garch.dcc_covariance(panel, holding_days=21))
        assert not np.allclose(one, legacy)
        assert not np.allclose(one, full)

    def test_a_longer_hold_pulls_variance_toward_the_unconditional_level(self):
        """The aggregation is monotone in the horizon, so successive holding
        periods must move variance in one direction, not oscillate."""
        panel = self._panel()
        variances = [
            float(np.diag(dcc_garch.dcc_covariance(panel, holding_days=h))[0])
            for h in (1, 5, 21, 63)
        ]
        assert variances == sorted(variances) or variances == sorted(variances, reverse=True)

    def test_the_cache_key_separates_different_holding_periods(self):
        """`dcc_covariance` is memoized on its inputs. If `holding_days` were
        missing from the key, the first call would poison every later one."""
        panel = self._panel(seed=3)
        a = dcc_garch.dcc_covariance(panel, holding_days=21)
        b = dcc_garch.dcc_covariance(panel, holding_days=0)
        c = dcc_garch.dcc_covariance(panel, holding_days=21)
        assert not np.allclose(a, b)
        np.testing.assert_allclose(a, c)
