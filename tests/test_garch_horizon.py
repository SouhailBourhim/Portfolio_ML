"""
Multi-period GARCH variance: does the closed form match the recursion it claims
to summarise?

`src/garch_horizon.py` replaces a 21-step forecast loop with one algebraic
expression. That is the kind of substitution which is either exactly right or
quietly wrong by a factor nobody notices, so the central test compares it
against the explicit recursion on random parameter draws rather than on a
hand-picked example.

The limiting cases matter as much as the general one: A(p, 1) must be exactly 1
because a one-period hold needs no aggregation, and A(1, H) must be exactly 1
because integrated GARCH never mean-reverts and so carries today's deviation
forward unchanged at every horizon.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from garch_horizon import (
    aggregation_factor, forecast_variance_path, multi_horizon_average_variance,
    one_step_forecast_variance,
)


class TestTheClosedFormMatchesTheRecursion:
    def test_on_random_admissible_parameters(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            alpha = float(rng.uniform(0.01, 0.20))
            beta = float(rng.uniform(0.50, 0.98 - alpha))
            omega = float(rng.uniform(1e-7, 1e-5))
            var1 = float(rng.uniform(1e-5, 1e-3))
            horizon = int(rng.integers(1, 60))
            closed = multi_horizon_average_variance(omega, alpha, beta, var1, horizon)
            explicit = float(forecast_variance_path(omega, alpha, beta, var1, horizon).mean())
            assert np.isclose(closed, explicit, rtol=1e-9), (
                f"closed form {closed} != recursion {explicit} at "
                f"alpha={alpha}, beta={beta}, H={horizon}"
            )

    def test_the_path_decays_geometrically_toward_the_unconditional_level(self):
        omega, alpha, beta, horizon = 2e-6, 0.08, 0.90, 40
        uncond = omega / (1 - alpha - beta)
        path = forecast_variance_path(omega, alpha, beta, uncond * 4, horizon)
        gaps = np.abs(path - uncond)
        assert np.all(np.diff(gaps) < 0), "forecast did not move monotonically toward uncond"
        ratios = gaps[1:] / gaps[:-1]
        np.testing.assert_allclose(ratios, alpha + beta, rtol=1e-8)


class TestTheAggregationFactor:
    def test_one_period_needs_no_aggregation(self):
        for p in (0.5, 0.9, 0.99):
            assert aggregation_factor(p, 1) == 1.0

    def test_integrated_garch_never_mean_reverts(self):
        for horizon in (1, 21, 250):
            assert aggregation_factor(1.0, horizon) == 1.0

    def test_it_falls_with_the_horizon_and_rises_with_persistence(self):
        factors = [aggregation_factor(0.95, h) for h in (1, 5, 21, 63)]
        assert factors == sorted(factors, reverse=True)
        by_p = [aggregation_factor(p, 21) for p in (0.80, 0.90, 0.95, 0.99)]
        assert by_p == sorted(by_p)

    def test_the_documented_project_value(self):
        """p ~ 0.95 over a 21-day hold: the number Limit #4's fix turns on."""
        assert aggregation_factor(0.95, 21) == pytest.approx(0.628, abs=5e-4)

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_a_persistence_outside_the_unit_interval_is_rejected(self, bad):
        with pytest.raises(ValueError, match="persistence"):
            aggregation_factor(bad, 21)

    def test_a_horizon_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="horizon"):
            aggregation_factor(0.95, 0)


class TestTheOneStepForecast:
    def test_it_reacts_to_the_most_recent_shock(self):
        """The property the current `dcc_garch` code discards by taking
        `conditional_volatility[T-1]`: a large r_tau must raise the forecast."""
        omega, alpha, beta, var = 2e-6, 0.10, 0.85, 1e-4
        calm = one_step_forecast_variance(omega, alpha, beta, 0.001, var)
        shock = one_step_forecast_variance(omega, alpha, beta, 0.05, var)
        assert shock > calm
        assert np.isclose(shock - calm, alpha * (0.05 ** 2 - 0.001 ** 2))

    def test_it_is_the_textbook_recursion(self):
        omega, alpha, beta, r, var = 3e-6, 0.07, 0.90, -0.02, 2e-4
        assert np.isclose(one_step_forecast_variance(omega, alpha, beta, r, var),
                          omega + alpha * r ** 2 + beta * var)


class TestTheDirectionOfTheCorrection:
    def test_aggregation_pulls_a_calm_day_up_and_a_turbulent_day_down(self):
        """Limit #4's claim is that a one-day input systematically OVER-weights
        the current volatility state. That means the correction must be
        two-sided, which is why the experiment reports the high-volatility
        decile separately from the mean."""
        omega, alpha, beta, horizon = 2e-6, 0.08, 0.90, 21
        uncond = omega / (1 - alpha - beta)
        calm = multi_horizon_average_variance(omega, alpha, beta, uncond * 0.25, horizon)
        wild = multi_horizon_average_variance(omega, alpha, beta, uncond * 4.0, horizon)
        assert calm > uncond * 0.25, "aggregation should raise a below-average variance"
        assert wild < uncond * 4.0, "aggregation should lower an above-average variance"
        assert calm < uncond < wild, "aggregation must not cross the unconditional level"
