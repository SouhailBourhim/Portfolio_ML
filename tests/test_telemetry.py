"""test_telemetry.py — no model label may hide degraded or fallback behaviour.

That sentence is the acceptance criterion for this work, and these tests are
what make it enforceable rather than aspirational. Three properties matter:

1. A substitution is RECORDED, with a reason.
2. The recorded rate does not depend on the memoization cache being warm.
3. The rebalance-level label reflects the substitution, so a chart legend and
   the audit field cannot disagree about what produced a number.

Property 2 is the subtle one and the reason this file exists. Both estimators
are content-addressed cached; a cache hit does not re-run the estimator, so a
naive implementation would record a fallback once and then stop, and the
measured fallback RATE would fall as the cache warmed. A metric whose whole
job is to be trustworthy must not decay with repetition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import telemetry
from backtest import build_cost_vector, run_backtest
from strategies import DCCGarchStrategy, EqualWeight, RandomForestSignalStrategy


ASSETS = ["SPY", "QQQ", "GLD", "TLT"]


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=700)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0004, 0.011, (len(dates), len(ASSETS))), index=dates, columns=ASSETS
    )


@pytest.fixture(scope="module")
def costs() -> pd.Series:
    return build_cost_vector(ASSETS, 10, 30)


class TestFitRecord:
    def test_a_fallback_without_a_reason_is_rejected(self):
        with pytest.raises(ValueError, match="must carry a reason"):
            telemetry.FitRecord(
                model_requested="dcc_garch", model_effective="ledoit_wolf",
                fit_status=telemetry.STATUS_FALLBACK, n_training_rows=100,
            )

    def test_a_fallback_that_substituted_nothing_is_rejected(self):
        """'fallback' and 'degraded' are different facts and must stay so."""
        with pytest.raises(ValueError, match="status is 'degraded'"):
            telemetry.FitRecord(
                model_requested="dcc_garch", model_effective="dcc_garch",
                fit_status=telemetry.STATUS_FALLBACK, n_training_rows=100,
                fallback_reason="something",
            )

    def test_an_unknown_status_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown fit_status"):
            telemetry.FitRecord(
                model_requested="x", model_effective="x",
                fit_status="probably_fine", n_training_rows=1,
            )


class TestCollector:
    def test_records_outside_a_window_are_dropped_not_raised(self):
        """Notebooks and experiments call these estimators with no interest in
        telemetry; a diagnostic that raised there would be worse than the
        opacity it replaces."""
        telemetry.record(telemetry.FitRecord("a", "a", telemetry.STATUS_OK, 1))

    def test_a_window_captures_only_its_own_records(self):
        with telemetry.collect() as outer:
            telemetry.record(telemetry.FitRecord("a", "a", telemetry.STATUS_OK, 1))
            with telemetry.collect() as inner:
                telemetry.record(telemetry.FitRecord("b", "b", telemetry.STATUS_OK, 2))
            assert [r.model_requested for r in inner] == ["b"]
        # The inner window's records still reach the outer one — nothing is lost.
        assert [r.model_requested for r in outer] == ["a", "b"]


class TestSummary:
    def test_a_clean_fit_keeps_the_strategy_label_unchanged(self):
        row = telemetry.summarize(
            [telemetry.FitRecord("dcc_garch", "dcc_garch", telemetry.STATUS_OK, 500)],
            "dcc_garch", 500,
        )
        assert row["model_effective"] == "dcc_garch"
        assert row["fit_status"] == telemetry.STATUS_OK

    def test_a_fallback_names_the_substitute_in_the_effective_label(self):
        """The acceptance criterion, at the level a reader sees."""
        row = telemetry.summarize(
            [telemetry.FitRecord(
                "dcc_garch", "ledoit_wolf", telemetry.STATUS_FALLBACK, 500,
                fallback_reason="GARCH did not converge",
            )],
            "dcc_garch", 500,
        )
        assert row["model_effective"] == "dcc_garch [via ledoit_wolf]"
        assert row["fit_status"] == telemetry.STATUS_FALLBACK
        assert "did not converge" in row["fallback_reason"]

    def test_a_rebalance_with_no_estimator_records_is_reported_ok(self):
        """EqualWeight fits no estimator; absence of news is good news here."""
        assert telemetry.summarize([], "equal_weight", 300)["fit_status"] == telemetry.STATUS_OK


class TestEngineRecordsEveryRebalance:
    def test_every_rebalance_gets_a_row_with_all_six_fields(self, returns, costs):
        result = run_backtest(returns, EqualWeight(), cost_bps=costs, universe_name="synthetic")
        assert len(result.fit_reports) == len(result.rebalance_dates)
        assert list(result.fit_reports.columns) == [
            "model_requested", "model_effective", "fit_status",
            "n_training_rows", "fallback_reason", "convergence_warning",
        ]
        assert result.fit_reports.index.equals(pd.Index(result.rebalance_dates))

    def test_a_strategy_forced_to_fall_back_reports_every_rebalance_degraded(
        self, returns, costs
    ):
        """An impossible row floor makes the ML signal a sample-mean strategy.

        Its Sharpe would still be filed under 'rf_signal' — which is exactly
        the mislabelling this work exists to prevent.
        """
        strategy = RandomForestSignalStrategy(min_train_rows=10**9)
        result = run_backtest(returns, strategy, cost_bps=costs, universe_name="synthetic")
        assert result.fallback_rate == 1.0
        assert (result.fit_reports["model_effective"]
                == "rf_signal [via naive_sample_mean]").all()
        assert result.fit_reports["fallback_reason"].notna().all()

    def test_fallback_mask_covers_the_days_a_degraded_fit_governed(self, returns, costs):
        """A rebalance's weights are in force until the NEXT rebalance.

        Flagging only the rebalance dates would attribute one day's return to
        a month of degraded weights.
        """
        strategy = RandomForestSignalStrategy(min_train_rows=10**9)
        result = run_backtest(returns, strategy, cost_bps=costs, universe_name="synthetic")
        mask = result.fallback_mask()
        assert mask.index.equals(result.net_returns.index)
        assert mask.all(), "every day should be flagged when every fit fell back"
        assert mask.sum() > len(result.rebalance_dates), (
            "the mask must propagate forward, not mark rebalance dates alone"
        )

    def test_a_strategy_with_no_fallbacks_reports_a_zero_rate(self, returns, costs):
        result = run_backtest(returns, EqualWeight(), cost_bps=costs, universe_name="synthetic")
        assert result.fallback_rate == 0.0
        assert not result.fallback_mask().any()


class TestTheRateDoesNotDecayWithTheCache:
    """The property that makes the number trustworthy.

    Both estimators are content-addressed cached. If the FitRecord were not
    stored and re-emitted alongside the cached value, a fallback would be
    counted on the cold run and vanish on the warm one — the measured rate
    would fall the more the pipeline was exercised.
    """

    def test_dcc_fallback_rate_is_identical_warm_and_cold(self, returns, costs):
        import dcc_garch

        dcc_garch._DCC_CACHE.clear()
        cold = run_backtest(returns, DCCGarchStrategy(), cost_bps=costs, universe_name="u")
        warm = run_backtest(returns, DCCGarchStrategy(), cost_bps=costs, universe_name="u")

        assert warm.fallback_rate == cold.fallback_rate
        pd.testing.assert_frame_equal(warm.fit_reports, cold.fit_reports)

    def test_ml_signal_fallback_rate_is_identical_warm_and_cold(self, returns, costs):
        import ml_signals

        ml_signals._PREDICTION_CACHE.clear()
        strategy = lambda: RandomForestSignalStrategy(min_train_rows=10**9)  # noqa: E731
        cold = run_backtest(returns, strategy(), cost_bps=costs, universe_name="u")
        warm = run_backtest(returns, strategy(), cost_bps=costs, universe_name="u")

        assert warm.fallback_rate == cold.fallback_rate == 1.0
        pd.testing.assert_frame_equal(warm.fit_reports, cold.fit_reports)

    def test_the_cache_actually_served_a_hit(self, returns, costs):
        """Guards the guard: if the cache never hit, the test above proves nothing."""
        import dcc_garch

        dcc_garch._DCC_CACHE.clear()
        run_backtest(returns, DCCGarchStrategy(), cost_bps=costs, universe_name="u")
        stats_before = dcc_garch._DCC_CACHE.stats()
        run_backtest(returns, DCCGarchStrategy(), cost_bps=costs, universe_name="u")
        stats_after = dcc_garch._DCC_CACHE.stats()
        assert stats_after["hits"] > stats_before["hits"], (
            "the second run did not hit the cache, so the warm/cold comparison "
            "above was not testing what it claims to test"
        )
