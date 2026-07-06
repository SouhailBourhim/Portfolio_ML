"""
test_backtest.py — The no-lookahead suite and engine mechanics.

This file is Phase 2's gate (CLAUDE.md §10.4): no modeling result is
credible until these tests are green. Each test is named for the rule it
locks in.
"""

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestResult, build_cost_vector, run_backtest
from metrics import annualized_sharpe
from strategies import EqualWeight, Strategy


# ── Instrumented strategies used as probes ───────────────────────────────────

class SpyStrategy(Strategy):
    """Records exactly what the engine lets it see."""
    name = "spy"

    def __init__(self) -> None:
        self.seen_train_ends: list[pd.Timestamp] = []
        self.seen_extras_ends: list[pd.Timestamp] = []

    def fit(self, train_returns, extras=None):
        self.seen_train_ends.append(train_returns.index.max())
        if extras:
            for frame in extras.values():
                self.seen_extras_ends.append(frame.index.max())
        n = train_returns.shape[1]
        return pd.Series(1.0 / n, index=train_returns.columns)


class PerfectForesight(Strategy):
    """
    A deliberately cheating strategy: constructed with the FULL returns frame
    (an out-of-band channel the engine knows nothing about). At fit() time it
    looks up the day AFTER its train window in the full frame and goes all-in
    on that day's winner. If the engine ever leaked future rows into the
    train window, this strategy's Sharpe would explode.
    """
    name = "perfect_foresight"

    def __init__(self, full_returns: pd.DataFrame) -> None:
        self.full_returns = full_returns

    def fit(self, train_returns, extras=None):
        last_seen = train_returns.index.max()
        future = self.full_returns.loc[self.full_returns.index > last_seen]
        w = pd.Series(0.0, index=train_returns.columns)
        if len(future) > 0:
            # Cheat: bet everything on tomorrow's actual winner
            w[future.iloc[0].idxmax()] = 1.0
        else:
            w[train_returns.iloc[-1].idxmax()] = 1.0
        return w


class MalformedWeights(Strategy):
    name = "malformed"

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def fit(self, train_returns, extras=None):
        n = train_returns.shape[1]
        if self.mode == "sum":
            return pd.Series(2.0 / n, index=train_returns.columns)      # sums to 2
        if self.mode == "negative":
            w = pd.Series(1.0 / n, index=train_returns.columns)
            w.iloc[0] = -0.5
            w.iloc[1] += 0.5 + 1.0 / n
            return w
        if self.mode == "index":
            return pd.Series(1.0 / n, index=[f"X{i}" for i in range(n)])  # wrong assets
        raise ValueError(self.mode)


# ── The no-lookahead gate ────────────────────────────────────────────────────

class TestNoLookahead:
    def test_engine_never_passes_future_data(self, synthetic_log_returns):
        spy = SpyStrategy()
        result = run_backtest(synthetic_log_returns, spy, min_train_days=100)
        for train_end, tau in zip(spy.seen_train_ends, result.rebalance_dates):
            assert train_end <= tau
        # And strictly before the first OOS day each window earns returns on
        oos_start = result.gross_returns.index.min()
        assert spy.seen_train_ends[0] < oos_start

    def test_engine_slices_extras_to_train_window(self, synthetic_log_returns):
        spy = SpyStrategy()
        extras = {"macro": synthetic_log_returns * 2}   # same index, full length
        result = run_backtest(synthetic_log_returns, spy, min_train_days=100, extras=extras)
        assert len(spy.seen_extras_ends) == len(result.rebalance_dates)
        for extras_end, tau in zip(spy.seen_extras_ends, result.rebalance_dates):
            assert extras_end <= tau

    def test_perfect_foresight_collapses_inside_engine(self, synthetic_log_returns):
        cheat = PerfectForesight(synthetic_log_returns)

        # God mode: hand the cheat the whole frame day by day OUTSIDE the engine
        simple = np.exp(synthetic_log_returns) - 1
        god_daily = []
        for i in range(100, len(synthetic_log_returns) - 1):
            train = synthetic_log_returns.iloc[: i + 1]
            w = cheat.fit(train)
            god_daily.append(float((w * simple.iloc[i + 1]).sum()))
        god_sharpe = annualized_sharpe(pd.Series(god_daily))

        # Through the engine: the cheat only ever receives properly sliced data,
        # so its "foresight" degenerates to betting on yesterday's winner.
        result = run_backtest(synthetic_log_returns, cheat, min_train_days=100)
        engine_sharpe = annualized_sharpe(result.gross_returns)

        assert god_sharpe > 5            # sanity: the cheat channel really works
        assert abs(engine_sharpe) < 1.5  # IID data: momentum has no real edge
        assert god_sharpe > 5 * abs(engine_sharpe)

    def test_new_weights_earn_returns_from_next_day_not_rebalance_day(self):
        # 2 assets, deterministic: asset A returns +10% ONLY on the rebalance
        # day itself. If the engine applied new weights same-day, a strategy
        # going all-in A would capture it; correct timing must NOT.
        dates = pd.bdate_range("2022-01-03", periods=8)
        log_r = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        log_r.loc[dates[4], "A"] = np.log(1.10)   # +10% simple on rebalance day

        class AllInA(Strategy):
            name = "all_in_a"
            def fit(self, train_returns, extras=None):
                return pd.Series({"A": 1.0, "B": 0.0})

        result = run_backtest(
            log_r, AllInA(), rebalance_freq="W-FRI", min_train_days=3
        )
        # dates[4] is a rebalance-window day; its +10% must NOT appear in OOS
        # returns earned by weights fitted ON that day
        first_rebalance = result.rebalance_dates[0]
        if first_rebalance == dates[4]:
            assert result.gross_returns.loc[dates[5]] == pytest.approx(0.0)
        assert result.gross_returns.max() < 0.10   # the spike day was never earned late

    def test_first_rebalance_respects_min_train_days(self, synthetic_log_returns):
        result = run_backtest(synthetic_log_returns, EqualWeight(), min_train_days=300)
        first_tau_pos = synthetic_log_returns.index.get_loc(result.rebalance_dates[0])
        assert first_tau_pos >= 299   # position 299 = 300th row


# ── Engine mechanics ─────────────────────────────────────────────────────────

class TestEngineMechanics:
    def test_portfolio_return_uses_simple_not_log_aggregation(self):
        # Big moves on the first OOS day, where log and simple differ materially:
        # A: log 0.2 → simple 22.14%; B: log −0.2 → simple −18.13%
        dates = pd.bdate_range("2022-01-03", periods=8)
        log_r = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        log_r.loc[dates[5], "A"] = 0.2        # dates[4] is the Friday rebalance
        log_r.loc[dates[5], "B"] = -0.2

        result = run_backtest(
            log_r, EqualWeight(), rebalance_freq="W-FRI", min_train_days=2
        )
        expected = 0.5 * (np.exp(0.2) - 1) + 0.5 * (np.exp(-0.2) - 1)  # +2.007%
        wrong_log_sum = 0.0                                             # naive log math
        actual = result.gross_returns.loc[dates[5]]
        assert actual == pytest.approx(expected, rel=1e-9)
        assert actual != pytest.approx(wrong_log_sum, abs=1e-4)

    def test_weights_drift_with_returns_between_rebalances(self):
        # Hand computation: start 50/50; day after first rebalance A +100%
        # (log ln2), B 0% → drifted: A 2/3, B 1/3.
        # 65 business days span 3 month-ends so ≥2 valid rebalances exist.
        dates = pd.bdate_range("2022-01-03", periods=65)
        log_r = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        log_r.iloc[21, 0] = np.log(2.0)  # first OOS day after the Jan-31 rebalance (pos 20)

        result = run_backtest(log_r, EqualWeight(), rebalance_freq="ME", min_train_days=10)
        assert len(result.rebalance_dates) >= 2
        drifted_at_2nd = result.drifted_weights.iloc[1]
        assert drifted_at_2nd["A"] == pytest.approx(2 / 3, rel=1e-9)
        assert drifted_at_2nd["B"] == pytest.approx(1 / 3, rel=1e-9)

    def test_turnover_measured_against_drifted_not_target_weights(self):
        dates = pd.bdate_range("2022-01-03", periods=65)
        log_r = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
        log_r.iloc[21, 0] = np.log(2.0)   # drift after first rebalance

        result = run_backtest(log_r, EqualWeight(), rebalance_freq="ME", min_train_days=10)
        # Second rebalance: back to 50/50 from drifted 2/3–1/3
        expected_turnover = abs(0.5 - 2 / 3) + abs(0.5 - 1 / 3)
        assert result.turnover.iloc[1] == pytest.approx(expected_turnover, rel=1e-9)

    def test_first_rebalance_turnover_is_one(self, synthetic_log_returns):
        result = run_backtest(synthetic_log_returns, EqualWeight(), min_train_days=100)
        assert result.turnover.iloc[0] == pytest.approx(1.0)

    def test_costs_reduce_net_but_never_gross(self, synthetic_log_returns):
        costly = run_backtest(
            synthetic_log_returns, EqualWeight(), min_train_days=100, cost_bps=50.0
        )
        assert (costly.net_returns <= costly.gross_returns + 1e-15).all()
        assert costly.net_returns.sum() < costly.gross_returns.sum()

    def test_zero_cost_makes_gross_equal_net(self, synthetic_log_returns):
        free = run_backtest(synthetic_log_returns, EqualWeight(), min_train_days=100)
        pd.testing.assert_series_equal(
            free.gross_returns, free.net_returns, check_names=False
        )

    def test_cost_deducted_exactly_once_per_rebalance(self, synthetic_log_returns):
        result = run_backtest(
            synthetic_log_returns, EqualWeight(), min_train_days=100, cost_bps=10.0
        )
        drag = (result.gross_returns - result.net_returns)
        n_charged_days = int((drag > 1e-15).sum())
        assert n_charged_days == len(result.rebalance_dates)
        assert drag.sum() == pytest.approx(result.costs.sum(), rel=1e-9)


# ── Weight validation and cost vector ────────────────────────────────────────

class TestEngineRejectsMalformedWeights:
    @pytest.mark.parametrize("mode,match", [
        ("sum", "sum"),
        ("negative", "negative"),
        ("index", "index"),
    ])
    def test_engine_rejects_bad_weights(self, synthetic_log_returns, mode, match):
        with pytest.raises(ValueError, match=match):
            run_backtest(synthetic_log_returns, MalformedWeights(mode), min_train_days=100)


class TestEngineEnforcesCap:
    """The cap is enforced at the trust boundary, not just promised by
    strategies — a Phase 4 model returning a concentrated book must fail."""

    class OverCap(Strategy):
        name = "over_cap"
        def fit(self, train_returns, extras=None):
            w = pd.Series(0.0, index=train_returns.columns)
            w.iloc[0] = 0.40                       # violates a 0.25 cap
            w.iloc[1:] = 0.60 / (len(w) - 1)
            return w

    def test_engine_rejects_weights_above_cap(self, synthetic_log_returns):
        with pytest.raises(ValueError, match="max_weight"):
            run_backtest(
                synthetic_log_returns, self.OverCap(),
                min_train_days=100, max_weight=0.25,
            )

    def test_engine_without_cap_allows_concentration(self, synthetic_log_returns):
        result = run_backtest(
            synthetic_log_returns, self.OverCap(), min_train_days=100
        )
        assert result.target_weights.iloc[0].max() == pytest.approx(0.40)


class TestCostVector:
    def test_bvc_assets_cost_more_than_etfs(self):
        cv = build_cost_vector(
            ["SPY", "IAM.CS", "GLD", "BCP.CS"], etf_cost_bps=10, bvc_cost_bps=30
        )
        assert cv["SPY"] == 10 and cv["GLD"] == 10
        assert cv["IAM.CS"] == 30 and cv["BCP.CS"] == 30

    def test_unknown_asset_in_cost_vector_raises(self):
        with pytest.raises(ValueError, match="Unknown asset"):
            build_cost_vector(["SPY", "MYSTERY"], etf_cost_bps=10, bvc_cost_bps=30)

    def test_override_wins_over_classification(self):
        cv = build_cost_vector(
            ["SPY", "IAM.CS"], etf_cost_bps=10, bvc_cost_bps=30, overrides={"IAM.CS": 99}
        )
        assert cv["IAM.CS"] == 99
