"""test_explainability.py — the explanation must describe the real decision.

An explanation is only worth its artifact if it (a) sees exactly what the model
saw, (b) is reproducible, and (c) is internally consistent. Each is a test here.

Tests are named after the rule they lock in, and use small synthetic fixtures
so the suite stays offline and fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explainability import (
    CAP_TOLERANCE,
    explain_expected_return_ranking,
    feature_distribution_summary,
    local_contributions,
    permutation_importance,
    trace_regime_decision,
)


ASSETS = ["AAA", "BBB", "CCC", "DDD"]


def assert_traces_agree(left, right, path: str = "") -> None:
    """Compare two traces: structure exactly, floats to tolerance.

    Exact equality is the wrong instrument here, and not for convenience.
    Writing into a frame beyond tau changes the block layout of the COPY, so
    numpy's pairwise summation walks a differently-aligned buffer and a mean
    over the identical 501 leading values can differ in the last bit. That is
    a summation-order artifact, not information flowing backwards from the
    future — every structural field (regime, sub-optimizer, weights,
    constraints, fallback) is still compared exactly, and those are what the
    no-lookahead guarantee is about.
    """
    assert type(left) is type(right), f"type differs at {path or 'root'}"
    if isinstance(left, dict):
        assert left.keys() == right.keys(), f"keys differ at {path or 'root'}"
        for key in left:
            assert_traces_agree(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert len(left) == len(right), f"length differs at {path}"
        for i, (a, b) in enumerate(zip(left, right)):
            assert_traces_agree(a, b, f"{path}[{i}]")
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=1e-9, abs=1e-12), (
            f"float differs materially at {path}: {left} vs {right}"
        )
    else:
        assert left == right, f"value differs at {path}: {left!r} vs {right!r}"


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=700)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0004, 0.011, size=(len(dates), len(ASSETS))),
        index=dates, columns=ASSETS,
    )


@pytest.fixture(scope="module")
def market_features(returns) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MARKET_RETURN": returns.mean(axis=1),
            "MARKET_VOL_SHORT": returns.mean(axis=1).rolling(21).std().bfill(),
            "AVG_PAIRWISE_CORR": returns.rolling(63).corr().groupby(level=0).mean().mean(axis=1).bfill(),
        },
        index=returns.index,
    ).fillna(0.0)


@pytest.fixture(scope="module")
def fitted_forest(returns):
    from sklearn.ensemble import RandomForestRegressor

    X = pd.DataFrame(
        {
            "MOM": returns["AAA"].rolling(21).sum().bfill(),
            "VOL": returns["BBB"].rolling(21).std().bfill(),
            "REGIME_BULL_PROB": np.linspace(0.2, 0.8, len(returns)),
        },
        index=returns.index,
    )
    y = returns["AAA"].shift(-1).fillna(0.0)
    model = RandomForestRegressor(n_estimators=12, max_depth=4, random_state=0, n_jobs=1)
    model.fit(X.to_numpy(), y.to_numpy())
    return model, X, y


class TestDecisionTraceDescribesTheRealDecision:
    def test_trace_reads_no_row_after_the_decision_date(self, returns, market_features):
        """The whole guarantee. Corrupting the FUTURE must not move the trace."""
        tau = returns.index[500]
        clean = trace_regime_decision(returns.loc[:tau], market_features.loc[:tau])

        poisoned_returns = returns.copy()
        poisoned_features = market_features.copy()
        poisoned_returns.loc[poisoned_returns.index > tau] = 99.0
        poisoned_features.loc[poisoned_features.index > tau] = -99.0
        poisoned = trace_regime_decision(
            poisoned_returns.loc[:tau], poisoned_features.loc[:tau]
        )
        # Structural fields exactly; floats to tolerance — see the helper.
        assert_traces_agree(clean, poisoned)
        assert clean["weights"] == poisoned["weights"], (
            "Weights changed when data AFTER the decision date was corrupted. "
            "Either the trace reads beyond tau, or the slicing is wrong."
        )
        assert clean["decision"] == poisoned["decision"]

    def test_trace_is_deterministic_across_repeated_calls(self, returns, market_features):
        tau = returns.index[400]
        first = trace_regime_decision(returns.loc[:tau], market_features.loc[:tau])
        second = trace_regime_decision(returns.loc[:tau], market_features.loc[:tau])
        assert first == second, (
            "An explanation that changes between runs cannot be audited."
        )

    def test_weights_sum_to_one_and_respect_the_cap(self, returns, market_features):
        tau = returns.index[400]
        trace = trace_regime_decision(returns.loc[:tau], market_features.loc[:tau])
        weights = trace["weights"]
        assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0
        assert max(weights.values()) <= trace["constraints"]["max_weight"] + 1e-6

    def test_assets_at_cap_are_reported_with_tolerance_not_equality(
        self, returns, market_features
    ):
        """SLSQP returns a converged optimum, not an exact boundary value.

        An equality test would report "nothing was binding" for a portfolio
        pinned to the cap, which is the single most important fact about
        allocation on a small universe.
        """
        tau = returns.index[400]
        trace = trace_regime_decision(returns.loc[:tau], market_features.loc[:tau])
        cap = trace["constraints"]["max_weight"]
        expected = {a for a, w in trace["weights"].items() if w >= cap - CAP_TOLERANCE}
        assert set(trace["constraints"]["assets_at_cap"]) == expected

    def test_a_nonconverged_fit_reports_the_defensive_fallback_and_says_why(
        self, returns, market_features
    ):
        """Below the history floor the trace must name the fallback, not hide it."""
        tau = returns.index[300]
        trace = trace_regime_decision(
            returns.loc[:tau], market_features.loc[:tau], min_regime_train_days=10**6
        )
        assert trace["fallback_used"] is True
        assert trace["fallback_reason"]
        assert trace["decision"]["selected_regime"] == "bear", (
            "An uncertain regime read must resolve to the DEFENSIVE sub-strategy."
        )

    def test_binding_note_flags_a_cap_that_nearly_determines_the_allocation(
        self, returns, market_features
    ):
        tau = returns.index[400]
        trace = trace_regime_decision(
            returns.loc[:tau], market_features.loc[:tau], max_weight=0.26
        )
        # 4 assets x 0.26 = 1.04 -> every feasible portfolio is nearly pinned.
        assert trace["constraints"]["cap_is_near_determining"] is True


class TestContributionsAreExactAndAdditive:
    """The property that makes a consistency check meaningful."""

    def test_forest_contributions_reconstruct_the_prediction(self, fitted_forest):
        model, X, _ = fitted_forest
        result = local_contributions(model, X.iloc[-1], "random_forest")
        assert result["reconstruction_error"] < 1e-9, (
            f"bias + contributions must reproduce the prediction; error was "
            f"{result['reconstruction_error']}."
        )

    def test_xgboost_contributions_reconstruct_the_prediction(self, fitted_forest):
        xgb = pytest.importorskip("xgboost")
        _, X, y = fitted_forest
        model = xgb.XGBRegressor(n_estimators=12, max_depth=3, random_state=0, n_jobs=1)
        model.fit(X.to_numpy(), y.to_numpy())
        result = local_contributions(model, X.iloc[-1], "xgboost")
        assert result["reconstruction_error"] < 1e-5

    def test_contribution_keys_match_the_training_feature_names(self, fitted_forest):
        model, X, _ = fitted_forest
        result = local_contributions(model, X.iloc[-1], "random_forest")
        assert list(result["contributions"]) == list(X.columns), (
            "Contributions keyed by anything other than the training matrix's own "
            "columns would silently mislabel which feature did what."
        )

    def test_local_contributions_are_deterministic(self, fitted_forest):
        model, X, _ = fitted_forest
        a = local_contributions(model, X.iloc[-1], "random_forest")
        b = local_contributions(model, X.iloc[-1], "random_forest")
        assert a == b


class TestGlobalImportance:
    def test_permutation_importance_is_seeded_and_reproducible(self, fitted_forest):
        model, X, y = fitted_forest
        assert permutation_importance(model, X, y, seed=0) == permutation_importance(
            model, X, y, seed=0
        )

    def test_importance_covers_exactly_the_training_features(self, fitted_forest):
        model, X, y = fitted_forest
        assert set(permutation_importance(model, X, y, seed=0)) == set(X.columns)

    def test_importance_is_returned_in_descending_order(self, fitted_forest):
        model, X, y = fitted_forest
        values = list(permutation_importance(model, X, y, seed=0).values())
        assert values == sorted(values, reverse=True)


class TestRankingToWeights:
    def test_ranking_link_reports_cap_membership_per_asset(self):
        mu = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=ASSETS)
        result = explain_expected_return_ranking(mu, weights, max_weight=0.25)
        assert all(row["at_cap"] for row in result["per_asset"].values())

    def test_concordance_is_undefined_not_zero_when_the_cap_pins_every_weight(self):
        """A fully-binding cap leaves no ordering to compare against.

        Reporting 0.0 here would read as "the weights ignore the ranking"; the
        truth is that the constraint left the optimizer nothing to express.
        """
        mu = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=ASSETS)
        result = explain_expected_return_ranking(mu, weights, max_weight=0.25)
        assert result["rank_concordance_spearman"] is None
        assert result["concordance_undefined_reason"]

    def test_a_reversed_allocation_is_reported_not_smoothed(self):
        """The interesting case: the optimizer contradicts the ranking."""
        mu = pd.Series([0.4, 0.3, 0.2, 0.1], index=ASSETS)
        weights = pd.Series([0.0, 0.2, 0.3, 0.5], index=ASSETS)
        result = explain_expected_return_ranking(mu, weights, max_weight=0.5)
        assert result["rank_concordance_spearman"] == pytest.approx(-1.0)

    def test_feature_distribution_reports_missingness(self):
        frame = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0], "B": [1.0, 1.0, 1.0, 1.0]})
        summary = feature_distribution_summary(frame)
        assert summary["A"]["missing"] == 1
        assert summary["B"]["missing"] == 0
