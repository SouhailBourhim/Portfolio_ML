"""
test_api.py — FastAPI service contract tests.

Uses `TestClient` (httpx under the hood) against a tmp_path Gold snapshot —
no live server, no network, no real data/ tree. Fast enough to sit in the
normal suite.

The load-bearing test here is the last one: `/compare` must never return a
headline lift without its confidence interval attached. That is the same
integrity rule the dashboard page enforces visually, applied to the machine-
readable surface — an API consumer should find it *harder* to quote the gain
dishonestly than honestly.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import api.main as api_main

    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True)

    dates = pd.bdate_range("2023-01-02", periods=60, name="Date")
    equity_rows, weight_rows = [], []
    for universe in ("etf_2017", "full_2021"):
        for strategy in ("equal_weight", "max_sharpe", "regime_conditional"):
            for d in dates:
                equity_rows.append({
                    "Date": d, "universe": universe, "strategy": strategy,
                    "gross_return": 0.001, "net_return": 0.0009,
                })
            for asset, w in (("SPY", 0.6), ("QQQ", 0.4)):
                weight_rows.append({
                    "Date": dates[-1], "universe": universe, "strategy": strategy,
                    "asset": asset, "weight": w,
                })
    pd.DataFrame(equity_rows).to_parquet(gold / "dashboard_equity.parquet", index=False)
    pd.DataFrame(weight_rows).to_parquet(gold / "dashboard_weights.parquet", index=False)

    def _universe_block(lift_pct: float) -> dict:
        return {
            "strategies": {
                "equal_weight": {"sharpe_net": 0.9, "avg_turnover": 0.03},
                "max_sharpe": {"sharpe_net": 0.95, "avg_turnover": 0.05},
                "regime_conditional": {"sharpe_net": 1.1, "avg_turnover": 0.30},
            },
            "best_classical": {"name": "max_sharpe", "sharpe_net": 0.95},
            "best_ml": {"name": "regime_conditional", "sharpe_net": 1.1},
            "headline_lift_pct": lift_pct,
            "headline_lift_absolute_sharpe": 0.15,
            "oos_start": "2023-01-02",
            "oos_end": "2023-03-24",
            "phase5_test_window": {
                "regime_conditional": {"test_sharpe_net": 1.1,
                                       "test_sharpe_ci": [0.2, 2.0]},
                "equal_weight": {"test_sharpe_net": 0.9,
                                 "test_sharpe_ci": [-0.1, 1.9]},
                "test_start": "2023-02-01",
                "test_end": "2023-03-24",
            },
        }

    (gold / "dashboard_showcase.json").write_text(json.dumps({
        "universes": {
            "etf_2017": _universe_block(15.79),
            "full_2021": _universe_block(15.79),
        },
        "assets_per_universe": {"etf_2017": ["SPY", "QQQ"],
                                "full_2021": ["SPY", "QQQ"]},
        "rebalance_freq": "ME",
        "max_weight": 0.25,
        "cost_bps": {"etf": 10, "bvc": 30},
    }))

    monkeypatch.setattr(api_main, "GOLD", gold)
    for fn in (api_main._showcase, api_main._equity, api_main._weights):
        fn.cache_clear()
    yield TestClient(api_main.app)
    for fn in (api_main._showcase, api_main._equity, api_main._weights):
        fn.cache_clear()


class TestCatalogue:
    def test_health_reports_ok_when_artifacts_present(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["missing_artifacts"] == []

    def test_strategies_lists_universes_with_backtest_context(self, client):
        body = client.get("/strategies").json()
        assert set(body["universes"]) == {"etf_2017", "full_2021"}
        assert "regime_conditional" in body["universes"]["full_2021"]["strategies"]
        # Constraints must travel with the catalogue — a consumer needs to know
        # the results assume a 25% cap and real transaction costs.
        assert body["backtest_context"]["max_weight"] == 0.25
        assert body["backtest_context"]["cost_bps"]["bvc"] == 30


class TestResults:
    def test_metrics_for_one_strategy(self, client):
        body = client.get("/metrics",
                          params={"universe": "full_2021",
                                  "strategy": "regime_conditional"}).json()
        assert body["metrics"]["sharpe_net"] == 1.1

    def test_metrics_without_strategy_returns_all_plus_comparison(self, client):
        body = client.get("/metrics", params={"universe": "full_2021"}).json()
        assert set(body["strategies"]) == {"equal_weight", "max_sharpe",
                                           "regime_conditional"}
        assert body["best_ml"]["name"] == "regime_conditional"

    def test_equity_returns_a_base_100_curve(self, client):
        body = client.get("/equity", params={"universe": "etf_2017",
                                             "strategy": "equal_weight"}).json()
        assert len(body["dates"]) == len(body["values"]) == 60
        assert body["values"][0] == pytest.approx(100.09, abs=0.01)
        assert body["values"][-1] > body["values"][0]

    def test_gross_curve_is_above_net_curve(self, client):
        params = {"universe": "etf_2017", "strategy": "equal_weight"}
        net = client.get("/equity", params={**params, "net": True}).json()["values"][-1]
        gross = client.get("/equity", params={**params, "net": False}).json()["values"][-1]
        assert gross > net, "les coûts doivent réduire la performance nette"

    def test_weights_latest_only_returns_one_allocation(self, client):
        body = client.get("/weights", params={"universe": "full_2021",
                                              "strategy": "max_sharpe"}).json()
        assert body["weights"] == {"SPY": 0.6, "QQQ": 0.4}
        assert body["as_of"] == "2023-03-24"

    def test_weights_history_returns_every_rebalance(self, client):
        body = client.get("/weights", params={"universe": "full_2021",
                                              "strategy": "max_sharpe",
                                              "latest_only": False}).json()
        assert "history" in body and body["history"]


class TestErrors:
    def test_unknown_universe_is_404_with_the_valid_options(self, client):
        resp = client.get("/metrics", params={"universe": "nope"})
        assert resp.status_code == 404
        assert "etf_2017" in resp.json()["detail"]

    def test_unknown_strategy_is_404_with_the_valid_options(self, client):
        resp = client.get("/metrics", params={"universe": "full_2021",
                                              "strategy": "nope"})
        assert resp.status_code == 404
        assert "regime_conditional" in resp.json()["detail"]


class TestIntegrity:
    def test_compare_always_ships_the_confidence_interval_with_the_lift(self, client):
        """The machine-readable counterpart of the dashboard's integrity rule:
        an API consumer must not be able to fetch the headline gain WITHOUT
        also receiving the out-of-sample interval and the caveat. Quoting the
        number dishonestly should require actively discarding fields."""
        body = client.get("/compare", params={"universe": "full_2021"}).json()
        assert "lift_pct" in body
        test = body["out_of_sample_test"]
        for strategy in ("regime_conditional", "equal_weight"):
            ci = test[strategy]["test_sharpe_ci"]
            assert isinstance(ci, list) and len(ci) == 2
        assert "estimation ponctuelle" in body["caveat"]
        assert "statistiquement démontrée" in body["caveat"]

    def test_metrics_bundle_also_carries_the_test_window(self, client):
        body = client.get("/metrics", params={"universe": "etf_2017"}).json()
        assert body["phase5_test_window"]["regime_conditional"]["test_sharpe_ci"]
