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

    (gold / "currency_manifest.json").write_text(json.dumps({
        "universes": {
            "full_2021": {
                "base_currency": "MAD", "converted": True,
                "hedge_status": "unhedged", "fx_series": "USDMAD",
            },
            "etf_2017": {
                "base_currency": "USD", "converted": False,
                "hedge_status": "not applicable — single-currency universe",
                "fx_series": None,
            },
        },
    }))

    # Minimal crisis artifact so the /crisis tests exercise the real endpoint
    # instead of skipping — a skipped test guards nothing.
    (gold / "crisis_windows.json").write_text(json.dumps({
        "crises": {"covid_2020": {"label": "COVID-19", "start": "2020-02-19",
                                  "end": "2020-03-23", "note": "-33.9%"}},
        "methodology": "external S&P peak-to-trough dates, fixed in advance",
        "universes": {"etf_2017": {"covid_2020": {
            "equal_weight": {"cum_return": -0.169, "max_drawdown": -0.191,
                             "recovery_days": 71, "n_days": 24, "worst_day": -0.06,
                             "ann_vol": 0.5, "partial_window": False},
            "min_variance_lw": {"cum_return": -0.135, "max_drawdown": -0.162,
                                "recovery_days": 37, "n_days": 24, "worst_day": -0.05,
                                "ann_vol": 0.4, "partial_window": False}}}},
        "regime_detection": {"etf_2017": {
            "bear_rate_in_crisis": 0.917, "bear_rate_outside": 0.292, "risk_ratio": 3.13,
            "n_crisis_rebalances": 36, "n_calm_rebalances": 212,
            "per_crisis": {"covid_2020": {"n_rebalances": 1, "n_bear": 1, "bear_rate": 1.0}},
            "significance": {"fisher_exact_p_liberal": 7.78e-13, "fisher_odds_ratio": 26.6,
                             "crises_exceeding_base_rate": "5/5",
                             "sign_test_p_conservative": 0.03125, "note": "lead with the sign test"},
        }},
    }))

    # A minimal, versioned explanation artifact. The API must serve this
    # artifact verbatim; it must never re-fit a strategy while handling a
    # request.
    (gold / "model_explanations.json").write_text(json.dumps({
        "universes": {
            "etf_2017": {
                "decision_date": "2023-03-24",
                "primary": {"decision": {"selected_regime": "bear"}},
                "challengers": {},
            },
            "full_2021": {
                "decision_date": "2023-03-24",
                "primary": {"decision": {"selected_regime": "bull"}},
                "challengers": {},
            },
        },
    }))

    monkeypatch.setattr(api_main, "GOLD", gold)
    # Caching moved down a layer: the loaders are plain functions now and the
    # cache lives on the (path, mtime)-keyed readers, so a regenerated artifact
    # is picked up without a restart. Tests still clear it so one test's
    # tmp_path cannot leak into the next.
    def _clear():
        api_main._read_json.cache_clear()
        api_main._read_parquet.cache_clear()

    _clear()
    yield TestClient(api_main.app)
    _clear()


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

    def test_version_labels_a_manifestless_tmp_snapshot_as_unverified(self, client):
        body = client.get("/version").json()
        assert body["api_version"] == "1.1.0"
        assert body["service_kind"] == "read_only_research_artifact_api"
        assert body["research_only"] is True
        assert body["order_execution_supported"] is False
        assert body["provenance"]["status"] == "unavailable"

    def test_version_exposes_manifest_identity_without_claiming_checksum_verification(self, client):
        import api.main as api_main

        (api_main.GOLD / "snapshot_manifest.json").write_text(json.dumps({
            "git_commit": "producing-commit",
            "generated_at_utc": "2026-08-03T00:00:00+00:00",
            "files": {"data/gold/dashboard_showcase.json": {"sha256": "abc"}},
        }))
        body = client.get("/version").json()
        assert body["provenance"] == {
            "status": "manifest_present",
            "manifest_path": "data/gold/snapshot_manifest.json",
            "producer_commit": "producing-commit",
            "generated_at_utc": "2026-08-03T00:00:00+00:00",
            "files_hashed": 1,
            "producer_tree_dirty": False,
            "note": (
                "Manifest metadata is present. Verify the release checksums with "
                "`python src/snapshot.py verify` before relying on this artifact."
            ),
        }

    def test_version_flags_a_manifest_written_from_a_dirty_tree(self, client):
        """A dirty manifest must not be served as if its commit were meaningful.

        `snapshot.py verify` rejects such a manifest outright, because
        `git_commit` names a revision whose tree does NOT contain the code that
        produced the artifacts. A client reading only `producer_commit` would
        have no way to tell, so the flag travels with the response.
        """
        import api.main as api_main

        (api_main.GOLD / "snapshot_manifest.json").write_text(json.dumps({
            "git_commit": "commit-plus-uncommitted-edits",
            "git_dirty": True,
            "git_dirty_paths": ["src/strategies.py"],
            "generated_at_utc": "2026-08-03T00:00:00+00:00",
            "files": {"data/gold/dashboard_showcase.json": {"sha256": "abc"}},
        }))
        provenance = client.get("/version").json()["provenance"]
        assert provenance["status"] == "manifest_dirty_tree"
        assert provenance["producer_tree_dirty"] is True
        assert "DIRTY" in provenance["note"]

    def test_every_declarable_provenance_status_is_reachable(self):
        """No dead values in the contract.

        An advertised status nothing can emit is worse than a missing one: a
        client branches on it, the branch is unreachable, and the contract
        implies a capability (checksum verification) the service does not have.
        """
        import inspect
        from typing import get_args

        from api.contracts import SnapshotProvenance
        import api.main as api_main

        declared = set(get_args(SnapshotProvenance.model_fields["status"].annotation))
        emitted = {
            literal
            for literal in declared
            if f'"{literal}"' in inspect.getsource(api_main._provenance)
        }
        assert declared == emitted, f"unreachable provenance status: {declared - emitted}"


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

    def test_published_allocation_has_a_typed_non_advisory_contract(self, client):
        body = client.get("/published-allocation", params={"universe": "full_2021"}).json()
        assert body["contract"] == "published_research_allocation/v1"
        assert body["strategy"] == "regime_conditional"
        assert body["weights"] == {"SPY": 0.6, "QQQ": 0.4}
        assert body["research_only"] is True
        assert body["order_execution_supported"] is False
        assert body["provenance"]["status"] == "unavailable"

    def test_explanation_is_served_from_the_published_artifact(self, client):
        body = client.get("/explanations/etf_2017").json()
        assert body["contract"] == "published_decision_explanation/v1"
        assert body["decision_date"] == "2023-03-24"
        assert body["explanation"]["primary"]["decision"]["selected_regime"] == "bear"
        assert "never refits" in body["caveat"]

    def test_model_card_is_available_for_the_primary_system(self, client):
        body = client.get("/model-card/regime_conditional").json()
        assert body["contract"] == "model_card/v1"
        assert "Model card" in body["markdown"]


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


class TestCrisisEndpoint:
    """The project's only statistically significant result, exposed to machines
    with its caveat attached — the same contract `/compare` honours for the
    Sharpe lift."""

    def test_crisis_returns_windows_and_detection(self, client):
        r = client.get("/crisis", params={"universe": "etf_2017"})
        if r.status_code == 404:
            pytest.skip("crisis_windows.json not generated in this environment.")
        assert r.status_code == 200
        body = r.json()
        assert body["per_crisis"], "no per-crisis statistics returned"
        assert body["regime_detection"], "regime-detection block missing"

    def test_the_crisis_diagnostic_ships_with_its_caveat(self, client):
        """A consumer must not take the diagnostic p-value as a production or
        economic-performance claim."""
        r = client.get("/crisis", params={"universe": "etf_2017"})
        if r.status_code == 404:
            pytest.skip("crisis_windows.json not generated in this environment.")
        body = r.json()
        sig = body["regime_detection"]["significance"]
        assert sig["sign_test_p_conservative"] is not None
        caveat = body["caveat"].lower()
        assert "contrainte" in caveat, "attribution caveat missing"
        assert "exploratoire" in caveat, "the exploratory-status caveat is missing"

    def test_unknown_universe_is_404(self, client):
        r = client.get("/crisis", params={"universe": "nope"})
        assert r.status_code == 404


class TestArtifactCacheIsNotStale:
    """Regression for the third stale-cache bug in this project (after the
    Dagster code server and Streamlit's module cache). `lru_cache(maxsize=1)`
    pinned artifacts for the process lifetime, so a regenerated Gold layer was
    served as the old numbers until someone restarted the API."""

    def test_regenerated_artifact_is_picked_up_without_restart(self, tmp_path, monkeypatch):
        import json as _json
        import api.main as m

        gold = tmp_path / "gold"
        gold.mkdir()
        path = gold / "dashboard_showcase.json"
        path.write_text(_json.dumps({"universes": {"u": {"marker": 1}}}))
        monkeypatch.setattr(m, "GOLD", gold)

        assert m._showcase()["universes"]["u"]["marker"] == 1

        # Rewrite with a distinct mtime, as a pipeline re-run would.
        import os
        path.write_text(_json.dumps({"universes": {"u": {"marker": 2}}}))
        os.utime(path, (path.stat().st_atime + 10, path.stat().st_mtime + 10))

        assert m._showcase()["universes"]["u"]["marker"] == 2, (
            "the API served a cached artifact after it was regenerated — the "
            "stale-cache bug this keying exists to prevent"
        )
