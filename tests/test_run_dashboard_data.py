"""
test_run_dashboard_data.py — The dashboard's honesty gate.

Two classes of test here, and the second is the load-bearing one:

  1. SHAPE — the runner emits the artifacts the dashboard expects, with the
     right columns and no missing strategies. Ordinary smoke coverage.

  2. NUMBERS-MATCH-SOURCE — every headline the dashboard shows is DERIVED,
     never hardcoded, and derived correctly. Specifically: the "our ML system
     beats classical Markowitz by X%" claim must equal
     (best_ml_sharpe - best_classical_sharpe) / |best_classical_sharpe|,
     computed from the very metrics table the same run produced, and the
     "best classical" / "best ML" picks must actually be the argmax of their
     respective groups.

     This is CLAUDE.md §16 ("when claiming a fact in docs/notebooks, verify it
     against current data first") enforced in code rather than by discipline.
     The failure it prevents is concrete and has happened in this project
     before (§17.1, the stale phase2_hurdle.json): a number gets quoted once,
     the underlying data is refreshed, and the quoted number silently becomes
     a lie. A stakeholder-facing dashboard is the worst possible place for
     that, so the invariant is a test, not a convention.

All offline/synthetic (tmp_path Gold snapshot), never the real data/ tree.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import run_dashboard_data
from ml_features import build_ml_feature_set


def _write_gold_snapshot(tmp_path, n: int = 200, assets=("SPY", "QQQ", "GLD"), seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n, name="Date")
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(n, len(assets))), index=dates, columns=list(assets)
    )
    macro = pd.DataFrame({"VIX": 18 + np.cumsum(rng.normal(0, 0.3, n))}, index=dates)
    features = build_ml_feature_set(returns, macro, {
        "volatility_short_window": 10,
        "volatility_long_window": 20,
        "correlation_window": 20,
        "correlation_min_periods": 10,
        "macro_lag_days": 1,
    })

    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True)
    returns.to_parquet(gold / "log_returns_etf.parquet")
    returns.to_parquet(gold / "log_returns.parquet")
    features.to_parquet(gold / "ml_features_etf.parquet")
    features.to_parquet(gold / "ml_features_full.parquet")

    # Minimal phase5_results.json — the runner reads it for test-window CIs.
    (gold / "phase5_results.json").write_text(json.dumps({
        "etf_2017": {
            "test_start": "2020-08-01", "test_end": "2020-10-13",
            "baselines": {
                "regime_conditional": {"test_sharpe_net": 1.0, "test_sharpe_ci": [0.1, 1.9]},
                "equal_weight": {"test_sharpe_net": 0.8, "test_sharpe_ci": [-0.1, 1.7]},
            },
        },
        "full_2021": {
            "test_start": "2020-08-01", "test_end": "2020-10-13",
            "baselines": {
                "regime_conditional": {"test_sharpe_net": 1.1, "test_sharpe_ci": [0.2, 2.0]},
                "equal_weight": {"test_sharpe_net": 0.9, "test_sharpe_ci": [-0.05, 1.8]},
            },
        },
    }))


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
        "regime": {
            "n_states": 2,
            "covariance_type": "diag",
            "n_restarts": 2,
            "random_state_base": 0,
            "min_regime_train_days": 30,
            "bull_strategy": "max_sharpe",
            "bear_strategy": "min_variance_lw",
        },
        "phase5": {"results_path": "data/gold/phase5_results.json"},
    }


@pytest.fixture()
def dashboard_run(tmp_path, monkeypatch):
    """Run the real runner against a synthetic Gold snapshot; return its outputs."""
    _write_gold_snapshot(tmp_path)
    monkeypatch.setattr(run_dashboard_data, "ROOT", tmp_path)
    monkeypatch.setattr(run_dashboard_data, "load_params", lambda: _params())
    # run_phase4's loaders resolve paths against their OWN ROOT, so patch it too.
    import run_phase4
    monkeypatch.setattr(run_phase4, "ROOT", tmp_path)

    run_dashboard_data.main()

    gold = tmp_path / "data" / "gold"
    return {
        "equity": pd.read_parquet(gold / "dashboard_equity.parquet"),
        "weights": pd.read_parquet(gold / "dashboard_weights.parquet"),
        "showcase": json.loads((gold / "dashboard_showcase.json").read_text()),
        "gold": gold,
    }


class TestArtifactShape:
    def test_all_four_headline_strategies_present_on_both_universes(self, dashboard_run):
        equity = dashboard_run["equity"]
        for universe in ("etf_2017", "full_2021"):
            got = set(equity[equity["universe"] == universe]["strategy"].unique())
            assert got == set(run_dashboard_data.HEADLINE_STRATEGIES), (
                f"{universe} missing strategies: "
                f"{set(run_dashboard_data.HEADLINE_STRATEGIES) - got}"
            )

    def test_equity_frame_has_gross_and_net_columns(self, dashboard_run):
        assert {"Date", "universe", "strategy", "gross_return", "net_return"} <= set(
            dashboard_run["equity"].columns
        )

    def test_weights_frame_is_long_form_per_asset(self, dashboard_run):
        w = dashboard_run["weights"]
        assert {"Date", "universe", "strategy", "asset", "weight"} <= set(w.columns)
        # Weights on each (universe, strategy, Date) must sum to ~1 — the engine
        # already validates this, so a failure here means the RESHAPE lost rows.
        sums = w.groupby(["universe", "strategy", "Date"])["weight"].sum()
        assert np.allclose(sums.to_numpy(), 1.0, atol=1e-6)

    def test_showcase_records_every_universe_and_strategy(self, dashboard_run):
        universes = dashboard_run["showcase"]["universes"]
        assert set(universes) == {"etf_2017", "full_2021"}
        for u in universes.values():
            assert set(u["strategies"]) == set(run_dashboard_data.HEADLINE_STRATEGIES)

    def test_regime_timeline_written_for_regime_conditional(self, dashboard_run):
        path = dashboard_run["gold"] / "dashboard_regime.parquet"
        assert path.exists(), "regime timeline artifact missing"
        rg = pd.read_parquet(path)
        assert {"Date", "universe", "bull_prob", "bear_prob", "regime"} <= set(rg.columns)
        assert set(rg["regime"].unique()) <= {"bull", "bear"}
        # Posteriors are probabilities.
        assert ((rg["bull_prob"] >= 0) & (rg["bull_prob"] <= 1)).all()


class TestNumbersMatchSource:
    """The honesty gate. Every dashboard headline must be DERIVED from the
    metrics this same run produced — never hardcoded, never stale."""

    def test_headline_lift_equals_the_arithmetic_on_its_own_metrics(self, dashboard_run):
        for universe, u in dashboard_run["showcase"]["universes"].items():
            base = u["best_classical"]["sharpe_net"]
            top = u["best_ml"]["sharpe_net"]
            expected_pct = (top - base) / abs(base) * 100
            assert u["headline_lift_pct"] == pytest.approx(expected_pct, abs=0.01), (
                f"{universe}: headline_lift_pct is not the arithmetic on its own "
                f"best_classical/best_ml — the dashboard would show a fabricated number."
            )
            assert u["headline_lift_absolute_sharpe"] == pytest.approx(top - base, abs=1e-4)

    def test_best_classical_is_the_argmax_of_the_classical_group(self, dashboard_run):
        classical = ("equal_weight", "min_variance_lw", "max_sharpe")
        for universe, u in dashboard_run["showcase"]["universes"].items():
            sharpes = {s: u["strategies"][s]["sharpe_net"] for s in classical}
            expected = max(sharpes, key=sharpes.get)
            assert u["best_classical"]["name"] == expected, (
                f"{universe}: best_classical claims {u['best_classical']['name']} but "
                f"{expected} has the higher Sharpe ({sharpes})"
            )
            assert u["best_classical"]["sharpe_net"] == sharpes[expected]

    def test_best_ml_sharpe_matches_its_own_strategies_entry(self, dashboard_run):
        for universe, u in dashboard_run["showcase"]["universes"].items():
            name = u["best_ml"]["name"]
            assert u["best_ml"]["sharpe_net"] == u["strategies"][name]["sharpe_net"], (
                f"{universe}: best_ml.sharpe_net disagrees with strategies[{name}]"
            )

    def test_phase5_test_window_cis_are_copied_verbatim_from_the_source(self, dashboard_run,
                                                                        tmp_path):
        """The CI shown next to every headline must be the committed Phase 5
        number, not a recomputation — recomputing would silently diverge from
        the published result the supervisor already reviewed."""
        source = json.loads(
            (tmp_path / "data" / "gold" / "phase5_results.json").read_text()
        )
        for universe, u in dashboard_run["showcase"]["universes"].items():
            shown = u["phase5_test_window"]
            expected = source[universe]
            assert shown["test_start"] == expected["test_start"]
            assert shown["test_end"] == expected["test_end"]
            for strat in ("regime_conditional", "equal_weight"):
                assert shown[strat] == expected["baselines"][strat], (
                    f"{universe}/{strat}: the CI the dashboard shows is not the "
                    f"committed Phase 5 value."
                )

    def test_no_hardcoded_sharpe_literals_in_the_runner_source(self):
        """Guards the whole premise: if someone 'fixes' a number by typing it
        into the runner, this fails. Any float that looks like a plausible
        Sharpe (0.5-2.0, 2+ decimals) in an assignment is suspicious."""
        import inspect
        import re

        source = inspect.getsource(run_dashboard_data)
        # Strip docstrings/comments — prose legitimately cites numbers.
        code_lines = []
        in_docstring = False
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.count('"""') == 1:
                in_docstring = not in_docstring
                continue
            if in_docstring or stripped.startswith("#"):
                continue
            code_lines.append(line.split("#")[0])
        code = "\n".join(code_lines)

        suspicious = re.findall(r"=\s*(0\.[5-9]\d|1\.\d{2})\b", code)
        assert not suspicious, (
            f"Hardcoded Sharpe-like literals found in run_dashboard_data.py: "
            f"{suspicious}. Every number the dashboard shows must be computed "
            f"from Gold artifacts, never typed in."
        )


class TestPhase5Freshness:
    """The guard for the failure that actually happened (2026-07-25).

    The dividend correction regenerated every Gold return series while
    `phase5_results.json` stayed at its three-day-old run, so the dashboard
    rendered pre-correction confidence intervals directly beneath corrected
    Sharpe ratios — under a sentence claiming the latter had been validated on
    the former. `TestNumbersMatchSource` could not catch it: it verifies the
    CIs are copied FAITHFULLY, which stays true when the source is stale.
    Fidelity and freshness are different properties and need different tests.
    """

    def test_publishing_is_refused_when_phase5_predates_the_gold_data(self, tmp_path):
        _write_gold_snapshot(tmp_path)
        gold = tmp_path / "data" / "gold"
        phase5 = gold / "phase5_results.json"
        # Gold regenerated a day after the last Phase 5 run — the exact shape
        # of the real incident.
        old = phase5.stat().st_mtime - 86_400
        os.utime(phase5, (old, old))

        with pytest.raises(run_dashboard_data.StalePhase5Results) as excinfo:
            run_dashboard_data._assert_phase5_describes_current_gold(
                phase5, [gold / "log_returns.parquet", gold / "log_returns_etf.parquet"]
            )
        message = str(excinfo.value)
        assert "log_returns" in message, "error must name the file that outdates it"
        assert "phase5_compare" in message, "error must name the command that fixes it"

    def test_missing_phase5_results_is_refused_not_silently_skipped(self, tmp_path):
        _write_gold_snapshot(tmp_path)
        gold = tmp_path / "data" / "gold"
        phase5 = gold / "phase5_results.json"
        phase5.unlink()

        with pytest.raises(run_dashboard_data.StalePhase5Results):
            run_dashboard_data._assert_phase5_describes_current_gold(
                phase5, [gold / "log_returns.parquet"]
            )

    def test_artifacts_written_in_the_same_run_are_accepted(self, tmp_path):
        """A guard that fires on its own pipeline is worse than no guard —
        one `dvc repro` writes these seconds apart, in either order."""
        _write_gold_snapshot(tmp_path)
        gold = tmp_path / "data" / "gold"
        phase5 = gold / "phase5_results.json"
        # Phase 5 written a hair BEFORE the Gold parquet, same run.
        just_before = (gold / "log_returns.parquet").stat().st_mtime - 0.5
        os.utime(phase5, (just_before, just_before))

        run_dashboard_data._assert_phase5_describes_current_gold(
            phase5, [gold / "log_returns.parquet", gold / "log_returns_etf.parquet"]
        )

    def test_the_real_runner_refuses_to_publish_a_stale_dashboard(
        self, tmp_path, monkeypatch
    ):
        """End-to-end: the guard is actually wired into the publish path, not
        merely available as a helper nobody calls."""
        _write_gold_snapshot(tmp_path)
        gold = tmp_path / "data" / "gold"
        phase5 = gold / "phase5_results.json"
        old = phase5.stat().st_mtime - 86_400
        os.utime(phase5, (old, old))

        monkeypatch.setattr(run_dashboard_data, "ROOT", tmp_path)
        monkeypatch.setattr(run_dashboard_data, "load_params", lambda: _params())
        import run_phase4

        monkeypatch.setattr(run_phase4, "ROOT", tmp_path)

        with pytest.raises(run_dashboard_data.StalePhase5Results):
            run_dashboard_data.main()

        assert not (gold / "dashboard_showcase.json").exists(), (
            "a refused run must leave NO showcase behind — a half-written "
            "artifact would be published by the next reader as if it were valid"
        )
