"""
test_orchestration.py — The Dagster code location must LOAD.

Why this file exists. On 2026-07-27 the nightly scheduled job failed with
`ModuleNotFoundError: No module named 'dividends'` — `clean.py` imported it
lazily inside `silver_pipeline`, so nothing detected the problem until a run
was 30 seconds deep, and the failure sat unread in Dagster's run history while
the 9-asset Silver universe quietly stopped refreshing (CLAUDE.md §17.9).

Nothing in the suite imported the orchestration package at all, so 365 tests
passed against a pipeline that could not actually start. These tests close
that: they import the asset graph the way Dagster does and assert every asset
is wired, which is the same check `dagster definitions validate` performs —
now run automatically on every commit instead of by hand.

Offline: importing the module only builds definitions; no asset is executed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ORCH = Path(__file__).resolve().parents[1] / "src" / "orchestration"
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))


@pytest.fixture(scope="module")
def defs():
    import definitions

    return definitions.defs


EXPECTED_ASSETS = {
    "raw_etf_prices", "raw_fred_macro", "raw_bvc_prices", "raw_bam_macro",
    "bvc_dividends", "bam_fx_reference",
    "log_returns", "log_returns_etf", "gold_layer",
    "ml_features_layer",
    # global_2004 experiment — registered in the same change that creates its
    # Gold outputs (§15.10, §17.7), but excluded from the nightly job below.
    "raw_global_prices", "global_2004_data",
}

# The frozen experiment must NOT be on the daily schedule. Refreshing its data
# nightly would move the ground underneath a committed protocol, which is the
# opposite of what pre-registration is for.
EXPERIMENT_ASSETS = {"raw_global_prices", "global_2004_data"}


def test_the_code_location_imports_at_all():
    """The regression itself: a module the assets import must be importable in
    a fresh process, or every scheduled run dies partway through."""
    import assets  # noqa: F401


def _keys(defs) -> set[str]:
    return {k.to_user_string() for a in defs.assets for k in a.keys}


def _deps_of(defs, name: str) -> set[str]:
    for a in defs.assets:
        for key, upstream in a.asset_deps.items():
            if key.to_user_string() == name:
                return {u.to_user_string() for u in upstream}
    raise AssertionError(f"asset {name!r} not found in the graph")


def test_every_expected_asset_is_registered(defs):
    got = _keys(defs)
    assert got == EXPECTED_ASSETS, (
        f"asset graph drifted — missing {EXPECTED_ASSETS - got}, "
        f"unexpected {got - EXPECTED_ASSETS}. Every medallion output that "
        f"should refresh on the schedule must be a wired asset (§15.10)."
    )


def test_log_returns_depends_on_the_dividend_scrape(defs):
    """The 9-asset universe is WRONG without dividends (docs/DIVIDEND_BIAS.md),
    so the dependency must be in the graph, not merely in a docstring."""
    deps = _deps_of(defs, "log_returns")
    assert "bvc_dividends" in deps, (
        f"log_returns must depend on bvc_dividends; got {sorted(deps)}"
    )


def test_the_mixed_universe_depends_on_the_fx_source(defs):
    """USD/MAD converts the ETF sleeve of the 9-asset universe into the MAD
    numéraire before returns are computed (src/currency.py), so it is an input to
    the NUMBER, not merely a macro feature. §17.7 is exactly this: a Silver/Gold
    input missing from the asset graph goes stale silently on every scheduled
    run, and nobody notices until someone reads an mtime by accident."""
    assert "bam_fx_reference" in _deps_of(defs, "log_returns"), (
        f"log_returns must depend on bam_fx_reference; got "
        f"{sorted(_deps_of(defs, 'log_returns'))}"
    )


def test_the_usd_universe_does_not_depend_on_the_fx_source(defs):
    """The other half of the per-universe policy, and the half that must not
    regress. `etf_2017` is five USD-denominated ETFs: one numéraire, nothing to
    convert. It also starts in 2004-11, which no obtainable USD/MAD series
    covers — so a spurious FX edge here would block a universe that was never
    broken, on data that does not exist."""
    for fx_asset in ("raw_bam_macro", "bam_fx_reference"):
        assert fx_asset not in _deps_of(defs, "log_returns_etf"), (
            f"log_returns_etf must NOT depend on {fx_asset}; "
            f"got {sorted(_deps_of(defs, 'log_returns_etf'))}"
        )


def test_the_etf_universe_does_not_depend_on_bvc_dividends(defs):
    """ETFs arrive dividend-adjusted from yfinance. A spurious dependency would
    block the ETF universe on a scrape it does not need."""
    assert "bvc_dividends" not in _deps_of(defs, "log_returns_etf")


def test_the_frozen_experiment_is_not_on_the_daily_schedule(defs):
    """global_2004 is registered for lineage, but must never auto-refresh.

    The pre-registration is committed and timestamped; a nightly job that
    re-downloads its prices and rebuilds its Gold layer would silently change
    the data the protocol was frozen against. Registered-but-unscheduled is
    the correct state, and it is narrow enough to be worth asserting rather
    than trusting.
    """
    # Resolve the job: `define_asset_job` returns an UNRESOLVED definition, and
    # the distinction matters here — an unresolved job with no explicit
    # selection silently means "every registered asset", which is exactly how
    # registering the experiment for lineage would have put it on the schedule.
    job = defs.get_job_def("phase1_pipeline_job")
    selected = {k.to_user_string() for k in job.asset_layer.executable_asset_keys}

    assert selected, "Could not resolve the job's asset selection."
    leaked = EXPERIMENT_ASSETS & selected
    assert not leaked, (
        f"Frozen experiment assets are on the daily schedule: {sorted(leaked)}. "
        "Re-downloading their prices nightly would move the data underneath a "
        "committed, timestamped protocol."
    )
    # The other direction: the released pipeline must still be fully scheduled,
    # so restricting the selection cannot silently drop a real asset (§17.7).
    assert selected == EXPECTED_ASSETS - EXPERIMENT_ASSETS, (
        f"The released pipeline's schedule drifted: missing "
        f"{(EXPECTED_ASSETS - EXPERIMENT_ASSETS) - selected}, unexpected "
        f"{selected - (EXPECTED_ASSETS - EXPERIMENT_ASSETS)}."
    )
