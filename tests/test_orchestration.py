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
    "bvc_dividends", "log_returns", "log_returns_etf", "gold_layer",
    "ml_features_layer",
}


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


def test_the_etf_universe_does_not_depend_on_bvc_dividends(defs):
    """ETFs arrive dividend-adjusted from yfinance. A spurious dependency would
    block the ETF universe on a scrape it does not need."""
    assert "bvc_dividends" not in _deps_of(defs, "log_returns_etf")
