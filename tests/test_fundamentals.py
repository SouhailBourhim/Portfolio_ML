"""
test_fundamentals.py — Locks in the causal guarantees of the fundamentals layer.

The load-bearing tests here aren't about the scraping (that's just parsing);
they're about the point-in-time seam. A subtle off-by-one in
`apply_publication_lag` or `build_point_in_time_panel` would silently
reintroduce the exact lookahead this experiment exists to avoid, and the
symptom would be an inflated IC score with no visible bug. So the tests are
named after the RULES they enforce, matching the repo's convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fundamentals import (
    apply_publication_lag,
    build_point_in_time_panel,
    extract_financial_data,
    fetch_html,
    financial_data_to_frame,
    run_fundamentals_features,
)


# ─── Parsing ────────────────────────────────────────────────────────────────

_SAMPLE_HTML = (
    'noise before '
    'financialData:{'
    'datekey:["TTM","2024-12-31","2023-12-31","2022-12-31"],'
    'fiscalYear:["2024","2024","2023","2022"],'
    'pe:[12.5,13.4,15.0,-.5],'
    'pb:[void 0,2.1,2.0,1.9],'
    'roe:[.15,.14,.13,NaN]'
    '} more noise'
)


def test_parser_recovers_datekey_and_metrics_from_inline_js():
    d = extract_financial_data(_SAMPLE_HTML)
    assert d is not None
    assert d["datekey"] == ["TTM", "2024-12-31", "2023-12-31", "2022-12-31"]
    assert d["pe"] == [12.5, 13.4, 15.0, -0.5]
    assert d["pb"] == [None, 2.1, 2.0, 1.9]  # void 0 → null → None
    assert d["roe"] == [0.15, 0.14, 0.13, None]  # NaN → null → None


def test_parser_returns_none_when_the_block_is_missing():
    # A page without the block is exactly what we'll get for a delisted or
    # renamed ticker — the parser must NOT throw here (that's the caller's
    # decision to raise), it must return None.
    assert extract_financial_data("<html>no data</html>") is None


def test_parser_handles_leading_dot_and_negative_dot_literals():
    # Common in the site's HTML: -.00049 (return growth), .5 (positive small).
    html = 'financialData:{datekey:["2024-12-31"],x:[-.00049],y:[.5]}'
    d = extract_financial_data(html)
    assert d["x"] == [-0.00049]
    assert d["y"] == [0.5]


def test_tidy_frame_drops_ttm_row_and_missing_fields():
    d = extract_financial_data(_SAMPLE_HTML)
    df = financial_data_to_frame(d, "IAM", ["pe", "pb", "roe", "not_in_page"])
    # 3 non-TTM periods × 3 present fields = 9 rows (roe's NaN survives as a row)
    assert set(df["metric"].unique()) == {"pe", "pb", "roe"}
    assert "not_in_page" not in df["metric"].unique()  # missing field silently absent
    assert (df["ticker"] == "IAM").all()
    assert "TTM" not in df["period_end"].astype(str).unique()


def test_tidy_frame_is_empty_when_datekey_missing_but_does_not_raise():
    df = financial_data_to_frame({"pe": [1.0]}, "IAM", ["pe"])
    assert df.empty
    assert list(df.columns) == ["period_end", "ticker", "metric", "value"]


# ─── Causal transform — the load-bearing block ─────────────────────────────

def _make_tidy() -> pd.DataFrame:
    # Two semi-annual reports for one ticker
    return pd.DataFrame({
        "period_end": pd.to_datetime(["2023-06-30", "2023-12-31"]),
        "ticker": ["IAM", "IAM"],
        "metric": ["pe", "pe"],
        "value": [15.0, 20.0],
    })


def test_publication_lag_shifts_period_end_forward_by_business_days():
    tidy = _make_tidy()
    out = apply_publication_lag(tidy, publication_lag_days=90)
    # 2023-06-30 + 90 bdays lands after ~4.5 months
    assert (out["available_from"] > out["period_end"]).all()
    # Zero-lag: BDay(0) is idempotent on a business day and rolls forward
    # to the next business day on a weekend period-end. That's correct — a
    # Sunday period-end is not "available" until Monday. So available_from
    # must be ≥ period_end always, never before.
    zero = apply_publication_lag(tidy, publication_lag_days=0)
    assert (zero["available_from"] >= zero["period_end"]).all()


def test_negative_publication_lag_raises_because_it_would_be_lookahead():
    # Structural, not a hyperparameter. -1 day would let the model see a
    # fundamental one day BEFORE its own period-end — impossible reality,
    # obvious leak.
    with pytest.raises(ValueError, match="publication_lag_days"):
        apply_publication_lag(_make_tidy(), publication_lag_days=-1)


def test_panel_is_nan_before_first_available_from_and_forward_fills_after():
    tidy = apply_publication_lag(_make_tidy(), publication_lag_days=90)
    dates = pd.bdate_range("2023-01-02", "2024-06-30")
    panel = build_point_in_time_panel(tidy, dates, ["IAM"])

    col = "IAM__FUND_pe"
    assert col in panel.columns

    first_af = tidy["available_from"].min()
    # Anywhere strictly before the first available_from → NaN.
    before = panel.loc[panel.index < first_af, col]
    assert before.isna().all()

    # On the first available_from and onward until the next report is available,
    # the value must be 15.0. Choose a date guaranteed to be inside that window.
    mid = first_af + pd.Timedelta(days=30)
    assert panel.loc[panel.index.asof(mid), col] == pytest.approx(15.0)

    # After the second report becomes available, the value flips to 20.0.
    second_af = tidy["available_from"].iloc[1]
    later = second_af + pd.Timedelta(days=30)
    assert panel.loc[panel.index.asof(later), col] == pytest.approx(20.0)


def test_a_query_at_date_t_never_returns_a_future_available_from_value():
    # The strongest form of the causal guarantee: for every t and every ticker,
    # the value on t equals the LAST report whose available_from ≤ t. This test
    # asserts that a report whose available_from is even ONE BUSINESS DAY after
    # t is invisible.
    tidy = apply_publication_lag(_make_tidy(), publication_lag_days=90)
    second_af = tidy["available_from"].iloc[1]
    # One business day BEFORE the second report is available: must still be 15.0.
    one_before = second_af - pd.tseries.offsets.BDay(1)
    dates = pd.bdate_range("2023-01-02", "2024-06-30")
    panel = build_point_in_time_panel(tidy, dates, ["IAM"])
    assert panel.loc[panel.index.asof(one_before), "IAM__FUND_pe"] == pytest.approx(15.0)


def test_future_value_corruption_cannot_change_any_past_panel_row():
    """
    The `test_phase3_integration.py`-style guarantee, but for fundamentals.
    Recompute the panel twice: once from the real tidy frame, once after
    corrupting the FUTURE (later-period) value. Every row on the panel
    whose date is < the corrupted period's available_from must be
    byte-identical between the two runs. This is the leakage gate.
    """
    tidy_a = apply_publication_lag(_make_tidy(), publication_lag_days=90)
    tidy_b = tidy_a.copy()
    # Corrupt the SECOND period's value only.
    tidy_b.loc[tidy_b["period_end"] == pd.Timestamp("2023-12-31"), "value"] = 999_999.0

    dates = pd.bdate_range("2023-01-02", "2024-12-31")
    panel_a = build_point_in_time_panel(tidy_a, dates, ["IAM"])
    panel_b = build_point_in_time_panel(tidy_b, dates, ["IAM"])

    second_af = tidy_a[tidy_a["period_end"] == pd.Timestamp("2023-12-31")][
        "available_from"
    ].iloc[0]

    past = panel_a.index < second_af
    # Every past row identical.
    pd.testing.assert_frame_equal(panel_a.loc[past], panel_b.loc[past])
    # And future rows genuinely CHANGED (proves the corruption wasn't a no-op).
    future_a = panel_a.loc[~past, "IAM__FUND_pe"]
    future_b = panel_b.loc[~past, "IAM__FUND_pe"]
    assert not future_a.equals(future_b)


def test_panel_column_naming_matches_ml_signals_convention():
    tidy = apply_publication_lag(_make_tidy(), publication_lag_days=90)
    dates = pd.bdate_range("2023-01-02", "2024-06-30")
    panel = build_point_in_time_panel(tidy, dates, ["IAM"])
    assert all("__" in c for c in panel.columns)
    assert panel.columns[0].startswith("IAM__FUND_")


def test_panel_missing_available_from_column_raises():
    # An easy mistake to make — pass tidy directly instead of the lagged frame.
    with pytest.raises(ValueError, match="available_from"):
        build_point_in_time_panel(_make_tidy(), pd.bdate_range("2023-01-01", periods=10), ["IAM"])


# ─── Offline scraping (fetches cache only) ─────────────────────────────────

def test_fetch_html_reuses_the_bronze_cache_without_network(tmp_path):
    """§15.5: Bronze data is immutable; a re-run must not hit the network.
    Prime the cache with a synthetic file, then call fetch_html and verify
    it returned exactly the cached content."""
    cache = tmp_path / "fundamentals"
    cache.mkdir()
    (cache / "IAM_income_semi.html").write_text("hello cached html", encoding="utf-8")
    got = fetch_html("IAM", "income_semi", cache_dir=cache)
    assert got == "hello cached html"


def test_fetch_html_rejects_unknown_page(tmp_path):
    with pytest.raises(ValueError, match="Unknown page"):
        fetch_html("IAM", "not_a_real_page", cache_dir=tmp_path)


# ─── End-to-end (offline via a hand-crafted cache) ─────────────────────────

def test_run_fundamentals_features_offline_produces_valid_gold_panel(tmp_path):
    """Full pipeline, offline: a cached HTML file whose content is a synthetic
    financialData block. Proves the wiring end-to-end without any HTTP call."""
    cache_dir = tmp_path / "bronze"
    cache_dir.mkdir()
    for t in ("A", "B"):
        (cache_dir / f"{t}_ratios_semi.html").write_text(_SAMPLE_HTML, encoding="utf-8")

    cfg = {
        "tickers": ["A", "B"],
        "ratio_fields": ["pe", "pb"],
        "publication_lag_days": 90,
        "cache_dir": "bronze",
        "output_path": "gold/fundamentals_features.parquet",
        "manifest_path": "gold/fundamentals_manifest.json",
    }
    idx = pd.bdate_range("2022-01-03", "2024-12-31")
    panel, manifest = run_fundamentals_features(cfg, idx, project_root=tmp_path)

    assert panel.shape[0] == len(idx)
    assert set(panel.columns) == {"A__FUND_pe", "A__FUND_pb", "B__FUND_pe", "B__FUND_pb"}
    md = manifest.to_dict()
    assert md["publication_lag_days"] == 90
    assert sorted(md["tickers"]) == ["A", "B"]
    # The end of the panel MUST hold the last-report values for each ticker
    # (or NaN if that ticker's report never became available, but our sample
    # has 2022-12-31 as the latest period + 90 bdays lag → well before end).
    assert np.isfinite(panel["A__FUND_pe"].iloc[-1])
