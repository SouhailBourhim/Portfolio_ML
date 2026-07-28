"""
test_pipeline.py — End-to-end integration test for the Bronze -> Silver
transformation (silver_pipeline), plus targeted tests for merge_bvc_prices
and the late-start truncation warning in align_calendars.

Unlike the other test modules, this one does not call individual functions
in isolation — it runs silver_pipeline() against synthetic Bronze Parquet
files on disk, the same way pipeline.py invokes it in production. This is
the only place a broken file path, a changed function signature, or a
missing directory would surface before runtime.
"""

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import clean
from clean import align_calendars, merge_bvc_prices, silver_pipeline
from schemas import ALL_ASSETS, BVC_ASSETS, ETF_ASSETS

N_ROWS = 520  # comfortably above the 500-row Pandera minimum


def _make_synthetic_prices(assets: list[str], start: str, n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(start, periods=n_rows)
    returns = rng.normal(0.0003, 0.012, size=(n_rows, len(assets)))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=assets)
    df.index.name = "Date"
    return df


class TestSilverPipelineIntegration:
    @pytest.fixture()
    def bronze_silver_dirs(self, tmp_path, monkeypatch):
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        bronze.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze)
        monkeypatch.setattr(clean, "SILVER_DIR", silver)
        return bronze, silver

    def test_runs_end_to_end_and_writes_outputs(self, bronze_silver_dirs):
        bronze, silver = bronze_silver_dirs
        prices = _make_synthetic_prices(ALL_ASSETS, "2023-01-02", N_ROWS)
        pq.write_table(pa.Table.from_pandas(prices), bronze / "raw_prices.parquet")

        result = silver_pipeline()

        assert (silver / "log_returns.parquet").exists()
        assert (silver / "validation_report.json").exists()
        assert list(result.columns) == ALL_ASSETS
        assert len(result) >= 500
        assert result.isna().sum().sum() == 0

    def test_raises_clear_error_when_bronze_missing(self, bronze_silver_dirs):
        with pytest.raises(FileNotFoundError, match="Bronze prices not found"):
            silver_pipeline()

    def test_merges_bvc_prices_when_present(self, bronze_silver_dirs):
        bronze, silver = bronze_silver_dirs
        etf_prices = _make_synthetic_prices(ETF_ASSETS, "2023-01-02", N_ROWS)
        bvc_prices = _make_synthetic_prices(BVC_ASSETS, "2023-01-02", N_ROWS)
        pq.write_table(pa.Table.from_pandas(etf_prices), bronze / "raw_prices.parquet")
        pq.write_table(pa.Table.from_pandas(bvc_prices), bronze / "bvc_prices.parquet")

        result = silver_pipeline()

        assert set(ALL_ASSETS).issubset(set(result.columns))


class TestEtfOnlyUniverse:
    """The Phase 2 dual-universe data path: silver_pipeline(include_bvc=False)
    must keep the full pre-BVC ETF history and never touch the 9-asset files."""

    @pytest.fixture()
    def dual_universe_bronze(self, tmp_path, monkeypatch):
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        bronze.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze)
        monkeypatch.setattr(clean, "SILVER_DIR", silver)
        # ETFs start early; BVC starts 200 business days later (the real-world gap)
        etf_prices = _make_synthetic_prices(ETF_ASSETS, "2021-01-04", N_ROWS + 200)
        bvc_prices = _make_synthetic_prices(BVC_ASSETS, "2021-01-04", N_ROWS + 200).iloc[200:]
        pq.write_table(pa.Table.from_pandas(etf_prices), bronze / "raw_prices.parquet")
        pq.write_table(pa.Table.from_pandas(bvc_prices), bronze / "bvc_prices.parquet")
        return bronze, silver

    def test_etf_only_universe_keeps_pre_bvc_history(self, dual_universe_bronze):
        full = silver_pipeline()
        etf = silver_pipeline(include_bvc=False, output_stem="log_returns_etf")
        assert etf.index.min() < full.index.min()
        assert len(etf) > len(full)
        assert list(etf.columns) == ETF_ASSETS

    def test_etf_only_output_does_not_overwrite_full_universe_file(self, dual_universe_bronze):
        _, silver = dual_universe_bronze
        silver_pipeline()
        silver_pipeline(include_bvc=False, output_stem="log_returns_etf")
        full_on_disk = pd.read_parquet(silver / "log_returns.parquet")
        etf_on_disk = pd.read_parquet(silver / "log_returns_etf.parquet")
        assert set(BVC_ASSETS).issubset(full_on_disk.columns)   # 9-asset file intact
        assert set(etf_on_disk.columns) == set(ETF_ASSETS)
        assert (silver / "validation_report.json").exists()
        assert (silver / "validation_report_log_returns_etf.json").exists()

    def test_no_bvc_warning_when_etf_only_is_intentional(self, dual_universe_bronze):
        import warnings as warnings_mod
        with warnings_mod.catch_warnings():
            warnings_mod.simplefilter("error", UserWarning)
            silver_pipeline(include_bvc=False, output_stem="log_returns_etf")


class TestMergeBvcPrices:
    def test_passthrough_when_no_bvc_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clean, "BRONZE_DIR", tmp_path)
        etf_prices = _make_synthetic_prices(ETF_ASSETS, "2023-01-02", 50)
        merged = merge_bvc_prices(etf_prices)
        assert list(merged.columns) == ETF_ASSETS

    def test_adds_bvc_columns_when_file_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(clean, "BRONZE_DIR", tmp_path)
        etf_prices = _make_synthetic_prices(ETF_ASSETS, "2023-01-02", 50)
        bvc_prices = _make_synthetic_prices(BVC_ASSETS, "2023-01-02", 50)
        pq.write_table(pa.Table.from_pandas(bvc_prices), tmp_path / "bvc_prices.parquet")

        merged = merge_bvc_prices(etf_prices)
        assert set(BVC_ASSETS).issubset(set(merged.columns))


class TestLateStartTruncationWarning:
    def test_warns_when_a_column_starts_late(self, caplog):
        # SPY/QQQ start on day 0; IAM.CS only starts 30 business days later —
        # this is exactly the BVC-vs-ETF gap the warning exists to catch.
        early = _make_synthetic_prices(["SPY", "QQQ"], "2023-01-02", 100)
        late = _make_synthetic_prices(["IAM.CS"], "2023-01-02", 100).iloc[30:]
        prices = early.join(late, how="outer")

        with caplog.at_level(logging.WARNING, logger="clean"):
            align_calendars(prices)

        assert any("dropping" in rec.message for rec in caplog.records)
        assert any("IAM.CS" in rec.message for rec in caplog.records)

    def test_no_warning_when_all_columns_start_together(self, caplog):
        prices = _make_synthetic_prices(["SPY", "QQQ"], "2023-01-02", 100)
        with caplog.at_level(logging.WARNING, logger="clean"):
            align_calendars(prices)
        assert not any("dropping" in rec.message for rec in caplog.records)


class TestDividendRequirement:
    """`require_dividends` — the difference between a WARNING nobody reads and
    a stopped pipeline.

    The regression this guards is not hypothetical and not merely possible: on
    2026-07-27, hiding the Bronze dividend cache (which is gitignored, so a
    fresh clone HAS no cache) made `silver_pipeline()` fall through its broad
    `except Exception`, emit one WARNING, and write a silently PRICE-ONLY
    Silver layer — while the whole test suite stayed green. That is the exact
    ~3.0-4.3%/yr BVC understatement docs/DIVIDEND_BIAS.md was written to
    eliminate, reappearing under a passing build.
    """

    @pytest.fixture()
    def bronze_with_bvc(self, tmp_path, monkeypatch):
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        bronze.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze)
        monkeypatch.setattr(clean, "SILVER_DIR", silver)
        etf = _make_synthetic_prices(ETF_ASSETS, "2023-01-02", N_ROWS)
        bvc = _make_synthetic_prices(BVC_ASSETS, "2023-01-02", N_ROWS)
        pq.write_table(pa.Table.from_pandas(etf), bronze / "raw_prices.parquet")
        pq.write_table(pa.Table.from_pandas(bvc), bronze / "bvc_prices.parquet")
        return bronze, silver

    # Patch the name where clean.py BINDS it, not where dividends.py defines it:
    # clean imports it at module level (see that import's comment), so patching
    # the source module would leave clean's reference untouched and the test
    # would silently exercise the real scrape.
    def _break_scrape(self, monkeypatch, exc=ConnectionError("no route to host")):
        def _boom(*args, **kwargs):
            raise exc

        monkeypatch.setattr(clean, "load_bvc_dividends", _boom)

    def _empty_scrape(self, monkeypatch):
        monkeypatch.setattr(
            clean, "load_bvc_dividends",
            lambda *a, **k: pd.DataFrame(
                columns=["ex_date", "ticker", "amount", "kind"]
            ),
        )

    def test_unavailable_dividends_stop_a_run_that_required_them(
        self, bronze_with_bvc, monkeypatch
    ):
        self._break_scrape(monkeypatch)
        with pytest.raises(clean.DividendDataUnavailable, match="PRICE-ONLY"):
            clean.silver_pipeline(require_dividends=True)

    def test_an_empty_scrape_is_treated_as_failure_not_success(
        self, bronze_with_bvc, monkeypatch
    ):
        """A scrape that returns zero rows is the same corruption as one that
        raises — it just looks like success. It must not pass."""
        self._empty_scrape(monkeypatch)
        with pytest.raises(clean.DividendDataUnavailable):
            clean.silver_pipeline(require_dividends=True)

    def test_a_failed_run_writes_no_silver_file(self, bronze_with_bvc, monkeypatch):
        """Refusing loudly is only useful if it also refuses to leave a
        price-only artifact behind for the next reader to trust."""
        _, silver = bronze_with_bvc
        self._break_scrape(monkeypatch)
        with pytest.raises(clean.DividendDataUnavailable):
            clean.silver_pipeline(require_dividends=True)
        assert not (silver / "log_returns.parquet").exists()

    def test_attended_callers_still_degrade_with_a_warning(
        self, bronze_with_bvc, monkeypatch, caplog
    ):
        """The default stays permissive so offline/ad-hoc use keeps working —
        but it must SAY so, loudly enough to act on."""
        self._break_scrape(monkeypatch)
        with caplog.at_level("WARNING"):
            result = clean.silver_pipeline()
        assert not result.empty
        assert any(
            "PRICE-ONLY" in r.getMessage() for r in caplog.records
        ), "degrading silently is the bug; the warning is the whole point"

    def test_the_etf_only_universe_is_unaffected(self, bronze_with_bvc, monkeypatch):
        """ETFs arrive dividend-adjusted from yfinance, so the ETF-only
        universe must not be blocked by a BVC scrape it never needed."""
        self._break_scrape(monkeypatch)
        result = clean.silver_pipeline(
            include_bvc=False, output_stem="log_returns_etf", require_dividends=True
        )
        assert list(result.columns) == ETF_ASSETS
