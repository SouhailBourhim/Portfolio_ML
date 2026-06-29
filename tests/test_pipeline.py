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
