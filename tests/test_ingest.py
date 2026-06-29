"""
test_ingest.py — Tests for the Bronze layer (ingest.py).

All network calls (yfinance, fredapi, BVCscrap) are mocked. These tests verify
the shape/contract of the ingestion functions, not live API behavior.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import ingest


def _fake_history(ticker: str, n: int = 10) -> pd.DataFrame:
    """Minimal yf.Ticker(...).history() return shape."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame({"Close": np.linspace(100, 110, n)}, index=dates)


class TestDownloadSingle:
    def test_returns_named_series_on_success(self, monkeypatch):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _fake_history("SPY")
        monkeypatch.setattr(ingest.yf, "Ticker", lambda t: mock_ticker)

        s = ingest._download_single("SPY", start="2020-01-01")
        assert s is not None
        assert s.name == "SPY"
        assert len(s) == 10

    def test_returns_none_after_exhausting_retries(self, monkeypatch):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        monkeypatch.setattr(ingest.yf, "Ticker", lambda t: mock_ticker)

        with patch("time.sleep", return_value=None):
            s = ingest._download_single("BAD.TICKER", start="2020-01-01", retries=2)
        assert s is None


class TestIngestPrices(object):
    def test_writes_bronze_parquet_and_returns_wide_frame(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)

        def fake_batch(tickers, start):
            dates = pd.bdate_range("2020-01-02", periods=5)
            return pd.DataFrame(
                {t: np.linspace(100, 105, 5) for t in tickers}, index=dates
            )

        monkeypatch.setattr(ingest, "_download_batch", fake_batch)

        prices = ingest.ingest_prices(start="2020-01-01")

        assert set(ingest.ALL_TICKERS).issubset(prices.columns)
        assert prices.index.name == "Date"
        assert (tmp_path / "raw_prices.parquet").exists()

    def test_raises_when_no_data_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)
        monkeypatch.setattr(ingest, "_download_batch", lambda tickers, start: pd.DataFrame())

        with pytest.raises(ValueError, match="no data"):
            ingest.ingest_prices(start="2020-01-01")


class TestIngestMacro:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
            ingest.ingest_macro(start="2020-01-01")

    def test_writes_bronze_parquet_with_mocked_fred(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)
        monkeypatch.setenv("FRED_API_KEY", "fake-key")

        dates = pd.bdate_range("2020-01-02", periods=5)
        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = pd.Series(
            np.linspace(20, 25, 5), index=dates
        )
        monkeypatch.setattr(ingest, "Fred", lambda api_key: mock_fred_instance)

        macro = ingest.ingest_macro(start="2020-01-01")

        assert list(macro.columns) == list(ingest.FRED_SERIES.keys())
        assert macro.index.name == "Date"
        assert (tmp_path / "raw_macro.parquet").exists()

    def test_raises_when_all_series_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)
        monkeypatch.setenv("FRED_API_KEY", "fake-key")

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.side_effect = RuntimeError("API down")
        monkeypatch.setattr(ingest, "Fred", lambda api_key: mock_fred_instance)

        with pytest.raises(RuntimeError, match="No macro series"):
            ingest.ingest_macro(start="2020-01-01")


class TestIngestBvc:
    def test_raises_without_bvcscrap_installed(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "BVCscrap":
                raise ImportError("No module named 'BVCscrap'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="BVCscrap not installed"):
            ingest.ingest_bvc()

    def test_raises_on_empty_scrape(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)

        fake_bvc_module = MagicMock()
        fake_bvc_module.loadmany.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"BVCscrap": fake_bvc_module}):
            with pytest.raises(RuntimeError, match="no data"):
                ingest.ingest_bvc()

    def test_writes_bronze_parquet_with_mocked_scrape(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)

        raw = pd.DataFrame(
            {
                "Maroc Telecom": [150.0, 151.0],
                "Attijariwafa": [450.0, 452.0],
                "CIH": [300.0, 301.0],
                "BCP": [250.0, 251.0],
            },
            index=["24/06/2021", "25/06/2021"],
        )
        fake_bvc_module = MagicMock()
        fake_bvc_module.loadmany.return_value = raw

        with patch.dict("sys.modules", {"BVCscrap": fake_bvc_module}):
            result = ingest.ingest_bvc()

        assert set(result.columns) == set(ingest.BVC_NAME_MAP.values())
        assert result.index.name == "Date"
        assert result.index.is_monotonic_increasing
        assert (tmp_path / "bvc_prices.parquet").exists()


class TestIngestBamMacro:
    def test_writes_bronze_parquet_with_mocked_fx(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)

        dates = pd.bdate_range("2020-01-02", periods=5)
        cols = pd.MultiIndex.from_product([["Close"], ["USDMAD=X", "EURMAD=X"]])
        fake_fx = pd.DataFrame(
            np.column_stack([np.linspace(10, 10.5, 5), np.linspace(11, 11.5, 5)]),
            index=dates,
            columns=cols,
        )
        monkeypatch.setattr(ingest.yf, "download", lambda *a, **kw: fake_fx)

        bam = ingest.ingest_bam_macro(start="2020-01-01")

        assert "USDMAD" in bam.columns
        assert "EURMAD" in bam.columns
        assert "TAUX_DIR" in bam.columns
        assert bam.index.name == "Date"
        assert (tmp_path / "raw_bam_macro.parquet").exists()

    def test_taux_directeur_is_step_function(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "BRONZE_DIR", tmp_path)

        dates = pd.bdate_range("2020-01-02", periods=5)
        cols = pd.MultiIndex.from_product([["Close"], ["USDMAD=X", "EURMAD=X"]])
        fake_fx = pd.DataFrame(
            np.column_stack([np.linspace(10, 10.5, 5), np.linspace(11, 11.5, 5)]),
            index=dates,
            columns=cols,
        )
        monkeypatch.setattr(ingest.yf, "download", lambda *a, **kw: fake_fx)

        bam = ingest.ingest_bam_macro(start="2020-01-01")
        # The 5 fx dates all fall before the first post-baseline decision (2020-03-17)
        first_five = bam.loc[dates, "TAUX_DIR"]
        assert first_five.nunique() == 1
        assert first_five.iloc[0] == 2.25
