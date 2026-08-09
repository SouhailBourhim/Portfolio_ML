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

import json
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


def _write_synthetic_fx(bronze, start: str, n_rows: int) -> pd.Series:
    """Write a Bronze BAM macro file holding a USD/MAD series.

    silver_pipeline now REQUIRES this to express the USD-denominated ETF sleeve
    in MAD (src/currency.py), so every fixture that runs the pipeline must
    provide it. Written deliberately wider than the price window so the
    fixtures test the conversion rather than its coverage guard — the coverage
    failures have their own tests in tests/test_currency.py.
    """
    rng = np.random.default_rng(21)
    dates = pd.bdate_range(pd.Timestamp(start) - pd.offsets.BDay(20), periods=n_rows + 60)
    fx = 9.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0025, size=len(dates))))
    frame = pd.DataFrame({"USDMAD": fx}, index=dates)
    frame.index.name = "Date"
    # The OFFICIAL Bank Al-Maghrib artifact — the only FX source silver_pipeline
    # reads. raw_bam_macro.parquet (the Yahoo quote) is written too, because the
    # macro FEATURE stages still consume it, but it must never reach a portfolio.
    pq.write_table(pa.Table.from_pandas(frame), bronze / "bam_fx_reference.parquet")
    yahoo = pd.DataFrame({"USDMAD": fx * 1.01, "EURMAD": fx * 1.08}, index=dates)
    yahoo.index.name = "Date"
    pq.write_table(pa.Table.from_pandas(yahoo), bronze / "raw_bam_macro.parquet")
    return frame["USDMAD"]


class TestSilverPipelineIntegration:
    @pytest.fixture()
    def bronze_silver_dirs(self, tmp_path, monkeypatch):
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        bronze.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze)
        monkeypatch.setattr(clean, "SILVER_DIR", silver)
        _write_synthetic_fx(bronze, "2023-01-02", N_ROWS)
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
        _write_synthetic_fx(bronze, "2021-01-04", N_ROWS + 200)
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


class TestBaseCurrencyConversion:
    """The Silver layer expresses every asset in one numéraire (MAD).

    Before this, `log_returns.parquet` held MAD-denominated BVC returns beside
    USD-denominated ETF returns and every portfolio built from it summed two
    currencies. These tests run the real pipeline end-to-end, because the unit
    tests in tests/test_currency.py cannot see a wiring mistake.
    """

    @pytest.fixture()
    def bronze_full(self, tmp_path, monkeypatch):
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        bronze.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze)
        monkeypatch.setattr(clean, "SILVER_DIR", silver)
        etf = _make_synthetic_prices(ETF_ASSETS, "2023-01-02", N_ROWS)
        bvc = _make_synthetic_prices(BVC_ASSETS, "2023-01-02", N_ROWS)
        pq.write_table(pa.Table.from_pandas(etf), bronze / "raw_prices.parquet")
        pq.write_table(pa.Table.from_pandas(bvc), bronze / "bvc_prices.parquet")
        fx = _write_synthetic_fx(bronze, "2023-01-02", N_ROWS)
        return bronze, silver, fx

    def test_etf_returns_gain_exactly_the_fx_log_return(self, bronze_full):
        """r_mad - r_usd must equal log(fx_t / fx_{t-1}) — the same number for
        every ETF, since they share one exchange rate."""
        _, _, fx = bronze_full
        mad = clean.silver_pipeline(adjust_dividends=False, mixed_universe_start='2000-01-01')
        usd = clean.silver_pipeline(
            adjust_dividends=False, convert_to_mad=False, output_stem="log_returns_usd",
            mixed_universe_start='2000-01-01',
        )

        fx_returns = np.log(fx / fx.shift(1)).reindex(mad.index)
        for asset in ETF_ASSETS:
            delta = mad[asset] - usd[asset]
            pd.testing.assert_series_equal(
                delta, fx_returns, check_names=False, atol=1e-12, rtol=0
            )

    def test_bvc_returns_are_bit_for_bit_unchanged_by_conversion(self, bronze_full):
        """Requirement 3 at pipeline level: the MAD sleeve must not move at all."""
        mad = clean.silver_pipeline(adjust_dividends=False, mixed_universe_start='2000-01-01')
        usd = clean.silver_pipeline(
            adjust_dividends=False, convert_to_mad=False, output_stem="log_returns_usd",
            mixed_universe_start='2000-01-01',
        )
        pd.testing.assert_frame_equal(
            mad[BVC_ASSETS], usd[BVC_ASSETS], check_exact=True
        )

    def test_the_etf_only_universe_stays_in_usd_and_needs_no_fx(self, bronze_full):
        """The per-universe policy. `etf_2017` is five USD-denominated ETFs and
        nothing else: one numéraire, no mixed-currency defect, nothing to fix.

        This is not a convenience. That universe runs from 2004-11 and the only
        USD/MAD series obtainable for it starts around 2018, so making FX
        mandatory here would permanently block a universe that was never broken.
        Converting it to MAD would be a REPORTING choice about whose experience
        is described, not a correctness fix."""
        bronze, _, _ = bronze_full
        result = clean.silver_pipeline(include_bvc=False, output_stem="log_returns_etf")
        no_fx = clean.silver_pipeline(
            include_bvc=False, convert_to_mad=False, output_stem="log_returns_etf_b"
        )
        # The flag makes no difference, because the policy never asks for FX here.
        pd.testing.assert_frame_equal(result, no_fx, check_exact=True)

    def test_the_etf_universe_runs_with_no_fx_file_present_at_all(self, tmp_path, monkeypatch):
        """The dependency must be ABSENT, not merely satisfied. Deleting the FX
        source is the only way to prove `etf_2017` does not reach for it."""
        bronze, silver = tmp_path / "bronze", tmp_path / "silver"
        bronze.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze)
        monkeypatch.setattr(clean, "SILVER_DIR", silver)
        etf = _make_synthetic_prices(ETF_ASSETS, "2023-01-02", N_ROWS)
        pq.write_table(pa.Table.from_pandas(etf), bronze / "raw_prices.parquet")
        # No raw_bam_macro.parquet anywhere.

        result = clean.silver_pipeline(include_bvc=False, output_stem="log_returns_etf")

        assert list(result.columns) == ETF_ASSETS
        report = json.loads((silver / "validation_report_log_returns_etf.json").read_text())
        assert report["currency"]["base_currency"] == "USD"
        assert report["currency"]["converted"] is False
        assert report["currency"]["conversion_required"] is False
        assert report["currency"]["policy"]["requires_fx"] is False
        assert report["currency"]["policy"]["is_mixed_currency"] is False

    def test_the_etf_universe_is_bit_identical_with_and_without_fx_available(
        self, bronze_full, tmp_path, monkeypatch
    ):
        """Bit-for-bit, per the agreed policy: the presence of an FX file must
        not perturb a single value in the USD universe."""
        with_fx = clean.silver_pipeline(include_bvc=False, output_stem="log_returns_etf")

        bronze2, silver2 = tmp_path / "b2", tmp_path / "s2"
        bronze2.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze2)
        monkeypatch.setattr(clean, "SILVER_DIR", silver2)
        pq.write_table(
            pa.Table.from_pandas(_make_synthetic_prices(ETF_ASSETS, "2023-01-02", N_ROWS)),
            bronze2 / "raw_prices.parquet",
        )
        without_fx = clean.silver_pipeline(include_bvc=False, output_stem="log_returns_etf")

        pd.testing.assert_frame_equal(with_fx, without_fx, check_exact=True)

    def test_dividend_total_return_handling_survives_conversion(self, bronze_full, monkeypatch):
        """Requirement 7. The BVC total-return identity r = ln((P+D)/P_prev) must
        still hold EXACTLY with conversion enabled — those are MAD payments on
        MAD prices and must never meet the exchange rate."""
        bronze, _, _ = bronze_full
        prices = pd.read_parquet(bronze / "bvc_prices.parquet")
        ex_date = prices.index[100]
        amount = 3.25
        monkeypatch.setattr(
            clean, "load_bvc_dividends",
            lambda *a, **k: pd.DataFrame(
                [{"ex_date": ex_date, "ticker": "IAM.CS", "amount": amount, "kind": "cash"}]
            ),
        )

        with_div = clean.silver_pipeline(adjust_dividends=True, mixed_universe_start='2000-01-01')
        without = clean.silver_pipeline(
            adjust_dividends=False, output_stem="log_returns_nodiv",
            mixed_universe_start='2000-01-01',
        )

        p_t = prices.loc[ex_date, "IAM.CS"]
        p_prev = prices["IAM.CS"].shift(1).loc[ex_date]
        assert with_div.loc[ex_date, "IAM.CS"] == pytest.approx(
            np.log((p_t + amount) / p_prev), abs=1e-12
        )
        # And only the ex-date moved — the dividend did not leak elsewhere.
        other = with_div.index[with_div.index != ex_date]
        pd.testing.assert_series_equal(
            with_div.loc[other, "IAM.CS"], without.loc[other, "IAM.CS"], check_exact=True
        )

    def test_missing_fx_stops_the_run_and_leaves_no_silver_file(self, tmp_path, monkeypatch):
        """Requirement 5, at the level that matters. Degrading here would emit a
        matrix whose ETF columns are USD while every downstream label says MAD —
        right shape, right dates, plausible magnitudes, wrong currency."""
        bronze = tmp_path / "bronze"
        silver = tmp_path / "silver"
        bronze.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze)
        monkeypatch.setattr(clean, "SILVER_DIR", silver)
        prices = _make_synthetic_prices(ALL_ASSETS, "2023-01-02", N_ROWS)
        pq.write_table(pa.Table.from_pandas(prices), bronze / "raw_prices.parquet")
        # No raw_bam_macro.parquet written.

        with pytest.raises(clean.FXDataUnavailable, match="cannot be expressed"):
            clean.silver_pipeline(adjust_dividends=False, mixed_universe_start='2000-01-01')
        assert not (silver / "log_returns.parquet").exists()

    def test_the_validation_report_states_the_numeraire(self, bronze_full):
        """A returns matrix carries no unit. Without this block a reader cannot
        tell MAD from USD from a mixture — the state the project was in."""
        _, silver, _ = bronze_full
        clean.silver_pipeline(adjust_dividends=False, mixed_universe_start='2000-01-01')
        report = json.loads((silver / "validation_report.json").read_text())

        currency = report["currency"]
        assert currency["converted"] is True
        assert currency["base_currency"] == "MAD"
        assert currency["etf_source_currency"] == "USD"
        assert currency["fx_series"] == "USDMAD"
        assert currency["fx_quote_convention"] == "MAD per USD"
        assert currency["hedge_status"] == "unhedged"
        assert currency["n_assets_converted"] == len(ETF_ASSETS)
        assert set(currency["assets_converted"]) == set(ETF_ASSETS)
        assert currency["conversion_coverage"]["start"] and currency["conversion_coverage"]["end"]

    def test_an_unconverted_run_labels_itself_as_such(self, bronze_full):
        """`converted: false` is recorded, not omitted — an uncorrected artifact
        must identify itself rather than look like one nobody annotated."""
        _, silver, _ = bronze_full
        clean.silver_pipeline(
            adjust_dividends=False, convert_to_mad=False, output_stem="log_returns_usd",
            mixed_universe_start='2000-01-01',
        )
        report = json.loads((silver / "validation_report_log_returns_usd.json").read_text())
        assert report["currency"]["converted"] is False
        assert report["currency"]["base_currency"] is None


class TestMixedUniverseStartDate:
    """`full_2021` cannot begin before 2021-07-29.

    Bank Al-Maghrib's archive has exactly one gap wider than the causal
    forward-fill limit — 2021-07-06 to 2021-07-28, 17 business days — and those
    dates cannot be covered without inventing a rate. Costs 19 of 1,321 rows and
    leaves the OOS window (2022-07-01 onward) untouched.
    """

    @pytest.fixture()
    def bronze_full(self, tmp_path, monkeypatch):
        bronze, silver = tmp_path / "bronze", tmp_path / "silver"
        bronze.mkdir()
        monkeypatch.setattr(clean, "BRONZE_DIR", bronze)
        monkeypatch.setattr(clean, "SILVER_DIR", silver)
        # Deliberately oversized: the truncation below must still leave more than
        # the 500-row Pandera floor, or the test would fail on row count rather
        # than on the behaviour it is checking.
        rows = N_ROWS + 200
        for assets, name in ((ETF_ASSETS, "raw_prices"), (BVC_ASSETS, "bvc_prices")):
            pq.write_table(
                pa.Table.from_pandas(_make_synthetic_prices(assets, "2023-01-02", rows)),
                bronze / f"{name}.parquet",
            )
        _write_synthetic_fx(bronze, "2023-01-02", rows)
        return bronze, silver

    def test_rows_before_the_start_are_dropped_and_the_loss_is_logged(
        self, bronze_full, caplog
    ):
        """Silent truncation is a bug (§15.13). Dropping history must say how
        much and why."""
        cutoff = "2023-06-01"
        with caplog.at_level(logging.WARNING, logger="clean"):
            result = clean.silver_pipeline(
                adjust_dividends=False, mixed_universe_start=cutoff
            )
        assert result.index.min() >= pd.Timestamp(cutoff)
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "Dropping" in messages
        assert "2021-07-06" in messages, "the reason must name the actual gap"
        assert "OOS window" in messages

    def test_the_usd_universe_ignores_the_cutoff_entirely(self, bronze_full):
        """The cutoff exists because of an FX gap. `etf_2017` has no FX
        dependency, so it must keep every row regardless."""
        early = clean.silver_pipeline(
            include_bvc=False, output_stem="log_returns_etf",
            mixed_universe_start="2000-01-01",
        )
        late = clean.silver_pipeline(
            include_bvc=False, output_stem="log_returns_etf_b",
            mixed_universe_start="2023-06-01",
        )
        pd.testing.assert_frame_equal(early, late, check_exact=True)


class TestTheStartDateMatchesTheRealArchive:
    """The constant must stay derivable from the data it describes.

    Skipped when the Bronze artifact is absent (it is DVC-managed, so a fresh
    clone has none). When present, this recomputes the boundary rather than
    trusting `MIXED_UNIVERSE_START` — a refreshed BAM archive that moved the gap
    would otherwise silently invalidate a hard-coded date, which is precisely
    the drift class §17.1 exists to prevent.
    """

    def test_the_configured_start_is_the_first_date_clear_of_every_oversized_gap(self):
        from pathlib import Path

        from currency import MIXED_UNIVERSE_START, validate_fx_gap_structure

        path = (
            Path(__file__).resolve().parents[1]
            / "data" / "bronze" / "bam_fx_reference.parquet"
        )
        if not path.is_file():
            pytest.skip("bam_fx_reference.parquet not present — run `dvc pull`.")

        fx = pd.read_parquet(path)["USDMAD"]
        fx.index = pd.to_datetime(fx.index)
        start = pd.Timestamp(MIXED_UNIVERSE_START)

        # From the configured start onward, nothing may exceed the fill limit.
        stats = validate_fx_gap_structure(
            fx, pd.bdate_range(start, fx.index.max()), ffill_limit=5
        )
        assert stats["n_gaps_over_ffill_limit"] == 0

        # The universe must OPEN on a rate BAM actually published. Starting on a
        # missing date would technically pass the gap rule — the fill limit
        # counts consecutive dates in the target window, and a single missing
        # head date is within it — while silently valuing day one at a rate up
        # to 17 business days stale. That is the failure the whole correction
        # exists to prevent, so it is checked separately from the gap rule.
        assert start in fx.index, (
            f"{start.date()} is not an observed BAM publication date; the mixed "
            f"universe would open on an inherited rate."
        )

        # And it must sit immediately after the unfillable hole, not later —
        # every observed date discarded beyond that is history thrown away.
        prior = start - pd.offsets.BDay(1)
        assert prior not in fx.index, (
            f"{prior.date()} is observed, so the universe could start there; "
            f"{start.date()} discards usable history."
        )


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
        _write_synthetic_fx(bronze, "2023-01-02", N_ROWS)
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
