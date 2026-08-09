"""
test_currency.py — the base-currency (numéraire) conversion, as checks.

Every test here locks in one clause of the financial contract in
src/currency.py. Named after the rule, not the function (AGENTS.md §16), because
the suite is also the argument for why the rule exists.

The defect being closed: the pipeline used to sum MAD-denominated BVC returns
and USD-denominated ETF returns into one portfolio, justified by "returns are
unitless, so the arithmetic is valid". A portfolio P&L is a sum of money and
needs one numéraire; that justification was wrong, and nothing in the suite
could tell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from currency import (
    BASE_CURRENCY,
    FX_QUALITY_THRESHOLDS,
    FX_QUOTE_CONVENTION,
    FXDataUnavailable,
    FXQualitySuspect,
    align_fx_to_dates,
    convert_prices_to_base_currency,
    enforce_fx_quality,
    fx_quality_report,
    load_fx_rates,
    resolve_currency_policy,
    validate_fx_gap_structure,
    validate_fx_series,
)
from schemas import BVC_ASSETS, ETF_ASSETS


def _convert(prices, fx, **kwargs):
    """Conversion with the quality gate overridden, for the ARITHMETIC tests.

    Those fixtures use two-point, constant, or deliberately-corrupted FX by
    design — degenerate series that cannot and should not clear a production
    quality bar. The gate itself is exercised on its own terms in
    `TestTheFxQualityGateBlocksProduction`, which calls the real function.
    """
    kwargs.setdefault("allow_suspect_fx", True)
    return convert_prices_to_base_currency(prices, fx, **kwargs)


def _fx(values: list[float], start: str = "2023-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    idx.name = "Date"
    return pd.Series(values, index=idx, dtype="float64")


def _prices(assets: list[str], values: np.ndarray, start: str = "2023-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(values))
    idx.name = "Date"
    return pd.DataFrame(values, index=idx, columns=assets, dtype="float64")


class TestHandCalculatedConversion:
    """Requirement 1 — the arithmetic, checked against numbers a human worked
    out on paper, not against the implementation's own output."""

    def test_two_day_etf_example_gives_the_exact_mad_price_and_log_return(self):
        # SPY: 100 USD then 110 USD. USDMAD: 10.0 then 9.0 (MAD per USD).
        #   MAD price  day 1 = 100 * 10.0 = 1000.0
        #   MAD price  day 2 = 110 *  9.0 =  990.0
        #   MAD log-return   = ln(990 / 1000) = ln(0.99)
        # The dirham strengthened enough to turn a +10% USD gain into a -1% MAD
        # loss. That sign flip is the entire point of the correction.
        prices = _prices(["SPY"], np.array([[100.0], [110.0]]))
        fx = _fx([10.0, 9.0])

        converted, meta = _convert(
            prices, fx, foreign_assets=["SPY"], domestic_assets=[]
        )

        assert converted.loc[converted.index[0], "SPY"] == pytest.approx(1000.0, abs=1e-12)
        assert converted.loc[converted.index[1], "SPY"] == pytest.approx(990.0, abs=1e-12)

        mad_return = np.log(converted["SPY"] / converted["SPY"].shift(1)).iloc[1]
        assert mad_return == pytest.approx(np.log(0.99), abs=1e-12)
        assert mad_return < 0 < np.log(110 / 100), (
            "a positive USD return became a negative MAD return; if this ever "
            "passes with both the same sign the conversion is not being applied"
        )
        assert meta["base_currency"] == BASE_CURRENCY
        assert meta["fx_quote_convention"] == FX_QUOTE_CONVENTION

    def test_a_flat_exchange_rate_leaves_returns_untouched(self):
        """The control: if FX never moves, MAD returns must equal USD returns.
        Guards against a conversion that accidentally scales returns."""
        prices = _prices(["SPY"], np.array([[100.0], [110.0], [99.0]]))
        converted, _ = _convert(
            prices, _fx([8.5, 8.5, 8.5]), foreign_assets=["SPY"], domestic_assets=[]
        )
        usd = np.log(prices / prices.shift(1)).dropna()
        mad = np.log(converted / converted.shift(1)).dropna()
        pd.testing.assert_frame_equal(usd, mad)


class TestReturnDecompositionIdentity:
    """Requirement 2 — r_mad == r_usd + r_fx, EXACTLY.

    This is the property that makes the whole correction defensible: because
    the project computes log-returns (§15.1), conversion is additive with no
    cross-term. If it ever holds only approximately, someone has changed the
    return convention and every FX statement in the report becomes wrong.
    """

    def test_converted_log_returns_equal_usd_plus_fx_log_returns(self):
        rng = np.random.default_rng(11)
        n = 260
        usd_prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, size=(n, len(ETF_ASSETS))), axis=0))
        prices = _prices(ETF_ASSETS, usd_prices)
        fx_path = 9.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, size=n)))
        fx = _fx(list(fx_path))

        converted, _ = _convert(
            prices, fx, foreign_assets=ETF_ASSETS, domestic_assets=[]
        )

        r_usd = np.log(prices / prices.shift(1)).dropna()
        r_fx = np.log(fx / fx.shift(1)).dropna()
        r_mad = np.log(converted / converted.shift(1)).dropna()

        expected = r_usd.add(r_fx, axis=0)
        pd.testing.assert_frame_equal(r_mad, expected, atol=1e-12, rtol=0)

    def test_the_identity_survives_a_forward_filled_fx_date(self):
        """Where FX is stale, the identity must still hold against the STALE
        rate — i.e. the fill happens once, in the level, not twice."""
        prices = _prices(["SPY"], np.array([[100.0], [101.0], [102.0]]))
        # No FX observation on day 2; day 1's rate must carry forward.
        fx = pd.Series(
            [10.0, 12.0],
            index=pd.DatetimeIndex([prices.index[0], prices.index[2]], name="Date"),
        )
        converted, _ = _convert(
            prices, fx, foreign_assets=["SPY"], domestic_assets=[]
        )
        assert converted["SPY"].iloc[1] == pytest.approx(101.0 * 10.0)
        assert converted["SPY"].iloc[2] == pytest.approx(102.0 * 12.0)


class TestDomesticAssetsAreUntouched:
    """Requirement 3 — BVC price levels and returns unchanged bit-for-bit.

    This is what preserves the dividend total-return handling: those are MAD
    payments added to MAD prices, and they must never meet the FX rate.
    """

    def test_bvc_price_levels_are_bit_for_bit_identical(self):
        rng = np.random.default_rng(3)
        assets = BVC_ASSETS + ETF_ASSETS
        values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(80, len(assets))), axis=0))
        prices = _prices(assets, values)
        fx = _fx(list(9.0 + rng.normal(0, 0.05, size=80)))

        converted, meta = _convert(prices, fx)

        pd.testing.assert_frame_equal(
            converted[BVC_ASSETS], prices[BVC_ASSETS], check_exact=True
        )
        assert meta["assets_passed_through"] == BVC_ASSETS
        assert meta["n_assets_converted"] == len(ETF_ASSETS)
        # And the ETF side really did move, so the test above is not vacuous.
        assert not np.allclose(converted[ETF_ASSETS].values, prices[ETF_ASSETS].values)

    def test_an_asset_with_no_declared_currency_is_rejected(self):
        """The trap this module exists to close: the NEXT asset added to the
        universe must not silently inherit a numéraire nobody chose."""
        prices = _prices(["SPY", "MYSTERY.XX"], np.ones((5, 2)) * 100)
        with pytest.raises(ValueError, match="MYSTERY.XX"):
            convert_prices_to_base_currency(prices, _fx([9.0] * 5))


class TestFxFailuresAreLoud:
    """Requirement 5 — every way the FX input can be wrong must stop the run.

    The alternative is a matrix whose ETF columns are still USD while every
    label downstream says MAD. That output has the right shape, the right dates
    and plausible magnitudes, so nothing else in the pipeline would catch it.
    """

    @pytest.fixture()
    def prices(self):
        return _prices(["SPY"], np.linspace(100, 120, 20).reshape(-1, 1))

    def test_missing_fx_file_raises(self, tmp_path):
        with pytest.raises(FXDataUnavailable, match="not found"):
            load_fx_rates(tmp_path / "absent.parquet")

    def test_missing_fx_column_raises(self, tmp_path):
        path = tmp_path / "raw_bam_macro.parquet"
        pd.DataFrame(
            {"EURMAD": [10.0, 10.1]},
            index=pd.DatetimeIndex(["2023-01-02", "2023-01-03"], name="Date"),
        ).to_parquet(path)
        with pytest.raises(FXDataUnavailable, match="USDMAD"):
            load_fx_rates(path)

    def test_an_all_nan_fx_column_raises(self, tmp_path):
        path = tmp_path / "raw_bam_macro.parquet"
        pd.DataFrame(
            {"USDMAD": [np.nan, np.nan]},
            index=pd.DatetimeIndex(["2023-01-02", "2023-01-03"], name="Date"),
        ).to_parquet(path)
        with pytest.raises(FXDataUnavailable, match="no non-NaN"):
            load_fx_rates(path)

    def test_zero_rate_raises(self):
        with pytest.raises(FXDataUnavailable, match="non-positive"):
            validate_fx_series(_fx([9.0, 0.0, 9.1]))

    def test_negative_rate_raises(self):
        with pytest.raises(FXDataUnavailable, match="non-positive"):
            validate_fx_series(_fx([9.0, -8.5, 9.1]))

    def test_nan_inside_the_series_raises(self):
        with pytest.raises(FXDataUnavailable, match="NaN"):
            validate_fx_series(_fx([9.0, np.nan, 9.1]))

    def test_unsorted_index_raises_rather_than_being_silently_sorted(self):
        """Sorting here would hide the disorder from whoever introduced it, and
        an unsorted series forward-fills BACKWARDS — i.e. lookahead."""
        fx = _fx([9.0, 9.1, 9.2])
        shuffled = fx.iloc[[2, 0, 1]]
        with pytest.raises(FXDataUnavailable, match="not sorted"):
            validate_fx_series(shuffled)

    def test_duplicate_dates_raise(self):
        fx = _fx([9.0, 9.1])
        duped = pd.concat([fx, fx.iloc[[0]]]).sort_index()
        with pytest.raises(FXDataUnavailable, match="duplicate"):
            validate_fx_series(duped)

    def test_fx_starting_after_the_first_price_date_raises(self, prices):
        """Coverage that begins late CANNOT be repaired: filling it would take a
        rate from the future. Caught by the gap check, which fires before
        alignment and names the date to restart from."""
        late = _fx([9.0] * 5, start=str(prices.index[10].date()))
        with pytest.raises(FXDataUnavailable, match="wider than the causal"):
            _convert(prices, late, foreign_assets=["SPY"], domestic_assets=[])

    def test_fx_ending_early_beyond_the_fill_limit_raises(self, prices):
        short = _fx([9.0] * 3, start=str(prices.index[0].date()))
        with pytest.raises(FXDataUnavailable, match="wider than the causal"):
            _convert(
                prices, short, foreign_assets=["SPY"], domestic_assets=[], ffill_limit=5
            )

    def test_a_short_series_still_reports_the_uncoverable_window(self, prices):
        """Whichever check fires, the operator must learn WHICH dates are the
        problem — a bare refusal is not actionable."""
        short = _fx([9.0] * 3, start=str(prices.index[0].date()))
        with pytest.raises(FXDataUnavailable) as excinfo:
            _convert(prices, short, foreign_assets=["SPY"], domestic_assets=[])
        assert "Start the universe at" in str(excinfo.value)

    def test_an_interior_gap_within_the_limit_is_allowed(self, prices):
        """The complement of the test above — the limit must not be so strict
        that an ordinary FX holiday stops the pipeline."""
        fx = _fx([9.0] * len(prices), start=str(prices.index[0].date()))
        with_gap = fx.drop(fx.index[[5, 6]])
        converted, meta = _convert(
            prices, with_gap, foreign_assets=["SPY"], domestic_assets=[], ffill_limit=5
        )
        assert converted["SPY"].notna().all()
        assert meta["fx_dates_forward_filled"] == 2


class TestNoFutureFxCanReachThePast:
    """Requirement 6 — the causality gate.

    Mirrors tests/test_phase3_integration.py's future-corruption pattern: change
    the future, prove the past cannot move.
    """

    def test_corrupting_future_fx_cannot_change_any_past_converted_price(self):
        prices = _prices(["SPY"], np.linspace(100, 150, 60).reshape(-1, 1))
        fx = _fx([9.0 + 0.01 * i for i in range(60)], start=str(prices.index[0].date()))
        cutoff = prices.index[30]

        baseline, _ = _convert(
            prices, fx, foreign_assets=["SPY"], domestic_assets=[]
        )

        corrupted = fx.copy()
        corrupted.loc[corrupted.index > cutoff] *= 1000.0
        after, _ = _convert(
            prices, corrupted, foreign_assets=["SPY"], domestic_assets=[]
        )

        pd.testing.assert_frame_equal(
            baseline.loc[:cutoff], after.loc[:cutoff], check_exact=True
        )
        # Non-vacuity: the future MUST have moved, or this proves nothing.
        assert not np.allclose(
            baseline.loc[baseline.index > cutoff].values,
            after.loc[after.index > cutoff].values,
        )

    def test_a_gap_is_filled_from_the_past_not_the_future(self):
        """The direct test of fill direction. With a missing middle date and
        wildly different neighbours, the filled value identifies which way the
        fill ran."""
        prices = _prices(["SPY"], np.array([[100.0], [100.0], [100.0]]))
        fx = pd.Series(
            [2.0, 500.0],
            index=pd.DatetimeIndex([prices.index[0], prices.index[2]], name="Date"),
        )
        aligned = align_fx_to_dates(fx, prices.index, ffill_limit=5)
        assert aligned.iloc[1] == 2.0, "the middle date took tomorrow's rate — backfill"


class TestGapStructureDecidesUsability:
    """Density does not decide usability; the LONGEST CONSECUTIVE GAP does.

    A series can be 93% dense and perfectly usable, or 93% dense and unusable,
    depending entirely on whether the missing 7% is scattered or arrives as one
    hole. BAM's archive is the second case in July 2021 — 17 consecutive
    business days — which is why `full_2021` starts 2021-07-29 and not 07-01.
    """

    def test_scattered_gaps_within_the_limit_are_accepted(self):
        dates = pd.bdate_range("2023-01-02", periods=60)
        fx = _fx([9.0] * 60, start="2023-01-02").drop(dates[[5, 20, 21, 40]])
        stats = validate_fx_gap_structure(fx, dates, ffill_limit=5)
        assert stats["n_gaps_over_ffill_limit"] == 0
        assert stats["longest_consecutive_missing"] == 2

    def test_one_oversized_hole_is_rejected_however_dense_the_rest(self):
        dates = pd.bdate_range("2023-01-02", periods=120)
        fx = _fx([9.0] * 120, start="2023-01-02").drop(dates[30:47])   # 17-day hole
        with pytest.raises(FXDataUnavailable) as excinfo:
            validate_fx_gap_structure(fx, dates, ffill_limit=5)
        message = str(excinfo.value)
        assert "17 business days" in message
        assert "Start the universe at" in message, (
            "the error must name the date to restart from, not merely refuse"
        )

    def test_the_suggested_restart_date_clears_the_gap(self):
        """The remediation the message gives must actually work."""
        dates = pd.bdate_range("2023-01-02", periods=120)
        fx = _fx([9.0] * 120, start="2023-01-02").drop(dates[30:47])
        resume = dates[47]
        stats = validate_fx_gap_structure(fx, dates[dates >= resume], ffill_limit=5)
        assert stats["n_gaps_over_ffill_limit"] == 0

    def test_the_gap_statistics_travel_with_the_conversion(self):
        prices = _prices(["SPY"], np.full((60, 1), 100.0))
        fx = _fx([9.0 + 0.001 * i for i in range(60)])
        _, meta = _convert(prices, fx, foreign_assets=["SPY"], domestic_assets=[])
        assert "gap_structure" in meta["fx_quality"]


class TestPerUniverseCurrencyPolicy:
    """A universe's numéraire is decided by what it HOLDS, not by its filename.

    The agreed policy:
        full_2021 -> MAD, BAM mandatory, not covered by any fallback
        etf_2017  -> USD, no BAM dependency, unchanged bit-for-bit

    Deriving it from the columns is what makes it self-maintaining. Keyed on an
    `output_stem` instead, a BVC name added to the ETF universe would keep its
    USD label on a matrix that had quietly become mixed — the silent-drift class
    of §17.8.
    """

    def test_the_mixed_universe_requires_mad_and_mandatory_fx(self):
        policy = resolve_currency_policy(BVC_ASSETS + ETF_ASSETS)
        assert policy["base_currency"] == "MAD"
        assert policy["requires_conversion"] is True
        assert policy["requires_fx"] is True
        assert policy["is_mixed_currency"] is True

    def test_the_etf_only_universe_is_usd_with_no_fx_dependency(self):
        policy = resolve_currency_policy(ETF_ASSETS)
        assert policy["base_currency"] == "USD"
        assert policy["requires_conversion"] is False
        assert policy["requires_fx"] is False
        assert policy["is_mixed_currency"] is False

    def test_a_bvc_only_universe_is_already_mad(self):
        policy = resolve_currency_policy(BVC_ASSETS)
        assert policy["base_currency"] == "MAD"
        assert policy["requires_conversion"] is False
        assert policy["requires_fx"] is False

    def test_adding_one_bvc_name_to_the_etf_universe_flips_the_policy(self):
        """The self-maintaining property, stated as a test. This is the scenario
        the column-derived rule exists for."""
        before = resolve_currency_policy(ETF_ASSETS)
        after = resolve_currency_policy([*ETF_ASSETS, "IAM.CS"])
        assert before["requires_fx"] is False
        assert after["requires_fx"] is True
        assert after["base_currency"] == "MAD"

    def test_an_undeclared_asset_blocks_policy_resolution(self):
        with pytest.raises(ValueError, match="MYSTERY.XX"):
            resolve_currency_policy([*ETF_ASSETS, "MYSTERY.XX"])

    def test_an_empty_universe_is_rejected(self):
        with pytest.raises(ValueError, match="empty universe"):
            resolve_currency_policy([])

    def test_the_rationale_explains_itself_rather_than_asserting(self):
        """The policy is persisted into the Silver report, where a reader needs
        the reason and not just the verdict."""
        mixed = resolve_currency_policy(BVC_ASSETS + ETF_ASSETS)["rationale"]
        single = resolve_currency_policy(ETF_ASSETS)["rationale"]
        assert "one numéraire" in mixed or "numéraire" in mixed
        assert "MANDATORY" in mixed
        assert "REPORTING choice" in single, (
            "the USD universe's note must say that converting it would be a "
            "presentation decision, not a correctness fix"
        )


class TestTheFxQualityGateBlocksProduction:
    """The release gate. A defective FX feed must STOP a production run.

    Before this existed, the pipeline could diagnose its own input as bouncing,
    write `bounce_suspected: true` into the metadata, and ship the artifact
    anyway. A finding nobody is forced to act on is indistinguishable from no
    finding — the `require_dividends` lesson (§17.8), one input later.

    Each check below has its own test because they fail independently: a feed
    can be correctly-scaled but bouncing, or smooth but inverted.
    """

    def _clean_fx(self, n: int = 800, seed: int = 5) -> pd.Series:
        rng = np.random.default_rng(seed)
        return _fx(list(9.0 * np.exp(np.cumsum(rng.normal(0, 0.0028, size=n)))))

    def _prices(self, fx: pd.Series) -> pd.DataFrame:
        return _prices(["SPY"], np.full((len(fx), 1), 100.0), start=str(fx.index[0].date()))

    def test_a_clean_series_passes(self):
        fx = self._clean_fx()
        report = fx_quality_report(fx)
        assert report["passed"] is True, report["blocking_failures"]
        assert enforce_fx_quality(report)["override_applied"] is False

    def test_a_bouncing_quote_series_is_blocked(self):
        """Two contributors quoting ~3% apart on alternating days — the exact
        pattern the live Yahoo USDMAD=X feed shows."""
        fx = _fx([9.6 if i % 2 == 0 else 9.9 for i in range(400)])
        with pytest.raises(FXQualitySuspect, match="bounce"):
            convert_prices_to_base_currency(
                self._prices(fx), fx, foreign_assets=["SPY"], domestic_assets=[]
            )

    def test_free_float_volatility_is_blocked(self):
        """The dirham is a managed float in a narrow band. A series realising
        equity-like volatility is not this exchange rate."""
        rng = np.random.default_rng(2)
        fx = _fx(list(9.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=400)))))
        report = fx_quality_report(fx)
        assert any("volatility" in f for f in report["blocking_failures"])
        with pytest.raises(FXQualitySuspect, match="volatility"):
            enforce_fx_quality(report)

    def test_an_inverted_series_is_blocked_by_the_convention_check(self):
        """THE most valuable check here. USD per MAD (~0.109) instead of MAD per
        USD (~9.2) would silently DIVIDE every ETF price rather than multiply
        it. Every downstream number would be wrong, in the right shape, with
        plausible-looking dates — and nothing else in the pipeline would notice."""
        inverted = 1.0 / self._clean_fx()
        report = fx_quality_report(inverted)
        assert any("convention" in f for f in report["blocking_failures"])
        with pytest.raises(FXQualitySuspect, match="INVERTED"):
            enforce_fx_quality(report)

    def test_a_decimal_scale_error_is_blocked(self):
        report = fx_quality_report(self._clean_fx() * 10.0)
        assert any("convention" in f for f in report["blocking_failures"])

    def test_systematic_bad_prints_are_blocked_as_outliers(self):
        fx = self._clean_fx(n=400).copy()
        fx.iloc[::40] *= 0.6           # a bad print every 40 days
        report = fx_quality_report(fx)
        assert any("outlier" in f for f in report["blocking_failures"])

    def test_one_genuine_policy_event_is_tolerated(self):
        """The complement: a real band widening must NOT block the pipeline.
        An outlier RATE rather than a count is what buys this."""
        fx = self._clean_fx(n=3000).copy()
        fx.iloc[1500:] *= 1.06         # a single rebasing
        report = fx_quality_report(fx)
        assert not any("outlier" in f for f in report["blocking_failures"]), (
            report["blocking_failures"]
        )

    def test_a_mostly_forward_filled_series_is_blocked_on_density(self):
        """A sparse feed can clear the per-gap ffill limit on every individual
        gap and still not be an observed exchange rate."""
        fx = self._clean_fx(n=400)
        sparse = fx.iloc[::4]          # one observation every 4 business days
        with pytest.raises(FXQualitySuspect, match="coverage"):
            convert_prices_to_base_currency(
                self._prices(fx), sparse, foreign_assets=["SPY"],
                domestic_assets=[], ffill_limit=5,
            )

    def test_too_few_observations_to_judge_is_blocked(self):
        with pytest.raises(FXQualitySuspect, match="too few"):
            enforce_fx_quality(fx_quality_report(_fx([9.0, 9.1, 9.05])))

    def test_the_override_downgrades_to_a_warning_and_is_recorded(self, caplog):
        fx = _fx([9.6 if i % 2 == 0 else 9.9 for i in range(400)])
        report = fx_quality_report(fx)
        with caplog.at_level("WARNING"):
            resolved = enforce_fx_quality(report, allow_suspect=True)
        assert resolved["override_applied"] is True, (
            "an overridden run must be identifiable from its own metadata"
        )
        assert any("OVERRIDDEN" in r.getMessage() for r in caplog.records)
        assert any("not releasable" in r.getMessage() for r in caplog.records)

    def test_the_failure_message_points_at_the_source_not_the_threshold(self):
        fx = _fx([9.6 if i % 2 == 0 else 9.9 for i in range(400)])
        with pytest.raises(FXQualitySuspect) as excinfo:
            enforce_fx_quality(fx_quality_report(fx))
        message = str(excinfo.value)
        assert "Bank Al-Maghrib" in message
        assert "Fix the SOURCE" in message

    def test_thresholds_are_persisted_with_the_verdict(self):
        """A verdict is not auditable unless the bar it was judged against
        travels with it."""
        report = fx_quality_report(self._clean_fx())
        assert report["thresholds"] == FX_QUALITY_THRESHOLDS

    def test_no_production_caller_enables_the_override(self):
        """The gate is structurally un-bypassable, not merely un-bypassed.

        Source inspection, in the style of the existing no-hardcoded-Sharpe test
        in tests/test_run_dashboard_data.py: a reviewer cannot be relied on to
        notice `allow_suspect_fx=True` appearing in an unattended code path.
        """
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        production = [
            root / "src" / "clean.py",
            root / "src" / "pipeline.py",
            root / "src" / "features.py",
            root / "src" / "orchestration" / "assets.py",
        ]
        offenders = []
        for path in production:
            if not path.is_file():
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                for kw in node.keywords:
                    if kw.arg not in ("allow_suspect_fx", "allow_suspect"):
                        continue
                    # Passing the parameter THROUGH (a Name) is how silver_pipeline
                    # forwards its own argument and is fine. Hard-coding True is not.
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            f"production code enables the FX quality override at {offenders}. That "
            f"parameter is reserved for tests and synthetic smoke runs; setting it in "
            f"a pipeline path would let a run complete on a feed already diagnosed as "
            f"defective, which is the exact failure this gate exists to prevent."
        )


class TestDvcLineageDeclaresTheFxInput:
    """Requirement 8 — USD/MAD must be visible to the graph that claims to
    track what produced each number.

    This is the §17.8 lesson applied to a second input: the BVC dividend cache
    silently changed every BVC return for two days because it was a real input
    that no lineage graph listed. USD/MAD now changes every ETF return, so it
    must not repeat that. Declaring it on `clean` is sufficient AND necessary:
    every downstream stage reaches it through the Silver/Gold file chain, which
    these tests verify is actually intact rather than assumed.
    """

    @pytest.fixture()
    def stages(self):
        import yaml
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        return yaml.safe_load((root / "dvc.yaml").read_text())["stages"]

    def test_the_conversion_module_is_a_declared_dependency_of_clean(self, stages):
        assert "src/currency.py" in stages["clean"]["deps"], (
            "editing the conversion logic must invalidate the Silver layer"
        )

    def test_clean_produces_the_silver_matrices_that_carry_the_conversion(self, stages):
        outs = stages["clean"]["outs"]
        for artifact in (
            "data/silver/log_returns.parquet",
            "data/silver/log_returns_etf.parquet",
        ):
            assert artifact in outs

    def test_the_chain_from_clean_to_the_gold_matrices_is_unbroken(self, stages):
        """Without this, declaring the dep on `clean` would be necessary but not
        sufficient: the modelling stages read Gold, not Silver."""
        features = stages["features"]
        assert "data/silver/log_returns.parquet" in features["deps"]
        assert "data/silver/log_returns_etf.parquet" in features["deps"]
        assert "data/gold/log_returns.parquet" in features["outs"]
        assert "data/gold/log_returns_etf.parquet" in features["outs"]

    def test_the_official_fx_rate_is_a_tracked_bronze_output(self, stages):
        """Provenance parity with the BVC dividend cache (§17.8). As a bare dep
        it would be tracked but never pushed, so a fresh clone could reproduce
        every committed result and still be unable to re-run the pipeline that
        made them — the exact failure verified against the R2 remote in 2026-08."""
        outs = stages["fetch_bam_fx"]["outs"]
        names = [list(o)[0] if isinstance(o, dict) else o for o in outs]
        assert "data/bronze/bam_fx_reference.parquet" in names
        assert "data/bronze/bam_fx_reference_quality.json" in names, (
            "the quality control must be versioned alongside the data it describes"
        )

    def test_the_official_fx_output_persists_across_repro(self, stages):
        """`persist: true` matters more here than anywhere else in the graph:
        DVC deletes an output before re-running its stage, and the BAM gateway
        allows 5 requests/minute, so a non-persistent out would throw away 1,225
        rates and spend ~4.4 hours re-earning them."""
        for out in stages["fetch_bam_fx"]["outs"]:
            assert isinstance(out, dict), f"{out} must carry persist: true"
            assert list(out.values())[0].get("persist") is True

    def test_clean_depends_on_the_official_rate_not_the_yahoo_quote(self, stages):
        deps = stages["clean"]["deps"]
        assert "data/bronze/bam_fx_reference.parquet" in deps
        assert "data/bronze/raw_bam_macro.parquet" not in deps, (
            "the Yahoo USDMAD=X quote must no longer be an input to Silver: its "
            "daily changes correlate with the official rate's at 0.028 and it "
            "overstates FX volatility 6x. It survives for macro FEATURES only."
        )

    def test_the_yahoo_quote_still_feeds_the_macro_features(self, stages):
        """The other half: removing it from `clean` must not orphan it. It is
        still a legitimate input to the Phase 1/3 macro feature stages."""
        for stage in ("features", "ml_features"):
            assert "data/bronze/raw_bam_macro.parquet" in stages[stage]["deps"]

    def test_the_official_rate_is_in_the_release_manifest(self, stages):
        deps = stages["snapshot_manifest"]["deps"]
        assert "data/bronze/bam_fx_reference.parquet" in deps, (
            "a file that determines every full_2021 return belongs in the manifest "
            "beside the price and dividend inputs"
        )

    def test_the_gold_currency_manifest_is_a_tracked_output(self, stages):
        assert "data/gold/currency_manifest.json" in stages["features"]["outs"], (
            "the numéraire of the Gold matrices must be a versioned artifact, not "
            "a fact recoverable only from the Silver layer that modelling code "
            "never reads (§7)"
        )


class TestFxQualityIsMeasuredAndRecorded:
    """Not a gate — an audit record. The converted covariance matrix inherits
    the FX series' variance wholesale, so a defect in the quote feed becomes a
    defect in every risk number downstream, and it must at least be visible."""

    def test_a_clean_series_is_not_flagged(self):
        rng = np.random.default_rng(5)
        path = 9.0 * np.exp(np.cumsum(rng.normal(0, 0.003, size=600)))
        report = fx_quality_report(_fx(list(path)))
        assert report["bounce_suspected"] is False

    def test_an_alternating_quote_series_is_flagged_as_a_bounce(self):
        """Two contributors quoting ~3% apart on alternating days — the exact
        pattern the live USDMAD feed shows. It inflates measured volatility
        several-fold without any market movement."""
        values = [9.6 if i % 2 == 0 else 9.9 for i in range(400)]
        report = fx_quality_report(_fx(values))
        assert report["bounce_suspected"] is True
        assert report["lag1_autocorrelation"] < -0.25

    def test_the_report_travels_inside_the_conversion_metadata(self):
        prices = _prices(["SPY"], np.ones((40, 1)) * 100)
        _, meta = _convert(
            prices,
            _fx([9.6 if i % 2 == 0 else 9.9 for i in range(40)]),
            foreign_assets=["SPY"],
            domestic_assets=[],
        )
        assert meta["fx_quality"]["bounce_suspected"] is True
        assert meta["hedge_status"] == "unhedged"
        assert "no forward contract" in meta["hedging_note"]
