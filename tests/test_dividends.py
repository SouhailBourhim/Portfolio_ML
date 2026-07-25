"""
test_dividends.py — Locks in the BVC total-return correction.

The bug this guards against is not a crash; it is a SILENT understatement of
one half of the universe by 3.6-4.3%/yr, which inflated every optimizer's
measured edge over `equal_weight` (docs/DIVIDEND_BIAS.md). Nothing raised, no
test failed — the numbers were just quietly wrong. So the tests here assert
the arithmetic directly, on hand-computable fixtures, and name the rule they
enforce rather than the function they call.

All offline: the parser is tested against a captured HTML fragment, never a
live fetch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clean import compute_log_returns
from dividends import (
    BVC_ISSUER_CODES,
    dividend_yield_summary,
    parse_dividend_table,
)

# A faithful reduction of the real casablanca-bourse.com markup: the
# `emetteur_dividendes` anchor, the French column headers, comma decimals,
# and both an Ordinaire and an Exceptionnel row on the same ex-date (CIH
# genuinely does this — 2015-06-26 and 2018-06-27).
_SAMPLE_HTML = """
<div id="emetteur_dividendes"><div><p>Dividendes</p>
<table><thead><tr>
<th><p>Année</p></th><th><p>Montant Dividende</p></th>
<th><p>Type dividende</p></th><th><p>Date de détachement</p></th>
<th><p>Date de paiement</p></th></tr></thead><tbody>
<tr><td>2024</td><td>14,00</td><td>Ordinaire</td><td>09/07/2024</td><td>18/07/2024</td></tr>
<tr><td>2023</td><td>2,00</td><td>Exceptionnel</td><td>04/07/2023</td><td>13/07/2023</td></tr>
<tr><td>2023</td><td>14,00</td><td>Ordinaire</td><td>04/07/2023</td><td>13/07/2023</td></tr>
<tr><td>2022</td><td>1 250,50</td><td>Ordinaire</td><td>01/07/2022</td><td>10/07/2022</td></tr>
</tbody></table></div></div>
"""


class TestParsing:
    def test_parses_amount_ex_date_and_kind(self):
        df = parse_dividend_table(_SAMPLE_HTML, "CIH.CS")
        assert len(df) == 4
        assert set(df["ticker"]) == {"CIH.CS"}
        row = df[df["ex_date"] == pd.Timestamp("2024-07-09")].iloc[0]
        assert row["amount"] == pytest.approx(14.00)
        assert row["kind"] == "Ordinaire"

    def test_french_decimal_comma_is_not_read_as_a_thousands_separator(self):
        """'14,00' is fourteen, not fourteen hundred. Getting this wrong would
        inflate returns by 100x and is the classic locale failure."""
        df = parse_dividend_table(_SAMPLE_HTML, "CIH.CS")
        assert df["amount"].max() == pytest.approx(1250.50)   # '1 250,50'
        assert sorted(df["amount"])[:2] == [pytest.approx(2.0), pytest.approx(14.0)]

    def test_exceptional_dividends_are_kept_not_discarded(self):
        """A special dividend is real cash to a holder. Dropping it would
        under-report the return exactly like the original bug."""
        df = parse_dividend_table(_SAMPLE_HTML, "CIH.CS")
        same_day = df[df["ex_date"] == pd.Timestamp("2023-07-04")]
        assert len(same_day) == 2
        assert set(same_day["kind"]) == {"Ordinaire", "Exceptionnel"}
        assert same_day["amount"].sum() == pytest.approx(16.0)

    def test_missing_section_returns_empty_frame_without_raising(self):
        df = parse_dividend_table("<html>no dividends here</html>", "IAM.CS")
        assert df.empty
        assert list(df.columns) == ["ex_date", "ticker", "amount", "kind"]

    def test_every_production_bvc_ticker_has_an_issuer_code(self):
        assert set(BVC_ISSUER_CODES) == {"IAM.CS", "ATW.CS", "CIH.CS", "BCP.CS"}


class TestTotalReturnArithmetic:
    """The load-bearing block: does the correction compute what it claims?"""

    def _prices(self) -> pd.DataFrame:
        dates = pd.bdate_range("2024-07-05", periods=5, name="Date")  # 05,08,09,10,11
        return pd.DataFrame({"CIH.CS": [100.0, 100.0, 90.0, 90.0, 90.0],
                             "SPY": [400.0, 400.0, 400.0, 400.0, 400.0]}, index=dates)

    def _dividend(self, amount=10.0, ex="2024-07-09") -> pd.DataFrame:
        return pd.DataFrame({"ex_date": [pd.Timestamp(ex)], "ticker": ["CIH.CS"],
                             "amount": [amount], "kind": ["Ordinaire"]})

    def test_price_drop_exactly_offset_by_the_dividend_is_a_zero_return(self):
        """The whole point. A stock that goes 100 -> 90 while paying a 10
        dividend returned ZERO to its holder, not -10.5%. Price-only returns
        record the loss and discard the cash."""
        prices, div = self._prices(), self._dividend()
        total = compute_log_returns(prices, dividends=div)
        price_only = compute_log_returns(prices, dividends=None)
        ex_day = pd.Timestamp("2024-07-09")
        assert total.loc[ex_day, "CIH.CS"] == pytest.approx(0.0, abs=1e-12)
        assert price_only.loc[ex_day, "CIH.CS"] == pytest.approx(np.log(0.9))
        assert price_only.loc[ex_day, "CIH.CS"] < -0.10

    def test_assets_without_dividends_are_untouched(self):
        """ETFs arrive already dividend-adjusted; adjusting them again would
        double-count. Their column must be bit-identical either way."""
        prices, div = self._prices(), self._dividend()
        total = compute_log_returns(prices, dividends=div)
        price_only = compute_log_returns(prices, dividends=None)
        pd.testing.assert_series_equal(total["SPY"], price_only["SPY"])

    def test_non_ex_dates_are_unchanged_for_the_paying_asset_too(self):
        prices, div = self._prices(), self._dividend()
        total = compute_log_returns(prices, dividends=div)
        price_only = compute_log_returns(prices, dividends=None)
        other = total.index[total.index != pd.Timestamp("2024-07-09")]
        pd.testing.assert_series_equal(total.loc[other, "CIH.CS"],
                                       price_only.loc[other, "CIH.CS"])

    def test_none_dividends_reproduces_the_old_behaviour_exactly(self):
        """Backward compatibility: every pre-fix number must still be
        reproducible, so the correction can be A/B'd honestly."""
        prices = self._prices()
        expected = np.log(prices / prices.shift(1)).dropna()
        pd.testing.assert_frame_equal(compute_log_returns(prices, None), expected)
        pd.testing.assert_frame_equal(compute_log_returns(prices), expected)

    def test_dividend_raises_the_realised_total_return(self):
        """Directional sanity: adding cash cannot lower a holder's return."""
        prices, div = self._prices(), self._dividend()
        assert (compute_log_returns(prices, div)["CIH.CS"].sum()
                > compute_log_returns(prices, None)["CIH.CS"].sum())

    def test_ex_date_off_the_trading_calendar_snaps_forward_not_dropped(self):
        """2024-07-06 is a Saturday. The payment is real and must land on the
        next session, not vanish — silent loss is the bug this file exists for."""
        prices = self._prices()
        weekend = self._dividend(amount=10.0, ex="2024-07-06")
        total = compute_log_returns(prices, dividends=weekend)
        monday = pd.Timestamp("2024-07-08")
        assert total.loc[monday, "CIH.CS"] == pytest.approx(np.log(110.0 / 100.0))

    def test_ex_date_after_the_price_window_is_skipped_not_misapplied(self):
        prices = self._prices()
        future = self._dividend(amount=10.0, ex="2030-01-01")
        pd.testing.assert_frame_equal(compute_log_returns(prices, dividends=future),
                                      compute_log_returns(prices, dividends=None))

    def test_multiple_dividends_on_one_date_accumulate(self):
        prices = self._prices()
        two = pd.DataFrame({
            "ex_date": [pd.Timestamp("2024-07-09")] * 2,
            "ticker": ["CIH.CS"] * 2, "amount": [6.0, 4.0],
            "kind": ["Ordinaire", "Exceptionnel"],
        })
        total = compute_log_returns(prices, dividends=two)
        # 6 + 4 = 10 => same zero return as the single 10.0 case
        assert total.loc[pd.Timestamp("2024-07-09"), "CIH.CS"] == pytest.approx(0.0, abs=1e-12)


class TestYieldSummary:
    def test_reports_a_plausible_annual_yield(self):
        dates = pd.bdate_range("2024-01-01", periods=260, name="Date")
        prices = pd.DataFrame({"CIH.CS": np.full(len(dates), 100.0)}, index=dates)
        div = pd.DataFrame({"ex_date": [dates[100]], "ticker": ["CIH.CS"],
                            "amount": [4.0], "kind": ["Ordinaire"]})
        summary = dividend_yield_summary(prices, div)
        # 4.0 paid on a flat 100 price over ~1 year => ~4%
        assert summary.iloc[0]["annual_yield_pct"] == pytest.approx(4.0, rel=0.05)

    def test_ignores_tickers_absent_from_the_price_matrix(self):
        dates = pd.bdate_range("2024-01-01", periods=10, name="Date")
        prices = pd.DataFrame({"SPY": np.full(10, 400.0)}, index=dates)
        div = pd.DataFrame({"ex_date": [dates[5]], "ticker": ["CIH.CS"],
                            "amount": [4.0], "kind": ["Ordinaire"]})
        assert dividend_yield_summary(prices, div).empty
