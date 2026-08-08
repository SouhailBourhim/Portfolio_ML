"""
test_bam_fx.py — the Bank Al-Maghrib reference-rate client, as checks.

Fully offline: `tests/conftest.py::_no_network` blocks outbound connections, so
every HTTP interaction here is a stub. That is deliberate — a test that reaches
bkam.ma would be a test that fails when Morocco has a public holiday.

The two rules worth the most here are the quote-convention normalisation
(`moyen / uniteDevise`) and the meaning of HTTP 204. Both are silent-corruption
routes: the first is a 100x error on some currencies, the second turns a
weekend into the end of the series.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import bam_fx
from bam_fx import (
    CREDENTIAL_ENV_VAR,
    SUBSCRIPTION_HEADER,
    BAMApiError,
    _normalise_record,
    fetch_reference_rate,
    fetch_reference_rate_series,
    has_publication_near,
)


class _Response:
    def __init__(self, status_code=200, payload=None, text="ok"):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    """Records every call so the tests can assert on headers and params."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        return self._responses.pop(0) if self._responses else _Response(204)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(CREDENTIAL_ENV_VAR, "test-key-not-a-real-credential")


def _rate(moyen, unit=1, currency="USD"):
    return _Response(200, [{"date": "2026-08-07", "libDevise": currency,
                            "moyen": moyen, "uniteDevise": unit}])


class TestQuoteConventionNormalisation:
    """`uniteDevise` is the trap. BAM publishes some currencies per 100 units,
    so `moyen` alone is the price of `uniteDevise` units, not of one."""

    def test_a_unit_of_one_passes_the_rate_through(self):
        assert _normalise_record(
            {"moyen": 9.3170, "uniteDevise": 1}, "USD"
        ) == pytest.approx(9.3170)

    def test_a_unit_of_one_hundred_is_divided(self):
        """JPY is quoted per 100. Without the division this is a 100x error that
        nothing downstream would catch."""
        assert _normalise_record(
            {"moyen": 6.31, "uniteDevise": 100}, "JPY"
        ) == pytest.approx(0.0631)

    def test_a_missing_unit_defaults_to_one(self):
        assert _normalise_record({"moyen": 9.3}, "USD") == pytest.approx(9.3)

    def test_an_implausible_usd_rate_raises_rather_than_returning(self):
        """If the convention ever flips to USD-per-MAD, the number arrives
        looking perfectly well-formed. The band check is what catches it."""
        with pytest.raises(BAMApiError, match="plausible band"):
            _normalise_record({"moyen": 0.107, "uniteDevise": 1}, "USD")

    def test_a_unit_error_on_usd_raises(self):
        with pytest.raises(BAMApiError, match="plausible band"):
            _normalise_record({"moyen": 9.32, "uniteDevise": 100}, "USD")

    def test_a_non_positive_unit_raises(self):
        with pytest.raises(BAMApiError, match="uniteDevise"):
            _normalise_record({"moyen": 9.32, "uniteDevise": 0}, "USD")


class TestNoPublicationIsNotAnError:
    """HTTP 204 means BAM published nothing that day — weekend, Moroccan public
    holiday, 25-26 December. Treating it as an error would abort a backfill on
    the first Saturday; treating it as a zero would be far worse."""

    def test_204_yields_none(self):
        session = _Session([_Response(204)])
        assert fetch_reference_rate(date(2026, 8, 8), session=session) is None

    def test_an_empty_payload_yields_none(self):
        session = _Session([_Response(200, [])])
        assert fetch_reference_rate(date(2026, 8, 8), session=session) is None

    def test_a_null_moyen_yields_none(self):
        session = _Session([_Response(200, [{"libDevise": "USD", "moyen": None}])])
        assert fetch_reference_rate(date(2026, 8, 8), session=session) is None

    def test_non_publication_days_are_skipped_not_filled_in_a_series(self):
        """The series must contain only OBSERVED rates. Projecting them onto a
        price calendar is currency.align_fx_to_dates' job, and it forward-fills
        causally with a bounded limit — doing it here would hide the gap."""
        session = _Session([_rate(9.30), _Response(204), _rate(9.32)])
        series = fetch_reference_rate_series(
            "2026-08-03", "2026-08-05", session=session, progress_every=0, pause_seconds=0
        )
        assert len(series) == 2
        assert series.isna().sum() == 0
        assert list(series.round(2)) == [9.30, 9.32]


class TestRequestConstruction:
    def test_the_key_travels_in_the_azure_apim_header(self):
        session = _Session([_rate(9.3)])
        fetch_reference_rate(date(2026, 8, 7), session=session)
        assert session.calls[0]["headers"][SUBSCRIPTION_HEADER] == "test-key-not-a-real-credential"

    def test_the_date_is_sent_iso_formatted_and_the_currency_labelled(self):
        session = _Session([_rate(9.3)])
        fetch_reference_rate(date(2026, 8, 7), currency="EUR", session=session)
        assert session.calls[0]["params"] == {"libDevise": "EUR", "date": "2026-08-07"}

    def test_the_series_walks_business_days_only(self):
        session = _Session([_rate(9.3)] * 10)
        fetch_reference_rate_series(
            "2026-08-03", "2026-08-09", session=session, progress_every=0, pause_seconds=0
        )
        requested = [c["params"]["date"] for c in session.calls]
        assert requested == ["2026-08-03", "2026-08-04", "2026-08-05",
                            "2026-08-06", "2026-08-07"]


class TestFailuresAreExplicit:
    def test_a_rejected_key_raises_without_echoing_the_credential(self):
        session = _Session([_Response(401)])
        with pytest.raises(BAMApiError) as excinfo:
            fetch_reference_rate(date(2026, 8, 7), session=session)
        message = str(excinfo.value)
        assert "Marché des changes" in message
        assert "test-key-not-a-real-credential" not in message, (
            "an error message must never echo the credential — it lands in logs"
        )

    def test_throttling_is_retried_then_surfaces(self, monkeypatch):
        monkeypatch.setattr(bam_fx.time, "sleep", lambda *_: None)
        session = _Session([_Response(429), _Response(429), _Response(429)])
        with pytest.raises(BAMApiError, match="after 3 attempts"):
            fetch_reference_rate(date(2026, 8, 7), session=session, max_retries=3)

    def test_a_transient_throttle_recovers(self, monkeypatch):
        monkeypatch.setattr(bam_fx.time, "sleep", lambda *_: None)
        session = _Session([_Response(429), _rate(9.31)])
        assert fetch_reference_rate(
            date(2026, 8, 7), session=session
        ) == pytest.approx(9.31)

    def test_a_missing_key_names_the_env_var_and_the_portal(self, monkeypatch):
        monkeypatch.delenv(CREDENTIAL_ENV_VAR, raising=False)
        with pytest.raises(BAMApiError, match=CREDENTIAL_ENV_VAR):
            fetch_reference_rate(date(2026, 8, 7))


class TestCoverageProbePredicate:
    """A single 204 cannot distinguish 'outside the served range' from 'public
    holiday'. The window predicate is what makes the binary search sound."""

    def test_a_window_of_holidays_reports_no_publication(self):
        assert has_publication_near(
            date(2005, 1, 3), window_days=3,
            session=_Session([_Response(204)] * 3), pause_seconds=0,
        ) is False

    def test_one_observation_anywhere_in_the_window_is_enough(self):
        session = _Session([_Response(204), _Response(204), _rate(8.7)])
        assert has_publication_near(
            date(2005, 1, 3), window_days=3, session=session, pause_seconds=0
        ) is True

    def test_it_stops_as_soon_as_it_finds_one(self):
        session = _Session([_rate(8.7)])
        has_publication_near(
            date(2005, 1, 3), window_days=12, session=session, pause_seconds=0
        )
        assert len(session.calls) == 1, "probing past a hit wastes the rate limit"


class TestCredentialHygiene:
    def test_no_key_literal_is_committed_in_this_module(self):
        """Mirrors tests/test_data_governance.py's rule, applied at the point a
        second credential entered the project."""
        source = (bam_fx.__file__)
        text = open(source, encoding="utf-8").read()
        assert "Ocp-Apim-Subscription-Key" in text          # the header name is fine
        for marker in ("baea705", "c2f49ad"):               # key prefixes must not appear
            assert marker not in text
