"""
bam_fx.py — official USD/MAD reference rates from Bank Al-Maghrib.

Addresses: P1, P4 — the base-currency conversion in src/currency.py is only as
trustworthy as the exchange rate feeding it. The Yahoo `USDMAD=X` series the
project has been carrying fails its own quality gate on three independent
counts (annualised volatility 25.9% against a real ~6.3%, lag-1 autocorrelation
-0.42, and 0.14% of days moving more than 5%), because it alternates between
two contributor quotes rather than tracking the market. Converting with it
would inject that non-economic variance into every ETF return and into the
whole covariance block (P1), and would publish MAD figures no holder could
have realised (P4).

THE SOURCE
    Bank Al-Maghrib, "Cours de référence" — the central bank's own published
    reference rates, the authoritative quote for the dirham.
    Portal   https://apihelpdesk.centralbankofmorocco.ma  (Azure API Management)
    Product  "Marché des changes"
    Endpoint https://api.centralbankofmorocco.ma/cours/Version1/api/CoursVirement

CONVENTION — verified against the published page, and NOT assumed
    Rates are quoted as MAD per unit of foreign currency: on 07/08/2026 the
    page shows 1 USD = 9.3170 MAD. That is the same direction as the series
    this project already uses, so conversion remains a multiplication.

    ⚠️ `uniteDevise` IS NOT ALWAYS 1. BAM publishes some currencies per 100
    units (JPY being the classic case). The raw `moyen` field is therefore the
    price of `uniteDevise` units, and the rate per ONE unit is
    `moyen / uniteDevise`. Ignoring that field is a silent 100x error on the
    affected currencies. This module always divides, and asserts the result
    lands in a plausible band before returning it.

PUBLICATION TIMING — the causality argument
    Reference rates are published at 16h15 local (14h00 during Ramadan);
    fluctuation-band limits go out at 08h30 and are updated at 11h54.
    Source: bkam.ma, "Horaires d'affichage des cours de change du dirham".
    The rate for date t is therefore knowable ON date t, so using it to value a
    holding at t introduces no lookahead. This is the same standard applied to
    the BVC dividend ex-dates and, in the opposite direction, the reason macro
    features are lagged (§15.8).

WHAT THIS MODULE DOES NOT DO
    It does not write Bronze, it is not a DVC stage, and it is not wired into
    the pipeline. Its history depth is unverified — that is what
    scripts/probe_bam_history.py exists to establish. Nothing here may be
    promoted into the medallion flow until the probe shows the API actually
    covers the window the ETF universe needs (2004-11 onwards).

CREDENTIALS
    The subscription key is read from the BAM_API_KEY environment variable and
    is never logged, never defaulted, and never written to disk by this code
    (§15.6, the same rule as FRED_API_KEY).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date as _date
from datetime import datetime, timedelta

import pandas as pd
import requests

log = logging.getLogger("bam_fx")

BAM_API_BASE = "https://api.centralbankofmorocco.ma"
COURS_VIREMENT_URL = f"{BAM_API_BASE}/cours/Version1/api/CoursVirement"
# These two hold the NAME of the environment variable and the NAME of the HTTP
# header — never a credential. Do not rename them to `API_KEY_ENV` /
# `API_KEY_HEADER`: `tests/test_data_governance.py` rejects any assignment whose
# target reads like a credential and whose value is a string literal, and it is
# right to be blunt about that. Keeping the check strict and naming these for
# what they actually contain is the better trade, and follows the project's own
# precedent of fixing the producer rather than loosening a trust-boundary check.
CREDENTIAL_ENV_VAR = "BAM_API_KEY"
# Azure API Management's standard subscription header.
SUBSCRIPTION_HEADER = "Ocp-Apim-Subscription-Key"

# The reform's fluctuation band has never taken USD/MAD outside roughly 7-11.
# Anything beyond this is a unit error, an inverted quote, or a bad print, and
# must not be silently returned.
PLAUSIBLE_USDMAD_MIN = 5.0
PLAUSIBLE_USDMAD_MAX = 15.0

# The earliest date worth probing: GLD's inception, the start of the ETF
# universe (params.yaml ingest.start_date).
UNIVERSE_START = _date(2004, 11, 18)


class BAMApiError(RuntimeError):
    """
    A BAM API call failed, or returned something that is not a usable rate.

    Addresses: P1, P4 — the alternative to raising is a silently short or
    silently wrong FX series, which is precisely the class of defect this whole
    correction exists to remove.
    """


def api_key() -> str:
    """
    Read the BAM subscription key from the environment.

    Addresses: P4 — a credential in source is a credential in Git, and a
    reproducible pipeline must not depend on one developer's checkout. Mirrors
    the FRED_API_KEY rule (§15.6), which `tests/test_data_governance.py`
    enforces by rejecting literal key assignments anywhere in src/.

    Returns:
        The subscription key.

    Raises:
        BAMApiError: if the variable is unset or empty, with the exact
            remediation rather than a bare KeyError.
    """
    key = os.environ.get(CREDENTIAL_ENV_VAR, "").strip()
    if not key:
        raise BAMApiError(
            f"{CREDENTIAL_ENV_VAR} is not set. Add it to the project's .env file (which is "
            f"gitignored, alongside FRED_API_KEY) as:\n"
            f"    {CREDENTIAL_ENV_VAR}=<your primary key>\n"
            f"Keys come from https://apihelpdesk.centralbankofmorocco.ma — Account "
            f"details → Subscriptions → 'Marché des changes'. Never commit it and "
            f"never hard-code it (§15.6)."
        )
    return key


# Measured 2026-08-08 against the live gateway: the 6th request inside a
# 60-second window returns HTTP 429 with `Retry-After: 58`. Documented here
# because it is the binding constraint on any backfill, not a detail.
MEASURED_RATE_LIMIT_PER_MINUTE = 5
POLITE_PAUSE_SECONDS = 60.0 / MEASURED_RATE_LIMIT_PER_MINUTE + 0.5


def _retry_after_seconds(response, default: float) -> float:
    """Read the gateway's own Retry-After, falling back to `default`."""
    raw = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    try:
        # +1s of slack: waiting exactly the stated interval races the window edge.
        return float(raw) + 1.0
    except (TypeError, ValueError):
        return default


def _normalise_record(record: dict, currency: str) -> float:
    """
    Turn one API record into MAD per ONE unit of `currency`.

    Addresses: P1 — `uniteDevise` is the quote-convention trap. BAM publishes
    some currencies per 100 units, so `moyen` alone is the price of
    `uniteDevise` units. Dividing is not optional, and the plausibility check
    afterwards is what stops a convention change from passing unnoticed.
    """
    try:
        moyen = float(record["moyen"])
        # `or 1` would be wrong here: it maps a literal 0 to 1 as well as None,
        # silently turning an invalid unit into a valid-looking rate and making
        # the guard below unreachable. Only ABSENCE defaults.
        raw_unit = record.get("uniteDevise")
        unit = 1.0 if raw_unit is None else float(raw_unit)
    except (KeyError, TypeError, ValueError) as exc:
        raise BAMApiError(f"Unparseable rate record for {currency}: {record!r}") from exc

    if unit <= 0:
        raise BAMApiError(f"Non-positive uniteDevise in record for {currency}: {record!r}")

    rate = moyen / unit
    if not (PLAUSIBLE_USDMAD_MIN <= rate <= PLAUSIBLE_USDMAD_MAX) and currency == "USD":
        raise BAMApiError(
            f"Normalised {currency}/MAD rate {rate:.6f} (moyen={moyen}, "
            f"uniteDevise={unit}) is outside the plausible band "
            f"[{PLAUSIBLE_USDMAD_MIN}, {PLAUSIBLE_USDMAD_MAX}]. Either the quote "
            f"convention changed or this is a bad print; both need a human."
        )
    return rate


def fetch_reference_rate(
    on: _date | datetime | str,
    currency: str = "USD",
    session: requests.Session | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> float | None:
    """
    Fetch one day's official reference rate, as MAD per one unit of `currency`.

    Addresses: P1, P4 — see module docstring.

    A day with no publication (weekend, Moroccan public holiday, 25-26 December)
    returns HTTP 204 and yields None. That is NOT an error and must not be
    treated as one: mistaking "market closed" for "series ended" would silently
    truncate the history, and mistaking it for a zero would be worse.

    Args:
        on: The date to price.
        currency: BAM currency label, e.g. "USD", "EUR".
        session: Optional requests.Session for connection reuse across a backfill.
        timeout: Per-request timeout in seconds.
        max_retries: Attempts on 429/5xx, with linear backoff.

    Returns:
        The rate, or None if BAM published nothing that day.

    Raises:
        BAMApiError: on auth failure, exhausted retries, or an unusable payload.
    """
    day = pd.Timestamp(on).date()
    getter = session.get if session is not None else requests.get
    headers = {SUBSCRIPTION_HEADER: api_key(), "Accept": "application/json"}
    params = {"libDevise": currency, "date": day.isoformat()}

    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = getter(
                COURS_VIREMENT_URL, headers=headers, params=params, timeout=timeout
            )
        except requests.RequestException as exc:
            last_error = f"transport error: {exc}"
            time.sleep(attempt)
            continue

        if response.status_code == 204:
            return None
        if response.status_code in (401, 403):
            # Never echo the key, not even truncated.
            raise BAMApiError(
                f"BAM API rejected the subscription key (HTTP {response.status_code}). "
                f"Check that {CREDENTIAL_ENV_VAR} holds a key subscribed to the "
                f"'Marché des changes' product."
            )
        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            # BAM's gateway states exactly how long to wait ("Rate limit is
            # exceeded. Try again in 58 seconds.") and sets Retry-After. Honour
            # it: the measured budget is 5 requests per 60-second window, so a
            # guessed linear backoff either burns the quota again immediately
            # or idles far longer than needed.
            wait = _retry_after_seconds(response, default=attempt * 2)
            log.debug("HTTP %s — waiting %.1fs before retry %d.",
                      response.status_code, wait, attempt)
            time.sleep(wait)
            continue
        if response.status_code != 200:
            raise BAMApiError(
                f"BAM API returned HTTP {response.status_code} for {currency} on {day}."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise BAMApiError(f"BAM API returned non-JSON for {currency} on {day}.") from exc

        records = payload if isinstance(payload, list) else [payload]
        records = [r for r in records if isinstance(r, dict) and r.get("moyen") is not None]
        if not records:
            return None
        return _normalise_record(records[0], currency)

    raise BAMApiError(
        f"BAM API unreachable for {currency} on {day} after {max_retries} attempts "
        f"({last_error})."
    )


def fetch_reference_rate_series(
    start: _date | str,
    end: _date | str,
    currency: str = "USD",
    session: requests.Session | None = None,
    pause_seconds: float = POLITE_PAUSE_SECONDS,
    progress_every: int = 25,
) -> pd.Series:
    """
    Walk a date range one business day at a time and assemble a rate series.

    Addresses: P1, P4 — the API serves a single date per call, so a backfill is
    inherently a loop. Non-publication days are skipped rather than filled, so
    the returned series contains only OBSERVED rates; deciding how to project
    them onto a price calendar is `currency.align_fx_to_dates`' job, and it
    forward-fills causally with a bounded limit.

    Args:
        start: First date to request (inclusive).
        end: Last date to request (inclusive).
        currency: BAM currency label.
        session: Optional session for connection reuse — worth passing.
        pause_seconds: Delay between calls. Defaults to the measured budget
            (5 requests / 60s), because pacing BELOW the limit is far cheaper
            than repeatedly tripping it: each 429 costs a ~58-second penalty,
            so an unpaced loop is slower than a paced one as well as ruder.
        progress_every: Emit an INFO line every N requests.

    Returns:
        Float Series indexed by observation date, ascending, no NaN.
    """
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    owned_session = session is None
    session = session or requests.Session()
    observations: dict[pd.Timestamp, float] = {}
    missing = 0

    try:
        for i, day in enumerate(days, start=1):
            rate = fetch_reference_rate(day, currency=currency, session=session)
            if rate is None:
                missing += 1
            else:
                observations[day] = rate
            if pause_seconds:
                time.sleep(pause_seconds)
            if progress_every and i % progress_every == 0:
                log.info(
                    "BAM %s: %d/%d requested, %d observed, %d without publication.",
                    currency, i, len(days), len(observations), missing,
                )
    finally:
        if owned_session:
            session.close()

    series = pd.Series(observations, dtype="float64").sort_index()
    series.index.name = "Date"
    series.name = f"{currency}MAD"
    log.info(
        "BAM %s: %d observations over %d business days (%d without publication).",
        currency, len(series), len(days), missing,
    )
    return series


def has_publication_near(
    anchor: _date,
    currency: str = "USD",
    window_days: int = 4,
    session: requests.Session | None = None,
    pause_seconds: float = POLITE_PAUSE_SECONDS,
) -> bool:
    """
    Test whether BAM serves ANY rate in a window around `anchor`.

    Addresses: P4 — a single 204 is ambiguous: it means "no publication", which
    covers weekends, Moroccan public holidays and 25-26 December as well as
    "outside the served range". Probing a window of consecutive business days
    removes that ambiguity, so the coverage boundary found by
    `scripts/probe_bam_history.py` is a real boundary and not a holiday.

    Args:
        anchor: Centre of the probe window.
        currency: BAM currency label.
        window_days: Business days to try, starting at `anchor`.
        session: Optional session for connection reuse.

    Returns:
        True as soon as one day in the window returns a rate.
    """
    owned_session = session is None
    session = session or requests.Session()
    try:
        for i, day in enumerate(pd.bdate_range(pd.Timestamp(anchor), periods=window_days)):
            # Pace from the second request on. Tripping the 5/min limit costs a
            # ~58s penalty, so staying under it is strictly faster than racing it.
            if i and pause_seconds:
                time.sleep(pause_seconds)
            if fetch_reference_rate(day, currency=currency, session=session) is not None:
                return True
        return False
    finally:
        if owned_session:
            session.close()
