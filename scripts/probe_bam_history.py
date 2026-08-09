"""
probe_bam_history.py — how far back does Bank Al-Maghrib's API actually serve?

This is the ONE question that decides whether the base-currency correction can
be completed as designed. The conversion is built and tested; the Yahoo feed
that would drive it fails its quality gate; BAM is the authoritative
replacement — but nothing BAM publishes documents the depth of its API, and the
public pages expose only the current and prior day. The ETF universe needs
2004-11 onwards (~5,656 business days).

⚠️ WHAT THIS PROBE DOES AND DOES NOT MEASURE. It finds the earliest ERA the API
serves anything for. It says NOTHING about DENSITY within that era, and the two
are independent: the predicate below is "at least one publication in a window of
`window_days` business days", which a sparse archive satisfies easily. Measured
afterwards on 2021 data, BAM returned `HTTP 200 []` for 17 of 20 consecutive
business days — so "covered" here must never be read as "complete". A separate
density measurement is required before any window is treated as usable, and a
boundary reported from this probe alone is a LOWER bound on availability, not a
guarantee of a continuous series.

Three possible outcomes, all reported honestly:

  (a) BAM serves the whole window       -> backfill and rebuild as planned.
  (b) BAM serves a shallower window     -> a decision is required: splice a
                                           documented secondary source over the
                                           tail, or shorten the MAD-converted
                                           evaluation window.
  (c) BAM serves only recent dates      -> the conversion cannot be driven by
                                           an official source at this history
                                           depth, and that is a finding, not a
                                           failure.

It also reports what a full backfill would cost, so that is known before it is
started rather than discovered during it. The binding constraint is the
gateway's rate limit, measured empirically at 5 requests per 60-second window
(the 6th returns HTTP 429 with `Retry-After: 58`) — not latency, which is a
comfortable 0.31s per call. That limit is what makes the search order here
matter: probes are paced, so the cheapest informative question is asked first.

READ-ONLY. Writes one JSON report to the path given by --out (default:
data/interim/bam_history_probe.json) and touches no Bronze, Silver or Gold
artifact, no DVC stage and no committed result.

Usage:
    export $(grep -v '^#' .env | xargs)      # or however you load .env
    python scripts/probe_bam_history.py
    python scripts/probe_bam_history.py --currency EUR --out /tmp/probe.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bam_fx import (  # noqa: E402
    BAMApiError,
    MEASURED_RATE_LIMIT_PER_MINUTE,
    POLITE_PAUSE_SECONDS,
    UNIVERSE_START,
    api_key,
    fetch_reference_rate,
    has_publication_near,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("probe_bam")


def find_earliest_served_date(
    currency: str,
    floor: date,
    ceiling: date,
    session: requests.Session,
    window_days: int = 4,
) -> tuple[date | None, int]:
    """
    Locate the coverage boundary on a budget of 5 requests per minute.

    The predicate is `has_publication_near`, not a single-date lookup, because a
    lone HTTP 204 cannot distinguish "outside the served range" from "weekend or
    Moroccan public holiday". Each probe therefore costs up to `window_days`
    requests, and at ~12.5s per request a probe is expensive enough that the
    search order matters.

    Hence coarse-to-fine, cheapest-informative-question-first:
      1. Test the floor. If BAM covers 2004-11 the search is over in one probe
         and the whole question is answered.
      2. Otherwise bisect on YEAR boundaries — ~4 probes across two decades.
      3. Refine to the month only inside the year that straddles the boundary.

    Returns:
        (earliest_served_date_or_None, n_probes)
    """
    probes = 0

    if has_publication_near(floor, currency, window_days, session):
        log.info("Coverage reaches the universe start (%s) — full coverage.", floor)
        return floor, probes + 1
    probes += 1
    log.info("  %s: no publication — boundary is later; bisecting by year.", floor)

    years = [date(y, 1, 15) for y in range(floor.year + 1, ceiling.year + 1)]
    lo, hi = 0, len(years) - 1         # lo: assumed-uncovered, hi: assumed-covered
    while lo < hi:
        mid = (lo + hi) // 2
        covered = has_publication_near(years[mid], currency, window_days, session)
        probes += 1
        log.info("  probe %s -> %s", years[mid], "covered" if covered else "empty")
        if covered:
            hi = mid
        else:
            lo = mid + 1

    if not has_publication_near(years[lo], currency, window_days, session):
        log.error("Even the most recent year probe came back empty — check the key.")
        return None, probes + 1
    probes += 1

    # Refine over the 12 months ENDING at the first covered anchor, not the
    # calendar year containing it. The boundary lies between the last empty
    # anchor and this one, and those are ~12 months apart on different sides of
    # a New Year — restricting the search to `years[lo].year` cannot reach the
    # earlier side, so it silently returns an UPPER BOUND rather than the edge.
    first_covered = years[lo]
    months = [
        (first_covered - pd.DateOffset(months=12 - i)).date() for i in range(12)
    ] + [first_covered]
    log.info("Boundary lies in (%s, %s] — refining to the month.", months[0], first_covered)
    lo_m, hi_m = 0, len(months) - 1
    while lo_m < hi_m:
        mid = (lo_m + hi_m) // 2
        covered = has_publication_near(months[mid], currency, window_days, session)
        probes += 1
        log.info("  probe %s -> %s", months[mid], "covered" if covered else "empty")
        if covered:
            hi_m = mid
        else:
            lo_m = mid + 1

    return months[lo_m], probes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--window-days", type=int, default=4)
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "interim" / "bam_history_probe.json")
    )
    args = parser.parse_args()

    try:
        api_key()
    except BAMApiError as exc:
        log.error("%s", exc)
        return 2

    today = date.today()
    report: dict = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "currency": args.currency,
        "endpoint": "cours/Version1/api/CoursVirement",
        "universe_start_required": UNIVERSE_START.isoformat(),
    }

    with requests.Session() as session:
        # ── 1. Liveness + convention, on a recent date ───────────────────────
        log.info("Checking a recent date and the quote convention...")
        recent_rate = None
        for back in range(1, 15):
            probe_day = today - timedelta(days=back)
            recent_rate = fetch_reference_rate(probe_day, args.currency, session=session)
            if recent_rate is not None:
                report["recent_observation"] = {
                    "date": probe_day.isoformat(),
                    "rate_mad_per_unit": round(recent_rate, 6),
                }
                log.info("  1 %s = %.4f MAD on %s", args.currency, recent_rate, probe_day)
                break
        if recent_rate is None:
            log.error("No rate returned for any of the last 14 days. Aborting.")
            report["status"] = "no_recent_data"
            _write(report, args.out)
            return 1

        # ── 2. Backfill cost, from the MEASURED gateway budget ───────────────
        # Not re-measured by hammering: the limit was established empirically
        # (the 6th request in a 60s window returns 429 with Retry-After: 58),
        # and re-deriving it here would burn the very quota the probe needs.
        business_days = len(pd.bdate_range(UNIVERSE_START, today))
        minutes = business_days / MEASURED_RATE_LIMIT_PER_MINUTE
        report["performance"] = {
            "measured_rate_limit_per_minute": MEASURED_RATE_LIMIT_PER_MINUTE,
            "seconds_per_request_unthrottled": 0.31,
            "business_days_in_full_window": business_days,
            "estimated_full_backfill_hours": round(minutes / 60, 1),
        }
        log.info(
            "  rate limit %d/min -> ~%.1f h to backfill %d business days",
            MEASURED_RATE_LIMIT_PER_MINUTE, minutes / 60, business_days,
        )

        # ── 3. The boundary ──────────────────────────────────────────────────
        log.info("Binary-searching the earliest served date (this takes a few minutes)...")
        earliest, probes = find_earliest_served_date(
            args.currency, UNIVERSE_START, today, session, args.window_days
        )
        report["probe_count"] = probes

    if earliest is None:
        report["status"] = "probe_failed"
    else:
        covers = earliest <= UNIVERSE_START
        report["earliest_served_date"] = earliest.isoformat()
        report["covers_universe_start"] = covers
        report["status"] = "covers_full_window" if covers else "partial_coverage"
        if not covers:
            report["uncovered_business_days"] = len(
                pd.bdate_range(UNIVERSE_START, earliest)
            ) - 1
        log.info(
            "EARLIEST SERVED: %s | required: %s | %s",
            earliest, UNIVERSE_START,
            "FULL COVERAGE" if covers else "PARTIAL — a decision is required",
        )

    _write(report, args.out)
    return 0


def _write(report: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    log.info("Probe report written -> %s", path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
