"""
backfill_bam_fx.py — fetch the official USD/MAD reference rate for `full_2021`.

SCOPE, deliberately narrow and authorised as such:
    Allowed   BAM official rates, 2021-07-01 -> the last required date only.
    Purpose   repair `full_2021`, the mixed MAD/USD universe.
    Forbidden 2018-2021 download, conversion of `etf_2017`, any DVC rebuild.

`etf_2017` is out of scope by policy, not by omission: it holds five
USD-denominated ETFs, has one numéraire already, and carries no FX dependency
(see `currency.resolve_currency_policy`). Widening this window to 2018 would
buy an extra MAD view of that universe, which repairs no published artifact and
is outside what was agreed.

WHY A NEW BRONZE FILE
    Writes `data/bronze/bam_fx_reference.parquet` rather than touching
    `raw_bam_macro.parquet`. Bronze is immutable (§15.5), the Yahoo series is
    still what feeds the Phase 1/3 macro FEATURES, and the two must remain
    separately auditable: one is a defective quote feed kept for provenance, the
    other is the official rate that will value a portfolio.

RESUMABLE, because it has to be
    ~1,330 business days at a measured 5 requests/minute is ~4.4 hours. A run
    that lost its progress to a dropped connection would be unacceptable, so
    every batch is flushed to disk and a restart re-fetches only the dates still
    missing. Re-running after completion is a no-op.

Usage:
    python scripts/backfill_bam_fx.py                 # 2021-07-01 -> today
    python scripts/backfill_bam_fx.py --end 2026-07-24
    python scripts/backfill_bam_fx.py --dry-run       # show the plan, fetch nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bam_fx import (  # noqa: E402
    BAMApiError,
    MEASURED_RATE_LIMIT_PER_MINUTE,
    POLITE_PAUSE_SECONDS,
    api_key,
    fetch_reference_rate,
)
from currency import FX_SERIES, fx_quality_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("backfill_bam")

# The authorised window. `full_2021`'s aligned price calendar starts 2021-07-01;
# fetching earlier would be outside scope, fetching later would leave the
# universe's opening dates uncovered.
AUTHORISED_START = date(2021, 7, 1)
OUTPUT_PATH = ROOT / "data" / "bronze" / "bam_fx_reference.parquet"
FLUSH_EVERY = 20


def load_existing(path: Path) -> pd.Series:
    """Read whatever a previous run already fetched, so a restart is cheap."""
    if not path.exists():
        return pd.Series(dtype="float64", name=FX_SERIES)
    existing = pd.read_parquet(path)[FX_SERIES].astype("float64")
    existing.index = pd.to_datetime(existing.index)
    log.info(
        "Resuming: %d rate(s) already on disk (%s -> %s).",
        len(existing), existing.index.min().date(), existing.index.max().date(),
    )
    return existing.sort_index()


def empty_dates_path(out_path: Path) -> Path:
    return out_path.with_name("bam_fx_reference_no_publication.json")


def load_known_empty(out_path: Path) -> set:
    """Dates already confirmed to carry no publication.

    Without this, a resume re-queries every non-publication day, because only
    SUCCESSFUL dates land in the parquet. On a feed that returns empty for a
    meaningful share of 2021 that is not a rounding error: at 5 requests per
    minute, re-asking a few hundred known-empty dates costs over an hour of
    wall-clock on every restart, and the answer is already known.
    """
    path = empty_dates_path(out_path)
    if not path.exists():
        return set()
    known = {pd.Timestamp(d) for d in json.loads(path.read_text())["no_publication"]}
    log.info("Resuming: %d date(s) already known to have no publication.", len(known))
    return known


def flush_known_empty(known: set, out_path: Path) -> None:
    empty_dates_path(out_path).write_text(json.dumps(
        {
            "note": (
                "Dates BAM confirmed carry no reference-rate publication (HTTP 200 "
                "with an empty body). Recorded so a resumed backfill does not re-ask "
                "a question already answered. These are NOT failures."
            ),
            "n": len(known),
            "no_publication": sorted(str(pd.Timestamp(d).date()) for d in known),
        },
        indent=2,
    ))


def flush(series: pd.Series, path: Path) -> None:
    """Persist progress. Called often — losing 4 hours to a dropped socket is
    a far worse outcome than a few extra parquet writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = series.sort_index().to_frame(FX_SERIES)
    frame.index.name = "Date"
    frame.to_parquet(path)


def write_reports(series, start, end, out_path) -> bool:
    """Persist the gap structure and quality verdict for a fetched series.

    Split out of `main` so a no-op resume refreshes it too: this JSON is a
    DVC output and the versioned quality control on the Bronze file, so it
    must always describe the data actually on disk.

    Returns:
        True if the series passes the FX quality gate.
    """
    # ── Gap structure: the number that actually decides usability ────────────
    # Density alone does not. `currency.align_fx_to_dates` fills forward with a
    # bounded limit, so a series can be 95% dense and still unusable if the 5%
    # arrives as one 17-day hole — which is exactly what 2021-07 looked like.
    # The LONGEST CONSECUTIVE RUN is therefore the decisive figure, and it is
    # reported here rather than left to be discovered by a failing pipeline.
    calendar = pd.bdate_range(start, end)
    present = calendar.isin(series.index)
    runs, current = [], 0
    for ok in present:
        if ok:
            if current:
                runs.append(current)
            current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    longest = max(runs) if runs else 0
    gaps = {
        "business_days_in_window": int(len(calendar)),
        "observed": int(present.sum()),
        "density": round(float(present.mean()), 4),
        "longest_consecutive_missing_business_days": int(longest),
        "n_gaps": len(runs),
        "n_gaps_over_ffill_limit_5": int(sum(1 for r in runs if r > 5)),
    }
    log.info("Gap structure: %s", gaps)
    if longest > 5:
        log.warning(
            "Longest gap is %d business days, beyond the causal forward-fill limit "
            "of 5. Those dates CANNOT be filled without inventing a rate, so the "
            "usable window starts after the last such gap — not at %s.",
            longest, start,
        )

    # ── Report the quality of what was actually fetched ──────────────────────
    # The whole point of switching source is that the Yahoo feed failed this
    # gate. Running it here means the answer is known BEFORE anything is wired
    # in, rather than discovered during a rebuild.
    report = fx_quality_report(series)
    verdict = "PASSES" if report["passed"] else "FAILS"
    log.info("FX quality gate: %s", verdict)
    for key in ("annualised_volatility", "lag1_autocorrelation", "outlier_share",
                "level_min", "level_max"):
        log.info("    %-24s %s", key, report.get(key))
    for failure in report.get("blocking_failures", []):
        log.error("    BLOCKING: %s", failure)

    summary_path = out_path.with_name("bam_fx_reference_quality.json")
    summary_path.write_text(json.dumps(
        {
            "generated_at": pd.Timestamp.now().isoformat(),
            "source": "Bank Al-Maghrib, Cours de référence (CoursVirement API)",
            "scope": "full_2021 repair only; 2018-2021 and etf_2017 out of scope",
            "window": {"start": str(series.index.min().date()),
                       "end": str(series.index.max().date()),
                       "n_observations": int(len(series))},
            "gap_structure": gaps,
            "quality": report,
        },
        indent=2, default=str,
    ))
    log.info("Quality summary -> %s", summary_path)
    return bool(report["passed"])

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=AUTHORISED_START.isoformat())
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()
    out_path = Path(args.out)

    if start < AUTHORISED_START:
        log.error(
            "Refusing to fetch from %s. The authorised window starts %s; earlier "
            "dates were explicitly excluded from scope (they repair no published "
            "artifact and would add an out-of-scope MAD view of etf_2017).",
            start, AUTHORISED_START,
        )
        return 2

    existing = load_existing(out_path)
    known_empty = load_known_empty(out_path)
    wanted = pd.bdate_range(start, end)
    missing = [d for d in wanted if d not in existing.index and d not in known_empty]

    log.info(
        "Window %s -> %s: %d business days, %d already held, %d to fetch "
        "(~%.1f h at %d req/min).",
        start, end, len(wanted), len(wanted) - len(missing), len(missing),
        len(missing) / MEASURED_RATE_LIMIT_PER_MINUTE / 60, MEASURED_RATE_LIMIT_PER_MINUTE,
    )

    if args.dry_run:
        log.info("--dry-run: nothing fetched.")
        return 0
    if not missing:
        # A no-op run must still (re)write the quality artifact. It is a DVC
        # output and the versioned control on this Bronze file; leaving it
        # untouched on a resume would let `dvc repro` see a missing/stale out
        # and would break the promise that the control describes the data.
        log.info("Nothing to do — the window is already complete; refreshing quality report.")
        write_reports(existing, start, end, out_path)
        return 0

    try:
        api_key()
    except BAMApiError as exc:
        log.error("%s", exc)
        return 2

    observations = dict(existing.items())
    no_publication = 0

    with requests.Session() as session:
        for i, day in enumerate(missing, start=1):
            try:
                rate = fetch_reference_rate(day, session=session)
            except BAMApiError as exc:
                # Persist before surfacing: partial progress is worth keeping.
                flush(pd.Series(observations, dtype="float64"), out_path)
                flush_known_empty(known_empty, out_path)
                log.error("Stopped at %s after %d/%d: %s", day.date(), i, len(missing), exc)
                log.error("Progress saved — re-run this script to resume from here.")
                return 1

            if rate is None:
                no_publication += 1
                known_empty.add(day)
            else:
                observations[day] = rate

            if i % FLUSH_EVERY == 0:
                flush(pd.Series(observations, dtype="float64"), out_path)
                flush_known_empty(known_empty, out_path)
                log.info(
                    "  %d/%d fetched (%d observed, %d non-publication days), "
                    "~%.1f h remaining.",
                    i, len(missing), len(observations) - len(existing), no_publication,
                    (len(missing) - i) / MEASURED_RATE_LIMIT_PER_MINUTE / 60,
                )
            if POLITE_PAUSE_SECONDS:
                import time
                time.sleep(POLITE_PAUSE_SECONDS)

    series = pd.Series(observations, dtype="float64").sort_index()
    flush(series, out_path)
    flush_known_empty(known_empty, out_path)
    log.info("Wrote %d observations -> %s", len(series), out_path)

    return 0 if write_reports(series, start, end, out_path) else 3



if __name__ == "__main__":
    raise SystemExit(main())
