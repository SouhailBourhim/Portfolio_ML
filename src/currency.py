"""
currency.py — base-currency (numéraire) conversion for the Silver layer.

Addresses: P1, P4 — a portfolio P&L is a sum of amounts of money, so every
term must be denominated in the same currency before it is added. Until this
module existed the pipeline summed MAD-denominated BVC returns and
USD-denominated ETF returns into one portfolio and defended it with "returns
are unitless, so the arithmetic is valid". That defence is wrong twice:

  * P1 — the covariance matrix built from mixed-numéraire returns omits the
    variance of, and every covariance with, the USD/MAD exchange rate. The
    optimiser therefore cannot see a risk the holder actually carries, and
    systematically understates the risk of the USD-denominated sleeve.
  * P4 — a Moroccan investor could not have earned the reported series. A
    published number that no holder could realise is exactly the class of
    result this project's out-of-sample discipline exists to eliminate.

Canonical numéraire: MAD. BVC equities are already MAD-denominated and are
passed through untouched (their dividend total-return handling is unaffected).
ETF prices are converted at the observed spot rate BEFORE returns are computed:

    price_mad[t] = price_usd[t] * USDMAD[t]                    (MAD per USD)

which, because this project computes log-returns (AGENTS.md §15.1), makes the
converted return exactly additive:

    r_mad[t] = r_usd[t] + log(USDMAD[t] / USDMAD[t-1])

No approximation, no cross-term. That identity is asserted directly by
tests/test_currency.py rather than assumed.

WHAT THE PRE-CORRECTION NUMBERS ACTUALLY WERE. State this precisely, because
the loose version is wrong and will be challenged. Summing local-currency
returns is NOT the return of a hedged portfolio: a real hedge is a position in
FX forwards, and its return depends on the forward points — i.e. on the
MAD/USD interest-rate differential — which appears nowhere in this project.
The correct characterisation is narrower and harsher: **the FX return component
was simply omitted.** The old series is not a portfolio anyone could hold,
hedged or otherwise; it is a portfolio whose USD sleeve was valued as though
the exchange rate did not exist.

HEDGING IS NOT MODELLED HERE EITHER. The result is an *unhedged* MAD series:
realised FX variation is included in the reported performance, and there is no
forward contract, hedge ratio, forward-point curve or rollover cost anywhere in
this module. A hedged variant needs a forward curve and is a different (and
larger) piece of work.

CAUSALITY. FX is aligned to the price grid by forward-fill only, bounded by
`ffill_limit`. Backward-filling an exchange rate would let tomorrow's rate
value today's holding, which is the same lookahead the rest of the pipeline
forbids (§15.4). Any date the aligned series cannot cover from the past is a
hard failure, never a silent pass-through of the USD number.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from schemas import BVC_ASSETS, ETF_ASSETS

log = logging.getLogger("currency")

# ── The conversion contract, in one place ────────────────────────────────────
# These are persisted verbatim into the Silver validation reports and the Gold
# currency manifest, so a reader of an artifact never has to infer the
# numéraire from the code that produced it.
BASE_CURRENCY = "MAD"
FOREIGN_CURRENCY = "USD"
FX_SERIES = "USDMAD"
FX_QUOTE_CONVENTION = "MAD per USD"
HEDGE_STATUS = "unhedged"

# ── The FX source of record ─────────────────────────────────────────────────
# Bank Al-Maghrib's own published reference rate, fetched by
# scripts/backfill_bam_fx.py. This is NOT `raw_bam_macro.parquet`: that file
# holds the Yahoo `USDMAD=X` quote, which is retained only as an input to the
# Phase 1/3 macro FEATURES and must never value a portfolio again. Measured
# head-to-head on 1,214 overlapping dates, Yahoo tracks the right LEVEL
# (correlation 0.958) but its daily CHANGES correlate with the official rate's
# at just 0.028 -- essentially noise, while returns are made of nothing else.
# It overstates FX volatility 6.0x (33.4% vs 5.6% annualised).
#
# There is deliberately no fallback from one to the other. A missing official
# file must stop the run, because the failure it would otherwise cause is
# invisible: plausible magnitudes, right dates, wrong numbers.
OFFICIAL_FX_FILENAME = "bam_fx_reference.parquet"
FX_SOURCE_NAME = "Bank Al-Maghrib — Cours de référence (CoursVirement API)"

# The mixed universe cannot start before this date. BAM's archive has exactly
# one gap wider than the causal forward-fill limit -- 2021-07-06 to 2021-07-28,
# 17 business days -- and those dates cannot be filled without inventing a rate.
# Costs 19 of 1,321 rows and leaves the OOS window (2022-07-01+) untouched.
# `validate_fx_gap_structure` re-derives this from the data on every run rather
# than trusting the constant, so a refreshed archive cannot silently invalidate it.
MIXED_UNIVERSE_START = "2021-07-29"

# ── FX quality thresholds — BLOCKING in production ──────────────────────────
# These live here rather than in params.yaml deliberately. `src/currency.py` is
# a declared DVC dependency of the `clean` stage, so editing a threshold
# invalidates the Silver layer exactly as a params change would; adding a
# params key that no code path actually reads would create the silent
# code-vs-config drift this project has been bitten by before (§17.1).
#
# Every value is justified by what the Moroccan dirham IS: a managed float
# inside a narrow band around a 60/40 EUR/USD basket. It is not a free-floating
# currency and must not behave like one.
FX_QUALITY_THRESHOLDS = {
    # Lag-1 autocorrelation of daily log-returns. A genuine spot rate sits near
    # zero. Strongly negative is the bid-ask / two-contributor bounce: the quote
    # alternates between two levels, inflating variance with no market movement.
    "min_lag1_autocorrelation": -0.25,
    # A managed basket peg realises roughly 5-7% annualised. 15% is a generous
    # ceiling that still separates a real rate from a bouncing quote feed by a
    # wide margin.
    "max_annualised_volatility": 0.15,
    # |daily log-return| above this is implausible for a band-managed currency.
    "outlier_abs_log_return": 0.05,
    # ...but a band widening or a rebasing IS a real event, so a handful across
    # two decades is tolerated. A rate, not a count: 0.1% of 5,000 observations
    # allows ~5 genuine policy events and still rejects a feed with systematic
    # bad prints.
    "max_outlier_share": 0.001,
    # QUOTE CONVENTION. USDMAD must be MAD per USD, i.e. ~8-11. This is the
    # check that catches an INVERTED series (USD per MAD, ~0.10): inversion
    # would silently divide every ETF price instead of multiplying it, and
    # nothing else in the pipeline would notice. Also catches a unit/decimal
    # error (~90 or ~0.9).
    "plausible_level_min": 5.0,
    "plausible_level_max": 15.0,
    # Fraction of required dates carrying a real observation rather than an
    # inherited one. Guards against a sparse feed that technically passes the
    # forward-fill limit on every individual gap.
    "min_observation_density": 0.90,
}


class FXDataError(RuntimeError):
    """Base: the FX input cannot support a trustworthy MAD valuation."""


class FXDataUnavailable(FXDataError):
    """
    The FX series required to express prices in the base currency is absent,
    malformed, or does not cover every date that needs converting.

    Addresses: P1, P4 — the alternative to raising is returning a portfolio
    whose USD sleeve is silently still in USD. That reintroduces the exact
    mixed-numéraire defect this module was written to remove, and it does so
    invisibly: the output has the right shape, the right dates and plausible
    magnitudes. Following `DividendDataUnavailable` (clean.py), the failure is
    made loud rather than degraded, because on the unattended Dagster schedule
    nobody reads a WARNING.
    """


class FXQualitySuspect(FXDataError):
    """
    The FX series is present and well-formed, but its statistical behaviour
    says it is not a usable exchange rate.

    Addresses: P1, P4 — this is a SEPARATE failure from absence, and a more
    dangerous one, because the run would otherwise succeed. A quote feed that
    bounces between two contributors carries variance that is not market
    movement; converting with it injects that variance into every ETF return,
    inflates the whole covariance block, and drives every risk-based optimiser
    to conclusions about an artifact. Measured on the live Yahoo `USDMAD=X`
    feed: annualised volatility 25.9% against a real ~6.3%, lag-1
    autocorrelation -0.42, and the mean pairwise ETF correlation rising from
    0.34 to 0.85 purely from the spurious common factor.

    A WARNING was not enough. A production run must stop.
    """


def load_fx_rates(bronze_path, series: str = FX_SERIES) -> pd.Series:
    """
    Load the FX spot series from the Bronze BAM macro file.

    Addresses: P1, P4 — see module docstring. This is the single reader of the
    FX column used for conversion, so the quote convention is asserted in one
    place instead of being re-derived by each caller.

    Args:
        bronze_path: Path to `raw_bam_macro.parquet`.
        series: Column name to extract. `USDMAD` is MAD per USD.

    Returns:
        Float Series indexed by date, ascending, NaN rows dropped.

    Raises:
        FXDataUnavailable: if the file or the column is missing, or the column
            holds no usable observation.
    """
    from pathlib import Path

    bronze_path = Path(bronze_path)
    if not bronze_path.exists():
        raise FXDataUnavailable(
            f"FX source {bronze_path} not found, so ETF prices cannot be expressed "
            f"in {BASE_CURRENCY}. Run `python src/ingest.py` to produce it. Refusing "
            f"to continue: without it the portfolio would mix {BASE_CURRENCY} and "
            f"{FOREIGN_CURRENCY} amounts in one sum."
        )

    frame = pd.read_parquet(bronze_path)
    if series not in frame.columns:
        raise FXDataUnavailable(
            f"FX column {series!r} absent from {bronze_path} "
            f"(columns present: {list(frame.columns)})."
        )

    fx = frame[series].astype("float64")
    fx.index = pd.to_datetime(fx.index)
    fx.index.name = "Date"
    fx = fx.dropna().sort_index()

    if fx.empty:
        raise FXDataUnavailable(
            f"FX column {series!r} in {bronze_path} has no non-NaN observation."
        )

    log.info(
        "FX %s loaded: %d observations, %s → %s (%s)",
        series, len(fx), fx.index.min().date(), fx.index.max().date(),
        FX_QUOTE_CONVENTION,
    )
    return fx


def validate_fx_series(fx: pd.Series, series: str = FX_SERIES) -> pd.Series:
    """
    Reject an FX series that cannot safely value a portfolio.

    Addresses: P1, P4 — every rejection here corresponds to a way the
    conversion could produce a plausible-looking but wrong MAD price. A zero
    or negative rate makes the log-return infinite or undefined; a duplicated
    date makes the multiplication ambiguous; an unsorted index makes
    forward-fill silently fill *backwards*, which is lookahead.

    Args:
        fx: Candidate FX spot series.
        series: Name used in error messages.

    Returns:
        The same series, unchanged, once every check passes.

    Raises:
        FXDataUnavailable: on any violation, naming the offending dates.
    """
    if not isinstance(fx, pd.Series):
        raise FXDataUnavailable(f"FX {series!r} must be a Series, got {type(fx).__name__}.")

    if fx.empty:
        raise FXDataUnavailable(f"FX {series!r} is empty.")

    if not isinstance(fx.index, pd.DatetimeIndex):
        raise FXDataUnavailable(
            f"FX {series!r} must have a DatetimeIndex, got {type(fx.index).__name__}."
        )

    duplicated = fx.index[fx.index.duplicated()]
    if len(duplicated) > 0:
        raise FXDataUnavailable(
            f"FX {series!r} has duplicate dates: {sorted({d.date() for d in duplicated})[:5]}. "
            f"Which rate applies on those dates is undefined."
        )

    if not fx.index.is_monotonic_increasing:
        raise FXDataUnavailable(
            f"FX {series!r} index is not sorted ascending. Forward-filling an "
            f"unsorted series propagates values BACKWARDS in time, which is "
            f"lookahead (§15.4). Sort at the source rather than here, so the "
            f"disorder is fixed where it was introduced."
        )

    if fx.isna().any():
        bad = fx.index[fx.isna()]
        raise FXDataUnavailable(
            f"FX {series!r} contains {len(bad)} NaN rows (first: {bad[0].date()}). "
            f"Drop them at load time — a NaN rate would silently NaN out an "
            f"entire converted price row."
        )

    nonpositive = fx.index[fx <= 0]
    if len(nonpositive) > 0:
        raise FXDataUnavailable(
            f"FX {series!r} has {len(nonpositive)} non-positive rate(s), first on "
            f"{nonpositive[0].date()}. An exchange rate of zero or below is not a "
            f"price; log-returns of the converted series would be infinite or undefined."
        )

    return fx


def align_fx_to_dates(
    fx: pd.Series,
    dates: pd.DatetimeIndex,
    ffill_limit: int = 5,
    series: str = FX_SERIES,
) -> pd.Series:
    """
    Project an FX series onto the price calendar using past information only.

    Addresses: P1, P4 — this is the single point where a future exchange rate
    could leak into a past valuation, so the fill is forward-only and bounded.
    `reindex(method="ffill")` resolves each target date to the most recent
    observation at or before it and can never reach forward; the `limit` then
    stops a stale rate from being carried across a long outage. Whatever the
    fill cannot cover is raised, not passed through (§15.13).

    Args:
        fx: Validated FX spot series.
        dates: The exact dates requiring a rate — the aligned price index.
        ffill_limit: Maximum consecutive dates that may inherit an older rate.
            Matches `clean.ffill_limit` so FX and prices age identically.
        series: Name used in error messages.

    Returns:
        Series indexed exactly by `dates`, no NaN.

    Raises:
        FXDataUnavailable: if any requested date cannot be covered from the past.
    """
    if len(dates) == 0:
        raise FXDataUnavailable("No dates requested for FX alignment.")

    aligned = fx.reindex(dates, method="ffill", limit=ffill_limit)
    aligned.index.name = dates.name or "Date"

    missing = aligned.index[aligned.isna()]
    if len(missing) > 0:
        leading = missing[missing < fx.index.min()]
        detail = (
            f"{len(leading)} of them precede the first FX observation "
            f"({fx.index.min().date()}) and CANNOT be filled without using a "
            f"future rate, which is forbidden"
            if len(leading) > 0
            else f"the gaps exceed ffill_limit={ffill_limit} business days"
        )
        raise FXDataUnavailable(
            f"FX {series!r} does not cover {len(missing)} required date(s) "
            f"({missing[0].date()} … {missing[-1].date()}); {detail}. "
            f"FX coverage is {fx.index.min().date()} → {fx.index.max().date()}; "
            f"prices need {dates.min().date()} → {dates.max().date()}. "
            f"Refusing to fall back to unconverted {FOREIGN_CURRENCY} prices."
        )

    n_filled = int((~aligned.index.isin(fx.index)).sum())
    if n_filled:
        log.info(
            "FX %s: %d of %d dates inherited an earlier rate (forward-fill ≤ %d days).",
            series, n_filled, len(aligned), ffill_limit,
        )
    return aligned


def fx_quality_report(fx: pd.Series, thresholds: dict | None = None) -> dict:
    """
    Judge whether an FX series is fit to value a portfolio.

    Addresses: P1, P4 — the converted covariance matrix inherits this series'
    variance wholesale, so a defect in the quote feed becomes a defect in every
    risk estimate downstream. This routine measures and JUDGES; it never
    cleans, and it never decides what to do about a failure. The verdict is
    carried in `blocking_failures`, which `enforce_fx_quality` acts on.

    Four independent families of check, because they fail independently:

      1. `bounce`     — lag-1 autocorrelation. A genuine spot rate sits near
                        zero; strongly negative means the quote alternates
                        between two sources rather than moving.
      2. `volatility` — a band-managed currency cannot realise free-float
                        volatility.
      3. `outliers`   — implausible single-day jumps, as a SHARE so that
                        genuine policy events are tolerated and systematic bad
                        prints are not.
      4. `convention` — the LEVEL must be consistent with MAD per USD. This is
                        the check that catches an inverted series, which would
                        otherwise silently divide every ETF price.

    Args:
        fx: Validated FX spot series (raw observations, not the aligned grid).
        thresholds: Override `FX_QUALITY_THRESHOLDS`; missing keys fall back to it.

    Returns:
        Diagnostics plus `blocking_failures: list[str]` and `passed: bool`.
    """
    limits = {**FX_QUALITY_THRESHOLDS, **(thresholds or {})}
    failures: list[str] = []

    level_min, level_max = float(fx.min()), float(fx.max())
    if level_min < limits["plausible_level_min"] or level_max > limits["plausible_level_max"]:
        failures.append(
            f"convention: levels span [{level_min:.4f}, {level_max:.4f}], outside the "
            f"plausible band [{limits['plausible_level_min']}, {limits['plausible_level_max']}] "
            f"for {FX_QUOTE_CONVENTION}. An INVERTED series (USD per MAD, ~0.10) would "
            f"look exactly like this and would silently divide every ETF price instead "
            f"of multiplying it."
        )

    returns = np.log(fx / fx.shift(1)).dropna()
    report: dict = {
        "n_observations": int(len(fx)),
        "n_return_observations": int(len(returns)),
        "level_min": round(level_min, 6),
        "level_max": round(level_max, 6),
        "thresholds": dict(limits),
    }

    if len(returns) < 30:
        failures.append(
            f"coverage: only {len(returns)} return observations — too few to judge "
            f"quality or to estimate a covariance from."
        )
        report.update({"blocking_failures": failures, "passed": not failures})
        return report

    std = float(returns.std(ddof=1))
    constant = (not np.isfinite(std)) or std == 0.0

    ann_vol = 0.0 if constant else std * np.sqrt(252)
    # A constant rate has undefined (0/0) autocorrelation. Reporting None is
    # honest; reporting 0.0 would assert a measurement never made.
    lag1 = None if constant else float(returns.autocorr(1))
    outlier_share = float((returns.abs() > limits["outlier_abs_log_return"]).mean())

    report.update({
        "annualised_volatility": round(ann_vol, 6),
        "lag1_autocorrelation": None if lag1 is None else round(lag1, 6),
        "max_abs_daily_log_return": round(float(returns.abs().max()), 6),
        "n_outlier_days": int((returns.abs() > limits["outlier_abs_log_return"]).sum()),
        "outlier_share": round(outlier_share, 6),
        "is_constant": constant,
    })

    if lag1 is not None and lag1 < limits["min_lag1_autocorrelation"]:
        failures.append(
            f"bounce: lag-1 autocorrelation of daily log-returns is {lag1:.3f}, below "
            f"{limits['min_lag1_autocorrelation']}. That is a quote-source/bid-ask "
            f"bounce, not market movement — the series alternates between two levels, "
            f"inflating variance with no economic content."
        )

    if ann_vol > limits["max_annualised_volatility"]:
        failures.append(
            f"volatility: {ann_vol:.2%} annualised exceeds {limits['max_annualised_volatility']:.0%}. "
            f"The dirham is a managed float inside a narrow band around a 60/40 EUR/USD "
            f"basket and realises roughly 5-7%; this series is not behaving like an "
            f"exchange rate."
        )

    if outlier_share > limits["max_outlier_share"]:
        failures.append(
            f"outliers: {outlier_share:.4%} of days move more than "
            f"{limits['outlier_abs_log_return']:.0%} (limit {limits['max_outlier_share']:.4%}). "
            f"Genuine band widenings are rare events; this rate implies systematic bad prints."
        )

    report["bounce_suspected"] = bool(
        lag1 is not None and lag1 < limits["min_lag1_autocorrelation"]
    )
    report["blocking_failures"] = failures
    report["passed"] = not failures
    return report


def enforce_fx_quality(report: dict, allow_suspect: bool = False) -> dict:
    """
    Turn an FX quality verdict into a stopped run, or an explicit, recorded override.

    Addresses: P1, P4 — this is the release gate. Before it existed, a run could
    complete on a feed the pipeline had already diagnosed as defective: the
    metadata said `bounce_suspected: true` and the artifact shipped anyway. A
    finding nobody is forced to act on is indistinguishable from no finding —
    the same lesson as `require_dividends` (§17.8), one input later.

    `allow_suspect` exists ONLY for tests and synthetic smoke runs, where the
    fixture FX is arbitrary by construction. No production caller may set it,
    and `tests/test_currency.py` inspects the source of `pipeline.py`,
    `clean.py` and `orchestration/assets.py` to prove none does — the gate is
    structurally un-bypassable rather than un-bypassed by convention.

    Args:
        report: Output of `fx_quality_report`.
        allow_suspect: Downgrade a failure to a WARNING. Tests/smoke only.

    Returns:
        The report, annotated with how the gate resolved.

    Raises:
        FXQualitySuspect: if any blocking check failed and `allow_suspect` is False.
    """
    failures = report.get("blocking_failures", [])
    report = {**report, "override_applied": bool(allow_suspect and failures)}

    if not failures:
        log.info("FX quality gate: PASSED (%d checks, no blocking failure).", 4)
        return report

    detail = "\n  - ".join(failures)
    if allow_suspect:
        log.warning(
            "FX quality gate OVERRIDDEN — %d blocking failure(s) downgraded to a "
            "warning because allow_suspect=True. This is valid ONLY for tests and "
            "synthetic smoke runs; any number produced under this override is not "
            "releasable.\n  - %s", len(failures), detail,
        )
        return report

    raise FXQualitySuspect(
        f"FX quality gate FAILED with {len(failures)} blocking failure(s):\n  - {detail}\n\n"
        f"Refusing to produce a {BASE_CURRENCY} valuation from this series. Converting "
        f"with it would inject non-economic variance into every converted return, "
        f"inflate the whole covariance block, and drive every risk-based optimiser to "
        f"conclusions about a quote artifact rather than about markets.\n\n"
        f"Fix the SOURCE (an official Bank Al-Maghrib reference rate), not this "
        f"threshold. `allow_suspect=True` is reserved for tests and synthetic smoke "
        f"runs and must never be set by a production caller."
    )


def validate_fx_gap_structure(
    fx: pd.Series,
    dates: pd.DatetimeIndex,
    ffill_limit: int = 5,
    series: str = FX_SERIES,
) -> dict:
    """
    Prove the FX series has no gap too wide to fill causally over `dates`.

    Addresses: P1, P4 — density alone does not decide usability, and reporting
    it alone would be misleading. `align_fx_to_dates` fills forward with a
    bounded limit, so a 93%-dense series is perfectly usable if the missing 7%
    is scattered and completely unusable if it arrives as one 17-day hole. The
    LONGEST CONSECUTIVE RUN is the number that decides, and it is checked here
    rather than discovered by a failing conversion halfway through a rebuild.

    This also keeps `MIXED_UNIVERSE_START` honest: the constant is re-derived
    from the data on every run, so a refreshed BAM archive that moved the gap
    would fail loudly instead of silently invalidating a hard-coded date.

    Args:
        fx: Validated FX spot series (observed rates only).
        dates: The calendar requiring coverage.
        ffill_limit: Maximum consecutive dates that may inherit an older rate.
        series: Name used in messages.

    Returns:
        Gap statistics, persisted into the conversion metadata.

    Raises:
        FXDataUnavailable: if any gap exceeds `ffill_limit`, naming the window
            and the date the caller should start from instead.
    """
    present = pd.Series(dates.isin(fx.index), index=dates)
    runs, current = [], []
    for day, ok in present.items():
        if ok:
            if current:
                runs.append(current)
            current = []
        else:
            current.append(day)
    if current:
        runs.append(current)

    oversized = [r for r in runs if len(r) > ffill_limit]
    stats = {
        "business_days_required": int(len(dates)),
        "observed": int(present.sum()),
        "density": round(float(present.mean()), 6),
        "n_gaps": len(runs),
        "longest_consecutive_missing": max((len(r) for r in runs), default=0),
        "n_gaps_over_ffill_limit": len(oversized),
        "ffill_limit": ffill_limit,
    }

    if oversized:
        worst = max(oversized, key=len)
        resume = (worst[-1] + pd.offsets.BDay(1)).date()
        raise FXDataUnavailable(
            f"FX {series!r} has {len(oversized)} gap(s) wider than the causal "
            f"forward-fill limit of {ffill_limit} business days over the requested "
            f"window. The worst runs {worst[0].date()} → {worst[-1].date()} "
            f"({len(worst)} business days). Those dates CANNOT be covered without "
            f"inventing a rate, which is not something this pipeline will do. "
            f"Start the universe at {resume} instead, or obtain the missing "
            f"observations from the source."
        )
    return stats


def resolve_currency_policy(
    columns: Sequence[str],
    foreign_assets: Sequence[str] | None = None,
    domestic_assets: Sequence[str] | None = None,
) -> dict:
    """
    Decide a universe's numéraire from what it actually holds.

    Addresses: P1, P4 — the defect this module exists to fix is MIXING
    currencies in one portfolio sum, and only a matrix that HOLDS both kinds of
    asset can have it. Deriving that from the columns rather than from a
    filename or a caller flag matters for two reasons:

      * `etf_2017` is five USD-denominated ETFs and nothing else. It has ONE
        numéraire already, so it has no defect, needs no conversion, and must
        not acquire a dependency on an FX series that does not cover its
        2004-2017 window. Forcing MAD on it would be a reporting choice
        masquerading as a correctness fix.
      * `full_2021` holds four MAD equities beside those five USD ETFs. It is
        the universe that was genuinely broken, and for it FX is mandatory.

    Keying on columns makes the rule self-maintaining: add a BVC name to the
    ETF universe and it becomes mixed, so conversion and the FX dependency turn
    themselves on. Keying on `output_stem` would have left a USD label on a
    matrix that had quietly become mixed — the same class of silent drift as an
    undeclared pipeline input (§17.8).

    Args:
        columns: Asset columns present in the price/returns matrix.
        foreign_assets: Defaults to `schemas.ETF_ASSETS` (USD-denominated).
        domestic_assets: Defaults to `schemas.BVC_ASSETS` (MAD-denominated).

    Returns:
        Policy dict: `base_currency`, `requires_conversion`, `requires_fx`,
        `is_mixed_currency`, `rationale`, and the resolved asset lists. It is
        persisted verbatim into the Silver validation report.

    Raises:
        ValueError: if a column belongs to neither list, or if the matrix is
            empty — an asset whose numéraire is assumed is the trap here.
    """
    foreign = list(ETF_ASSETS if foreign_assets is None else foreign_assets)
    domestic = list(BVC_ASSETS if domestic_assets is None else domestic_assets)

    present_foreign = [c for c in columns if c in foreign]
    present_domestic = [c for c in columns if c in domestic]
    undeclared = [c for c in columns if c not in foreign and c not in domestic]

    if undeclared:
        raise ValueError(
            f"Columns {undeclared} have no declared currency, so this universe's "
            f"numéraire cannot be resolved. Add each to the foreign (USD) or domestic "
            f"(MAD) list before it reaches a portfolio."
        )
    if not present_foreign and not present_domestic:
        raise ValueError("Cannot resolve a currency policy for an empty universe.")

    if present_foreign and present_domestic:
        policy = {
            "base_currency": BASE_CURRENCY,
            "requires_conversion": True,
            "requires_fx": True,
            "is_mixed_currency": True,
            "rationale": (
                f"Mixed universe: {len(present_domestic)} {BASE_CURRENCY}-denominated "
                f"and {len(present_foreign)} {FOREIGN_CURRENCY}-denominated assets. A "
                f"portfolio sum needs one numéraire, so the {FOREIGN_CURRENCY} sleeve "
                f"is converted at the observed {FX_SERIES} spot rate. FX is MANDATORY: "
                f"without it these returns cannot be summed at all."
            ),
        }
    elif present_foreign:
        policy = {
            "base_currency": FOREIGN_CURRENCY,
            "requires_conversion": False,
            "requires_fx": False,
            "is_mixed_currency": False,
            "rationale": (
                f"Single-currency universe: all {len(present_foreign)} assets are "
                f"{FOREIGN_CURRENCY}-denominated. Summing them is already coherent, so "
                f"there is no numéraire defect and no FX dependency. Expressing this "
                f"universe in {BASE_CURRENCY} would be a REPORTING choice about whose "
                f"experience is described, not a correctness fix, and is deliberately "
                f"not made here."
            ),
        }
    else:
        policy = {
            "base_currency": BASE_CURRENCY,
            "requires_conversion": False,
            "requires_fx": False,
            "is_mixed_currency": False,
            "rationale": (
                f"Single-currency universe: all {len(present_domestic)} assets are "
                f"already {BASE_CURRENCY}-denominated."
            ),
        }

    policy["assets_foreign"] = present_foreign
    policy["assets_domestic"] = present_domestic
    return policy


def convert_prices_to_base_currency(
    prices: pd.DataFrame,
    fx_rates: pd.Series,
    base_currency: str = BASE_CURRENCY,
    *,
    foreign_assets: Sequence[str] | None = None,
    domestic_assets: Sequence[str] | None = None,
    ffill_limit: int = 5,
    fx_series_name: str = FX_SERIES,
    quality_thresholds: dict | None = None,
    allow_suspect_fx: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Express a mixed-currency price matrix in a single base currency.

    Addresses: P1, P4 — see module docstring. Conversion happens on PRICE
    LEVELS, before returns are computed, so the resulting log-return is exactly
    `r_foreign + log(fx_t / fx_{t-1})` with no cross-term and no approximation.
    Doing it on returns instead would work too, but converting the level keeps
    one rule ("value every holding in MAD, then difference") rather than two.

    Domestic assets are returned bit-for-bit unchanged, which is what preserves
    the BVC dividend total-return handling in `clean.compute_log_returns`: those
    payments are MAD amounts added to MAD prices and never touch this function.

    Args:
        prices: Wide price matrix, DatetimeIndex × asset columns. Expected to be
            the calendar-aligned frame (no NaN), so the set of dates requiring a
            rate is unambiguous.
        fx_rates: Spot series quoted as `base_currency` per unit of the foreign
            currency — for USDMAD, MAD per USD, so conversion MULTIPLIES.
        base_currency: Reporting numéraire. Only "MAD" is wired today; the
            argument exists so the assumption is visible at every call site.
        foreign_assets: Columns denominated in the foreign currency. Defaults to
            `schemas.ETF_ASSETS`.
        domestic_assets: Columns already in `base_currency`. Defaults to
            `schemas.BVC_ASSETS`. Declared explicitly so that an asset belonging
            to NEITHER list is an error rather than an assumption.
        ffill_limit: Passed to `align_fx_to_dates`.
        fx_series_name: Name used in messages and metadata.
        quality_thresholds: Override `FX_QUALITY_THRESHOLDS`.
        allow_suspect_fx: Downgrade the quality gate to a warning. Tests and
            synthetic smoke runs ONLY — see `enforce_fx_quality`.

    Returns:
        `(converted_prices, metadata)`. The metadata dict is what gets persisted
        into the Silver validation report and the Gold currency manifest.

    Raises:
        FXDataUnavailable: if the FX series is invalid or does not cover the
            price calendar.
        FXQualitySuspect: if the FX series fails a blocking quality check and
            `allow_suspect_fx` is False.
        ValueError: if `prices` carries a column whose currency is undeclared —
            the trap that would otherwise let the next asset inherit the wrong
            numéraire silently.
    """
    if base_currency != BASE_CURRENCY:
        raise ValueError(
            f"base_currency={base_currency!r} is not supported; the pipeline's "
            f"numéraire is {BASE_CURRENCY!r} and the {fx_series_name} series is "
            f"quoted as {FX_QUOTE_CONVENTION}."
        )

    foreign = list(ETF_ASSETS if foreign_assets is None else foreign_assets)
    domestic = list(BVC_ASSETS if domestic_assets is None else domestic_assets)

    undeclared = [c for c in prices.columns if c not in foreign and c not in domestic]
    if undeclared:
        raise ValueError(
            f"Columns {undeclared} have no declared currency. Add each to the "
            f"foreign (converted) or domestic (pass-through) list before it reaches "
            f"a portfolio — an asset whose numéraire is assumed rather than stated "
            f"is precisely the defect this module exists to remove."
        )

    to_convert = [c for c in prices.columns if c in foreign]
    passthrough = [c for c in prices.columns if c in domestic]

    fx = validate_fx_series(fx_rates, series=fx_series_name)
    quality = fx_quality_report(fx, thresholds=quality_thresholds)

    if not to_convert:
        # Not an error: a domestic-only universe genuinely needs no conversion.
        # It IS worth a warning, because silently doing nothing is how a
        # misconfigured asset list would look (§15.13).
        log.warning(
            "No %s-denominated columns in the price matrix — nothing to convert. "
            "Columns present: %s", FOREIGN_CURRENCY, list(prices.columns),
        )
        metadata = _conversion_metadata(
            prices.index, [], passthrough, base_currency, fx_series_name,
            enforce_fx_quality(quality, allow_suspect=allow_suspect_fx), fx,
        )
        return prices.copy(), metadata

    # Gap structure BEFORE alignment: `align_fx_to_dates` would also raise, but
    # only after the fact and without naming the date to restart from. Checking
    # here turns "this failed" into "start here instead".
    gap_stats = validate_fx_gap_structure(
        fx, prices.index, ffill_limit=ffill_limit, series=fx_series_name
    )

    aligned_fx = align_fx_to_dates(
        fx, prices.index, ffill_limit=ffill_limit, series=fx_series_name
    )

    # Observation density is a property of the FX series MEASURED AGAINST the
    # price calendar, so it can only be judged here, after alignment. A feed
    # sparse enough to be mostly inherited passes every per-gap ffill_limit
    # check and is still not an observed exchange rate.
    density = float(aligned_fx.index.isin(fx.index).mean())
    limits = {**FX_QUALITY_THRESHOLDS, **(quality_thresholds or {})}
    quality = {**quality, "observation_density": round(density, 6), "gap_structure": gap_stats}
    if density < limits["min_observation_density"]:
        quality = {
            **quality,
            "blocking_failures": [
                *quality["blocking_failures"],
                f"coverage: only {density:.1%} of the {len(aligned_fx)} required dates "
                f"carry a real observation (limit {limits['min_observation_density']:.0%}); "
                f"the rest inherit an earlier rate. A mostly forward-filled series "
                f"understates realised FX variation.",
            ],
        }
        quality["passed"] = False

    quality = enforce_fx_quality(quality, allow_suspect=allow_suspect_fx)

    converted = prices.copy()
    converted[to_convert] = prices[to_convert].mul(aligned_fx, axis=0)

    log.info(
        "Converted %d %s-denominated column(s) to %s at %s: %s. "
        "%d domestic column(s) passed through unchanged: %s",
        len(to_convert), FOREIGN_CURRENCY, base_currency, fx_series_name,
        to_convert, len(passthrough), passthrough or "none",
    )

    metadata = _conversion_metadata(
        prices.index, to_convert, passthrough, base_currency,
        fx_series_name, quality, fx, aligned_fx=aligned_fx,
    )
    return converted, metadata


def _conversion_metadata(
    index: pd.DatetimeIndex,
    converted: list[str],
    passthrough: list[str],
    base_currency: str,
    fx_series_name: str,
    quality: dict,
    fx: pd.Series,
    aligned_fx: pd.Series | None = None,
) -> dict:
    """Assemble the audit record persisted alongside every converted artifact."""
    record = {
        "base_currency": base_currency,
        "etf_source_currency": FOREIGN_CURRENCY,
        "fx_series": fx_series_name,
        "fx_quote_convention": FX_QUOTE_CONVENTION,
        "hedge_status": HEDGE_STATUS,
        "conversion_coverage": {
            "start": str(index.min().date()),
            "end": str(index.max().date()),
            "n_dates": int(len(index)),
        },
        "n_assets_converted": len(converted),
        "assets_converted": list(converted),
        "assets_passed_through": list(passthrough),
        "fx_observation_coverage": {
            "start": str(fx.index.min().date()),
            "end": str(fx.index.max().date()),
            "n_observations": int(len(fx)),
        },
        "fx_quality": quality,
        "hedging_note": (
            "Unhedged. Realised FX variation is included in the reported MAD "
            "performance; no forward contract, hedge ratio, forward-point model "
            "or rollover cost is modelled."
        ),
    }
    if aligned_fx is not None:
        record["fx_dates_forward_filled"] = int((~aligned_fx.index.isin(fx.index)).sum())
        record["fx_rate_at_coverage_start"] = round(float(aligned_fx.iloc[0]), 6)
        record["fx_rate_at_coverage_end"] = round(float(aligned_fx.iloc[-1]), 6)
    return record
