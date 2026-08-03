"""
telemetry.py — record what a model ACTUALLY did at each rebalance.

Addresses: P4 — a strategy labelled `dcc_garch` that silently fell back to
Ledoit-Wolf on half its rebalances is, on those dates, a Ledoit-Wolf result
wearing a DCC-GARCH label. The fallbacks were always logged at WARNING, but a
WARNING on an unattended run is a message nobody reads, and no committed
artifact stated how often it fired. This module makes the degradation a
FIELD rather than a log line.

The invariant this exists to protect: **no model label may hide degraded or
fallback behaviour.**

WHY A COLLECTOR AND NOT A RETURN VALUE
--------------------------------------
The failure happens deep — inside `dcc_covariance`, inside
`_predict_expected_returns_uncached` — several frames below the `Strategy`
object the engine holds. Threading a report back up would change
`estimate_covariance`, `fit_predict_expected_returns`, every `Strategy.fit`
signature and the ABC itself, for a diagnostic. A collector scoped to one
`fit()` call keeps the seam where it already is: the engine opens a window,
whatever estimators run inside it record what they did, the engine reads the
window shut.

THE MEMOIZATION TRAP, AND WHY IT IS HANDLED HERE
------------------------------------------------
Both estimators are content-addressed cached (`memo`). A cache hit does not
re-run the estimator, so it would not re-emit its fallback event — and the
recorded fallback RATE would fall as the cache warmed, which is the worst
possible failure for a number whose entire job is to be trustworthy. The
caches therefore store the FitRecord alongside the value and re-emit it on
every hit. `tests/test_telemetry.py` pins this: the recorded rate must be
identical with a warm cache and a cold one.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger("telemetry")

# Status vocabulary. Deliberately three values, not a boolean: "the estimator
# ran but warned" is a different fact from "the estimator was replaced", and
# collapsing them would hide the first.
STATUS_OK = "ok"                # ran as requested, no warning
STATUS_DEGRADED = "degraded"    # ran as requested, but warned
STATUS_FALLBACK = "fallback"    # did NOT run as requested; a substitute produced the result

VALID_STATUSES = (STATUS_OK, STATUS_DEGRADED, STATUS_FALLBACK)


@dataclass(frozen=True)
class FitRecord:
    """What one estimator call actually did.

    `model_requested` is what the configuration asked for; `model_effective`
    is what produced the number. When they differ, the label on the result is
    not the whole truth and `fallback_reason` says why.
    """

    model_requested: str
    model_effective: str
    fit_status: str
    n_training_rows: int
    fallback_reason: str | None = None
    convergence_warning: str | None = None

    def __post_init__(self) -> None:
        if self.fit_status not in VALID_STATUSES:
            raise ValueError(
                f"Unknown fit_status {self.fit_status!r}; expected one of {VALID_STATUSES}."
            )
        if self.fit_status == STATUS_FALLBACK and not self.fallback_reason:
            raise ValueError(
                "A fallback must carry a reason. An unexplained substitution is "
                "exactly the opacity this module exists to remove."
            )
        if self.fit_status == STATUS_FALLBACK and self.model_effective == self.model_requested:
            raise ValueError(
                f"fit_status='fallback' but model_effective == model_requested "
                f"({self.model_requested!r}). If nothing was substituted, the status "
                f"is 'degraded', not 'fallback'."
            )

    @property
    def is_fallback(self) -> bool:
        return self.fit_status == STATUS_FALLBACK

    def as_dict(self) -> dict[str, object]:
        return {
            "model_requested": self.model_requested,
            "model_effective": self.model_effective,
            "fit_status": self.fit_status,
            "n_training_rows": int(self.n_training_rows),
            "fallback_reason": self.fallback_reason,
            "convergence_warning": self.convergence_warning,
        }


# Thread-local so a parallel runner cannot interleave two rebalances' records.
# Nothing in this project currently fits strategies in parallel; the isolation
# is here so that adding it later cannot silently corrupt the counts.
_STATE = threading.local()


def _active() -> list[FitRecord] | None:
    return getattr(_STATE, "records", None)


@contextmanager
def collect() -> Iterator[list[FitRecord]]:
    """Open a recording window; yields the list the window fills.

    Nesting is supported and the inner window wins, so a caller that wraps a
    sub-computation gets only its own records. Records are never dropped
    silently: an inner window's contents are copied outward on exit.
    """
    previous = _active()
    records: list[FitRecord] = []
    _STATE.records = records
    try:
        yield records
    finally:
        _STATE.records = previous
        if previous is not None:
            previous.extend(records)


def record(entry: FitRecord) -> None:
    """Note what an estimator did. A no-op outside a `collect()` window.

    Silent outside a window ON PURPOSE: `dcc_covariance` and
    `fit_predict_expected_returns` are called from notebooks, experiments and
    tests that have no interest in telemetry, and a diagnostic that raises in
    those contexts would be worse than the opacity it replaces.
    """
    active = _active()
    if active is not None:
        active.append(entry)


def summarize(records: list[FitRecord], requested_label: str, n_training_rows: int) -> dict:
    """Collapse one rebalance's records into a single reportable row.

    `requested_label` is the strategy's own name — the label a reader sees on
    a chart. `model_effective` downgrades it whenever any estimator inside
    that fit was substituted, so a chart legend and this field cannot disagree
    about what produced the number.
    """
    fallbacks = [r for r in records if r.is_fallback]
    warnings = [r.convergence_warning for r in records if r.convergence_warning]

    if fallbacks:
        status = STATUS_FALLBACK
        # Name the substitution, not just the fact of it: "dcc_garch" telling
        # you it fell back is far less useful than telling you to what.
        effective = f"{requested_label} [via {'+'.join(sorted({r.model_effective for r in fallbacks}))}]"
        reason = "; ".join(dict.fromkeys(r.fallback_reason for r in fallbacks if r.fallback_reason))
    elif warnings:
        status, effective, reason = STATUS_DEGRADED, requested_label, None
    else:
        status, effective, reason = STATUS_OK, requested_label, None

    return {
        "model_requested": requested_label,
        "model_effective": effective,
        "fit_status": status,
        "n_training_rows": int(n_training_rows),
        "fallback_reason": reason,
        "convergence_warning": "; ".join(dict.fromkeys(warnings)) or None,
    }
