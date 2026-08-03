"""
purged_kfold.py — Purged, embargoed K-Fold cross-validation (Phase 5).

Addresses: P4 — the single reason this module exists. Ordinary K-Fold leaks
in a time series: a training sample adjacent to a test fold shares
information with it (its label window overlaps the fold, or serial
correlation bleeds across the boundary), so cross-validated hyperparameter
scores are optimistic and the "best" configuration is partly an artifact of
that leak. Purging drops training samples whose label window touches the
test fold; an embargo additionally drops a buffer of samples immediately
after each fold. This is the leakage-free selection tool the whole of
Phase 5 rests on — a purge off-by-one silently reintroduces exactly the bias
the phase exists to remove, so it is built and tested in isolation before
any model consumes it.

Custom implementation (López de Prado 2018, Ch. 7), NOT `mlfinlab` — that
package moved to a restricted/paid model and is a dependency risk
(CLAUDE.md §3.3, §15.11). Subclasses `sklearn`'s `BaseCrossValidator` only
for `.split()`/`.get_n_splits()` API compatibility; the purge/embargo logic
below is hand-written.

Label horizon: F7's labels are strictly one period ahead
(`ml_signals.build_supervised_dataset`: y[t] = return[t → t+1]), so a
training date t leaks into a test fold starting at t_start exactly when
t + label_horizon ≥ t_start. The purge is therefore a `label_horizon`-date
buffer on the left of each fold; the embargo is a separate buffer on the
right for serial correlation. Folds are contiguous in TIME (not shuffled) —
a shuffled fold in a time series is meaningless.

Panel convention: the F7 dataset is a `(Date, ASSET)` MultiIndex panel — many
rows share one date. Purging is done BY DATE (all assets on a purged date are
purged together), never by row, or a single date could land partly in train
and partly in test.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

log = logging.getLogger("purged_kfold")


def _sample_dates(X) -> pd.DatetimeIndex:
    """Per-row dates for X, whether it is a (Date, ASSET) panel or date-indexed."""
    idx = X.index
    if isinstance(idx, pd.MultiIndex):
        if "Date" in (idx.names or []):
            return pd.DatetimeIndex(idx.get_level_values("Date"))
        return pd.DatetimeIndex(idx.get_level_values(0))
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(
            "PurgedKFold needs dated samples: X must have a DatetimeIndex or a "
            "MultiIndex with a 'Date' level (the ml_signals panel convention)."
        )
    return idx


class PurgedKFold(BaseCrossValidator):
    """
    K-Fold with a purge + embargo gap between train and test folds.

    Args:
        n_splits: Number of contiguous, time-ordered test folds (≥ 2).
        embargo_frac: Fraction of the unique-date count to embargo AFTER each
            test fold (guards serial correlation on the right edge). 0.0
            disables the embargo; the left-side purge always applies.
        label_horizon: Forward span of a label, in periods. 1 for F7's
            one-period-ahead returns — the left-side purge width.
    """

    def __init__(self, n_splits: int = 5, embargo_frac: float = 0.01, label_horizon: int = 1) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be ≥ 2, got {n_splits}.")
        if embargo_frac < 0.0:
            raise ValueError(f"embargo_frac must be ≥ 0, got {embargo_frac}.")
        if label_horizon < 0:
            raise ValueError(f"label_horizon must be ≥ 0, got {label_horizon}.")
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac
        self.label_horizon = label_horizon

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Yield (train_idx, test_idx) integer-position arrays per fold.

        Test folds partition the unique dates into `n_splits` contiguous
        blocks. For each fold, training samples are every row whose date is
        NOT in the test block, NOR within `label_horizon` dates before it
        (purge), NOR within the embargo window after it.
        """
        dates = _sample_dates(X)
        unique = pd.DatetimeIndex(np.sort(dates.unique()))
        n_dates = len(unique)
        if n_dates < self.n_splits:
            raise ValueError(
                f"Cannot make {self.n_splits} folds from {n_dates} unique dates."
            )

        embargo = math.ceil(self.embargo_frac * n_dates)
        # Contiguous, near-equal blocks of the DATE axis (np.array_split
        # handles a non-divisible count without dropping dates).
        date_blocks = np.array_split(np.arange(n_dates), self.n_splits)
        all_positions = np.arange(len(dates))
        date_values = dates.values

        for block in date_blocks:
            if len(block) == 0:
                continue
            test_start_i, test_end_i = block[0], block[-1]
            test_start, test_end = unique[test_start_i], unique[test_end_i]

            purge_lo_i = max(0, test_start_i - self.label_horizon)
            embargo_hi_i = min(n_dates - 1, test_end_i + embargo)
            forbidden_lo = unique[purge_lo_i]          # start of left purge buffer
            forbidden_hi = unique[embargo_hi_i]        # end of right embargo buffer

            test_mask = (date_values >= test_start) & (date_values <= test_end)
            # Train excludes the test block AND its purge/embargo buffer.
            forbidden_mask = (date_values >= forbidden_lo) & (date_values <= forbidden_hi)
            train_mask = ~forbidden_mask

            test_idx = all_positions[test_mask]
            train_idx = all_positions[train_mask]
            if len(test_idx) == 0 or len(train_idx) == 0:
                # A fold whose train or test is emptied by the purge/embargo is
                # dropped, so the effective fold count can fall below n_splits.
                # Log it (never silent — CLAUDE.md §13.13) so a caller relying
                # on get_n_splits() is warned that this run yielded fewer.
                log.warning(
                    "PurgedKFold: fold [%s..%s] dropped (train=%d, test=%d) — "
                    "purge/embargo left it empty; effective folds < n_splits. "
                    "Reduce embargo_frac or n_splits, or add history.",
                    test_start.date(), test_end.date(), len(train_idx), len(test_idx),
                )
                continue
            yield train_idx, test_idx


class InsufficientHistory(ValueError):
    """A universe cannot supply the configured walk-forward fold geometry.

    Addresses: P4 — raised loudly rather than silently yielding fewer folds
    (or none). A selection run that quietly degrades to two folds, or to
    zero, would still produce "chosen" hyperparameters, and nothing
    downstream could tell that the protocol had not actually been applied.
    """


class PurgedWalkForwardSplit(BaseCrossValidator):
    """Strictly forward-only (rolling-origin) splits with purge + embargo.

    Addresses: P4 — `PurgedKFold` above trains on every date outside the
    purge/embargo band, INCLUDING dates after the validation fold. That is
    the textbook López de Prado construction and is defensible when labels
    overlap and the question is "does this feature carry signal". It is NOT a
    simulation of sequential portfolio decisions: scoring 2018 with a model
    fitted partly on 2019-2024 answers a question no live allocator can ask.

    This splitter answers the sequential question instead. Every fold
    satisfies, by construction and by assertion:

        train_start <= train_end < embargo_start <= val_start <= val_end
        and val_end < final_test_start   (enforced by the caller's split)

    Geometry. Validation windows tile the date axis forward from the first
    date at which `min_train_dates` of history exist. Fold i validates on
    `val_dates` dates; training is every date up to `train_end`, where

        train_end = val_start - 1 - embargo_dates - label_horizon

    The `label_horizon` term is the purge: a training date t whose label
    spans [t, t + horizon] must not reach into validation. The
    `embargo_dates` term is the separate serial-correlation buffer.

    Mode. `expanding` (default) grows training from the first available date,
    which matches how the production backtest actually retrains. `rolling`
    keeps a fixed-length training window, which is the right choice when
    older regimes are believed harmful; it is offered because that belief is
    testable, not because it is assumed.

    Panel convention. Identical to `PurgedKFold`: grouping is BY DATE, so all
    assets sharing a date always land on the same side of a boundary.

    Args:
        min_train_dates: Unique dates required before the first fold.
        val_dates: Unique dates in each validation window.
        n_splits: Maximum folds to emit. Fewer are emitted only if the date
            axis runs out, which raises rather than degrading silently.
        embargo_dates: Dates dropped between train end and validation start,
            on top of the purge.
        label_horizon: Forward span of a label, in dates. 1 for F7.
        mode: "expanding" or "rolling".
        step_dates: Advance between consecutive validation starts. Defaults
            to `val_dates` (contiguous, non-overlapping validation windows).
    """

    def __init__(
        self,
        min_train_dates: int = 504,
        val_dates: int = 126,
        n_splits: int = 5,
        embargo_dates: int = 5,
        label_horizon: int = 1,
        mode: str = "expanding",
        step_dates: int | None = None,
    ) -> None:
        if min_train_dates < 1:
            raise ValueError(f"min_train_dates must be >= 1, got {min_train_dates}.")
        if val_dates < 1:
            raise ValueError(f"val_dates must be >= 1, got {val_dates}.")
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
        if embargo_dates < 0:
            raise ValueError(f"embargo_dates must be >= 0, got {embargo_dates}.")
        if label_horizon < 0:
            raise ValueError(f"label_horizon must be >= 0, got {label_horizon}.")
        if mode not in ("expanding", "rolling"):
            raise ValueError(f"mode must be 'expanding' or 'rolling', got {mode!r}.")
        self.min_train_dates = min_train_dates
        self.val_dates = val_dates
        self.n_splits = n_splits
        self.embargo_dates = embargo_dates
        self.label_horizon = label_horizon
        self.mode = mode
        self.step_dates = step_dates if step_dates is not None else val_dates

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _fold_geometry(self, n_dates: int) -> list[tuple[int, int, int, int]]:
        """Return (train_lo, train_hi, val_lo, val_hi) date positions per fold.

        Separated from `split` so the audit artifact and the tests can inspect
        the geometry without materialising row indices.
        """
        gap = self.embargo_dates + self.label_horizon
        first_val = self.min_train_dates + gap
        if first_val + self.val_dates > n_dates:
            raise InsufficientHistory(
                f"Need at least {first_val + self.val_dates} unique dates for one fold "
                f"(min_train_dates={self.min_train_dates} + embargo={self.embargo_dates} "
                f"+ label_horizon={self.label_horizon} + val_dates={self.val_dates}), "
                f"but only {n_dates} are available. Reduce the windows or supply more "
                f"history — this run would otherwise select hyperparameters under a "
                f"protocol that was never actually applied."
            )

        folds: list[tuple[int, int, int, int]] = []
        val_lo = first_val
        while len(folds) < self.n_splits and val_lo + self.val_dates <= n_dates:
            val_hi = val_lo + self.val_dates - 1
            train_hi = val_lo - gap - 1
            train_lo = 0 if self.mode == "expanding" else max(
                0, train_hi - self.min_train_dates + 1
            )
            if train_hi >= train_lo:
                folds.append((train_lo, train_hi, val_lo, val_hi))
            val_lo += self.step_dates

        if not folds:
            raise InsufficientHistory(
                f"No forward fold could be built from {n_dates} unique dates."
            )
        if len(folds) < self.n_splits:
            log.warning(
                "PurgedWalkForwardSplit: %d of %d requested folds fit in %d dates; "
                "the rest would run past the end of the selection window.",
                len(folds), self.n_splits, n_dates,
            )
        return folds

    def split(self, X, y=None, groups=None):
        """Yield (train_idx, val_idx) integer positions, forward-only."""
        dates = _sample_dates(X)
        unique = pd.DatetimeIndex(np.sort(dates.unique()))
        values = dates.values
        positions = np.arange(len(dates))

        for train_lo, train_hi, val_lo, val_hi in self._fold_geometry(len(unique)):
            train_mask = (values >= unique[train_lo].to_datetime64()) & (
                values <= unique[train_hi].to_datetime64()
            )
            val_mask = (values >= unique[val_lo].to_datetime64()) & (
                values <= unique[val_hi].to_datetime64()
            )
            train_idx, val_idx = positions[train_mask], positions[val_mask]
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise InsufficientHistory(
                    f"Fold [{unique[val_lo].date()}..{unique[val_hi].date()}] has "
                    f"train={len(train_idx)} val={len(val_idx)} rows."
                )
            # The invariant this class exists for, asserted rather than assumed:
            # nothing in train may be dated at or after validation starts.
            assert dates[train_idx].max() < dates[val_idx].min(), (
                "forward-only violated: a training date is not strictly before "
                "the validation window"
            )
            yield train_idx, val_idx

    def describe(self, X) -> list[dict]:
        """Fold geometry as plain dicts, for the audit artifact.

        Addresses: P4 — the protocol claim ("training never post-dates
        validation") is only credible if the realised windows are published
        alongside the results, not merely asserted in a docstring.
        """
        dates = _sample_dates(X)
        unique = pd.DatetimeIndex(np.sort(dates.unique()))
        values = dates.values
        out = []
        for i, (t_lo, t_hi, v_lo, v_hi) in enumerate(self._fold_geometry(len(unique)), start=1):
            train_mask = (values >= unique[t_lo].to_datetime64()) & (
                values <= unique[t_hi].to_datetime64()
            )
            val_mask = (values >= unique[v_lo].to_datetime64()) & (
                values <= unique[v_hi].to_datetime64()
            )
            out.append({
                "fold": i,
                "mode": self.mode,
                "train_start": str(unique[t_lo].date()),
                "train_end": str(unique[t_hi].date()),
                "embargo_start": str(unique[min(t_hi + 1, len(unique) - 1)].date()),
                "embargo_end": str(unique[max(v_lo - 1, 0)].date()),
                "val_start": str(unique[v_lo].date()),
                "val_end": str(unique[v_hi].date()),
                "n_train_rows": int(train_mask.sum()),
                "n_val_rows": int(val_mask.sum()),
                "n_train_dates": int(t_hi - t_lo + 1),
                "n_val_dates": int(v_hi - v_lo + 1),
                "embargo_dates": self.embargo_dates,
                "label_horizon": self.label_horizon,
            })
        return out
