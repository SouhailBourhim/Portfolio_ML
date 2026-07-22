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
