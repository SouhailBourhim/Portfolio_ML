"""
test_purged_kfold.py — The Phase 5 leakage gate.

Analogous to tests/test_backtest.py::TestNoLookahead for Phase 2: nothing in
Phase 5 may consume PurgedKFold until this suite is green, because a purge
off-by-one silently reintroduces exactly the cross-validation leakage the
whole phase exists to eliminate. Tests are named after the rule they lock in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purged_kfold import PurgedKFold, _sample_dates


def _panel(n_dates: int = 60, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    """A (Date, ASSET) MultiIndex panel like ml_signals.build_supervised_dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates, name="Date")
    assets = [f"A{i}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product([dates, assets], names=["Date", "ASSET"])
    return pd.DataFrame(rng.normal(size=(len(index), 3)), index=index,
                        columns=["f0", "f1", "f2"])


class TestPurgeGate:
    def test_no_train_sample_within_purge_and_embargo_of_any_test_fold(self):
        """THE load-bearing test. For every fold, no training DATE may fall in
        [test_start − label_horizon, test_end + embargo] — the exact window a
        leaked sample would occupy."""
        panel = _panel(n_dates=80)
        dates = _sample_dates(panel)
        unique = pd.DatetimeIndex(np.sort(dates.unique()))
        horizon = 1
        cv = PurgedKFold(n_splits=5, embargo_frac=0.05, label_horizon=horizon)
        embargo = int(np.ceil(0.05 * len(unique)))

        for train_idx, test_idx in cv.split(panel):
            train_dates = dates[train_idx]
            test_dates = dates[test_idx]
            t_start, t_end = test_dates.min(), test_dates.max()

            start_pos = unique.get_loc(t_start)
            end_pos = unique.get_loc(t_end)
            lo = unique[max(0, start_pos - horizon)]
            hi = unique[min(len(unique) - 1, end_pos + embargo)]

            # No training sample anywhere in the forbidden buffer (which
            # includes the test span itself).
            assert not ((train_dates >= lo) & (train_dates <= hi)).any()

    def test_train_and_test_never_share_a_date(self):
        panel = _panel()
        cv = PurgedKFold(n_splits=4, embargo_frac=0.0)
        dates = _sample_dates(panel)
        for train_idx, test_idx in cv.split(panel):
            assert set(dates[train_idx]).isdisjoint(set(dates[test_idx]))

    def test_a_date_is_never_split_across_train_and_test(self):
        """Panel rows share dates; purging by row (not date) would put one
        date's assets partly in train and partly in test — the subtle bug."""
        panel = _panel(n_assets=5)
        cv = PurgedKFold(n_splits=4, embargo_frac=0.02)
        dates = _sample_dates(panel)
        for train_idx, test_idx in cv.split(panel):
            train_d, test_d = set(dates[train_idx]), set(dates[test_idx])
            assert train_d.isdisjoint(test_d)


class TestFoldStructure:
    def test_test_folds_partition_every_date_exactly_once(self):
        panel = _panel(n_dates=50)
        cv = PurgedKFold(n_splits=5, embargo_frac=0.0)
        dates = _sample_dates(panel)
        unique = set(dates.unique())
        seen: set = set()
        for _, test_idx in cv.split(panel):
            fold_dates = set(dates[test_idx])
            assert seen.isdisjoint(fold_dates), "test folds must be disjoint in time"
            seen |= fold_dates
        assert seen == unique, "every date must appear in exactly one test fold"

    def test_get_n_splits_matches_yielded_folds(self):
        panel = _panel()
        cv = PurgedKFold(n_splits=6, embargo_frac=0.01)
        assert cv.get_n_splits() == 6
        assert sum(1 for _ in cv.split(panel)) == 6

    def test_larger_embargo_never_leaves_more_training_data(self):
        """The embargo can only REMOVE training samples near a fold; increasing
        it must weakly shrink the training set, never grow it."""
        panel = _panel(n_dates=100)
        small = PurgedKFold(n_splits=5, embargo_frac=0.0)
        large = PurgedKFold(n_splits=5, embargo_frac=0.10)
        small_sizes = [len(tr) for tr, _ in small.split(panel)]
        large_sizes = [len(tr) for tr, _ in large.split(panel)]
        for s, l in zip(small_sizes, large_sizes):
            assert l <= s

    def test_indices_are_positional_and_in_range(self):
        panel = _panel(n_dates=40)
        cv = PurgedKFold(n_splits=4, embargo_frac=0.02)
        n = len(panel)
        for train_idx, test_idx in cv.split(panel):
            assert train_idx.dtype.kind in "iu" and test_idx.dtype.kind in "iu"
            assert train_idx.max() < n and test_idx.max() < n
            assert train_idx.min() >= 0 and test_idx.min() >= 0


class TestInputHandling:
    def test_accepts_a_plain_datetime_indexed_frame(self):
        dates = pd.bdate_range("2021-01-04", periods=40, name="Date")
        X = pd.DataFrame(np.random.default_rng(1).normal(size=(40, 2)), index=dates)
        cv = PurgedKFold(n_splits=4, embargo_frac=0.0)
        folds = list(cv.split(X))
        assert len(folds) == 4

    def test_rejects_a_non_dated_index(self):
        X = pd.DataFrame(np.zeros((10, 2)))  # RangeIndex
        with pytest.raises(TypeError, match="dated samples"):
            list(PurgedKFold(n_splits=2).split(X))

    def test_rejects_fewer_dates_than_folds(self):
        panel = _panel(n_dates=3)
        with pytest.raises(ValueError, match="Cannot make"):
            list(PurgedKFold(n_splits=5).split(panel))

    @pytest.mark.parametrize("bad", [{"n_splits": 1}, {"embargo_frac": -0.1}, {"label_horizon": -1}])
    def test_invalid_constructor_args_raise(self, bad):
        with pytest.raises(ValueError):
            PurgedKFold(**bad)


# ═══════════════════════════════════════════════════════════════════════════
#  PurgedWalkForwardSplit — the Phase 2.1 forward-only gate.
#
#  PurgedKFold above is retained and still tested; these lock in the property
#  it deliberately does NOT have. Each test is named after the rule it pins.
# ═══════════════════════════════════════════════════════════════════════════
from purged_kfold import InsufficientHistory, PurgedWalkForwardSplit  # noqa: E402


class TestForwardOnlyGate:
    def test_no_training_date_is_at_or_after_its_validation_window(self):
        """THE load-bearing test — the entire reason this splitter exists.

        PurgedKFold fails this by construction: it trains on every date
        outside the purge band, including the future.
        """
        panel = _panel(n_dates=400)
        dates = _sample_dates(panel)
        cv = PurgedWalkForwardSplit(min_train_dates=120, val_dates=40,
                                    n_splits=4, embargo_dates=5)
        folds = 0
        for train_idx, val_idx in cv.split(panel):
            folds += 1
            assert dates[train_idx].max() < dates[val_idx].min(), (
                "a training date is not strictly before the validation window"
            )
        assert folds >= 2, "expected several folds from this geometry"

    def test_the_old_kfold_would_fail_the_same_check(self):
        """Documents WHY the new splitter exists, rather than asserting it.

        If this ever stops failing, PurgedKFold's semantics changed and the
        justification for having two splitters needs revisiting.
        """
        panel = _panel(n_dates=400)
        dates = _sample_dates(panel)
        trains_on_future = any(
            dates[tr].max() > dates[te].min()
            for tr, te in PurgedKFold(n_splits=4, embargo_frac=0.02).split(panel)
        )
        assert trains_on_future, "PurgedKFold no longer trains on post-validation dates"

    def test_a_date_is_never_split_across_train_and_validation(self):
        """Panel rows share dates; a boundary must never cut through one."""
        panel = _panel(n_dates=300, n_assets=5)
        dates = _sample_dates(panel)
        cv = PurgedWalkForwardSplit(min_train_dates=100, val_dates=30, n_splits=3)
        for train_idx, val_idx in cv.split(panel):
            assert not (set(dates[train_idx]) & set(dates[val_idx])), (
                "a single date landed in both train and validation"
            )

    def test_embargo_and_purge_gap_is_respected(self):
        """The gap between train end and validation start is at least
        embargo_dates + label_horizon unique dates."""
        panel = _panel(n_dates=400)
        dates = _sample_dates(panel)
        unique = pd.DatetimeIndex(np.sort(dates.unique()))
        embargo, horizon = 7, 1
        cv = PurgedWalkForwardSplit(min_train_dates=120, val_dates=40, n_splits=3,
                                    embargo_dates=embargo, label_horizon=horizon)
        for train_idx, val_idx in cv.split(panel):
            gap = (unique.get_loc(dates[val_idx].min())
                   - unique.get_loc(dates[train_idx].max()))
            assert gap >= embargo + horizon, f"gap {gap} < {embargo + horizon}"

    def test_insufficient_history_raises_rather_than_yielding_fewer_folds(self):
        """A silently degraded protocol still emits 'chosen' hyperparameters
        and nothing downstream could tell it was never applied."""
        panel = _panel(n_dates=50)
        cv = PurgedWalkForwardSplit(min_train_dates=400, val_dates=60, n_splits=3)
        with pytest.raises(InsufficientHistory, match="unique dates"):
            list(cv.split(panel))

    def test_expanding_mode_grows_the_training_window(self):
        panel = _panel(n_dates=400)
        dates = _sample_dates(panel)
        cv = PurgedWalkForwardSplit(min_train_dates=100, val_dates=40, n_splits=4,
                                    mode="expanding")
        sizes = [len(set(dates[tr])) for tr, _ in cv.split(panel)]
        assert sizes == sorted(sizes) and sizes[-1] > sizes[0]

    def test_rolling_mode_keeps_the_training_window_bounded(self):
        panel = _panel(n_dates=400)
        dates = _sample_dates(panel)
        cv = PurgedWalkForwardSplit(min_train_dates=100, val_dates=40, n_splits=4,
                                    mode="rolling")
        sizes = [len(set(dates[tr])) for tr, _ in cv.split(panel)]
        assert max(sizes) <= 100, f"rolling window grew to {max(sizes)}"

    def test_describe_matches_the_folds_split_actually_yields(self):
        """The audit artifact must report the geometry that was really used."""
        panel = _panel(n_dates=400)
        dates = _sample_dates(panel)
        cv = PurgedWalkForwardSplit(min_train_dates=120, val_dates=40, n_splits=3)
        described = cv.describe(panel)
        actual = list(cv.split(panel))
        assert len(described) == len(actual)
        for d, (tr, va) in zip(described, actual):
            assert d["train_end"] == str(dates[tr].max().date())
            assert d["val_start"] == str(dates[va].min().date())
            assert d["val_end"] == str(dates[va].max().date())
            assert d["n_val_rows"] == len(va)

    def test_validation_never_reaches_the_frozen_test_segment(self):
        """Selection sees only train+validation, so every validation date must
        precede the caller's frozen test start."""
        panel = _panel(n_dates=400)
        dates = _sample_dates(panel)
        unique = pd.DatetimeIndex(np.sort(dates.unique()))
        test_start = unique[300]
        selection_panel = panel[dates < test_start]
        cv = PurgedWalkForwardSplit(min_train_dates=120, val_dates=40, n_splits=3)
        for _, val_idx in cv.split(selection_panel):
            assert _sample_dates(selection_panel)[val_idx].max() < test_start
