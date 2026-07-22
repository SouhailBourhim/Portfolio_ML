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
