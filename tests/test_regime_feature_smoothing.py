"""
test_regime_feature_smoothing.py — The regime feature is smoothed, and this
file says so out loud, with a magnitude.

WHY THIS FILE EXISTS. `regime.predict_regime_posterior_series` calls
`hmmlearn`'s `model.predict_proba`, which returns SMOOTHED posteriors —
gamma_t = P(state_t | x_1...x_T) from the forward-backward algorithm. For
`RegimeConditionalStrategy` that is harmless: it reads only the last row, and
at t = T the smoothed posterior equals the filtered one. For
`ml_signals.attach_regime_feature` it is not. Every historical row's
`REGIME_BULL_PROB` at date t was computed using observations AFTER t, up to
the window end, while at inference the same column is a filtered posterior.
That is an in-window lookahead and a train/serve mismatch: the model trains on
a cleaner version of a feature than the one it is scored with.

WHY IT IS A DIAGNOSTIC AND NOT AN `xfail`. An expected-failure test normalises
a defect — it goes green while the thing it names stays broken, and a reader
learns nothing from a passing suite. These tests PASS by measuring the
dependency and bounding it. If someone later switches to filtered posteriors,
`test_the_regime_feature_depends_on_rows_after_the_row_it_labels` will FAIL,
and its message tells them to delete this file. A test that must be deleted
when a bug is fixed is a more honest marker than one permitted to fail forever.

WHAT THE MEASUREMENT ISOLATES, and why that took two attempts. The obvious
diagnostic — call `attach_regime_feature` on a short window and a long one and
compare — CONFLATES two effects, because that function refits the HMM. A
refit on more data changes the fitted parameters themselves, which moves the
posteriors far more than smoothing does (drift ~0.7 in probability units on
this fixture, and partly just relabelled states). That would have made this
file report a large number for the wrong reason.

The tests below therefore hold the FITTED MODEL FIXED and vary only the length
of the window handed to `predict_proba`. Any difference is then attributable
to the backward pass and nothing else. Measured that way the effect is real
but modest and highly localised: on the fixture here the maximum drift is
~1e-3 in probability units, fewer than 10 of 141 dates move at all, and the
movement is concentrated in the final rows of the window — which is exactly
the signature of forward-backward smoothing, since the backward pass carries
most information where the fewest future observations exist. Old rows are
already pinned by everything that followed them.

WHAT IS AND IS NOT AT RISK. This does NOT inflate out-of-sample results.
Nothing after the rebalance date tau enters the evaluation: the engine slices
every `extras` frame to `:tau` before each fit, and
`tests/test_phase4b_integration.py::TestNoLookahead::
test_future_return_corruption_cannot_change_past_weights` proves corrupting
the future cannot change any past weight for exactly the F7 strategies that
consume this column. That test is the BOUND; this file is the DISCLOSURE.
They are separate because "confined to the training window" and "absent" are
different claims and only one of them is true.

Consequence, stated plainly: the regime feature is EXPLORATORY. If anything
the mismatch degrades live performance, because a tree learns to trust a
variable quieter in training than at inference. See
`docs/EVALUATION_LIMITS.md` section 3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("hmmlearn", reason="the diagnostic needs a real HMM fit")

from ml_signals import build_asset_features, melt_to_panel  # noqa: E402
from regime import (  # noqa: E402
    REGIME_FEATURES,
    fit_hmm,
    predict_regime_posterior_series,
)

N_PERIODS = 200
MIN_REGIME_TRAIN_DAYS = 60
EARLY_ROW = 140

# Below this, a difference is float noise rather than the backward pass.
DRIFT_EPS = 1e-6


def _market_features(n: int = N_PERIODS) -> pd.DataFrame:
    """Two OVERLAPPING regimes — separable, but not trivially so.

    The separation matters in both directions and both were observed while
    building this file. Regimes that are too distinct drive the posteriors to
    exactly 0 and 1, where smoothed and filtered agree by saturation and the
    diagnostic reports zero drift for a reason that has nothing to do with the
    defect. Regimes that are indistinguishable stop the HMM converging at all
    and the code falls back to a neutral 0.5 everywhere, which also compares
    equal. Every feature carries noise for the same reason: constant
    within-regime values make the states perfectly identifiable and saturate
    the posteriors.
    """
    rng = np.random.default_rng(3)
    half = n // 2
    index = pd.bdate_range("2020-01-01", periods=n, name="Date")
    return pd.DataFrame(
        {
            REGIME_FEATURES[0]: np.concatenate([
                rng.normal(0.0008, 0.010, half),
                rng.normal(-0.0008, 0.010, n - half)]),
            REGIME_FEATURES[1]: np.concatenate([
                rng.normal(0.10, 0.03, half),
                rng.normal(0.13, 0.03, n - half)]),
            REGIME_FEATURES[2]: np.concatenate([
                rng.normal(0.30, 0.06, half),
                rng.normal(0.36, 0.06, n - half)]),
        },
        index=index,
    )


def _returns(periods: int = N_PERIODS, assets: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2020-01-01", periods=periods, name="Date")
    return pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(periods, assets)),
        index=index, columns=[f"A{i}" for i in range(assets)],
    )


@pytest.fixture(scope="module")
def smoothing_drift() -> pd.Series:
    """Per-date |smoothed(short window) - smoothed(long window)|, ONE model.

    Holding the fit fixed is the whole point: it removes the refit and
    state-relabelling confounds, so what remains is the backward pass.
    """
    market = _market_features()
    early, late = market.index[EARLY_ROW], market.index[-1]

    fit = fit_hmm(market.loc[:late], min_regime_train_days=MIN_REGIME_TRAIN_DAYS)
    assert fit.converged, (
        "The HMM did not converge on this fixture, so the neutral 0.5 fallback "
        "would make every comparison below trivially equal. Fix the fixture "
        "rather than the assertions."
    )

    short = predict_regime_posterior_series(fit, market.loc[:early])["bull"]
    long_ = predict_regime_posterior_series(fit, market.loc[:late])["bull"]
    shared = short.index.intersection(long_.index)
    assert len(shared) > MIN_REGIME_TRAIN_DAYS, "fixture too short to be meaningful"

    interior = ((short > 0.02) & (short < 0.98)).mean()
    assert interior > 0.0, (
        "Every posterior saturated to 0 or 1. Smoothed and filtered agree by "
        "saturation there, so this fixture cannot detect the defect — reduce "
        "the separation between the two regimes."
    )
    return (short.loc[shared] - long_.loc[shared]).abs()


class TestRegimeFeatureSmoothingIsDisclosed:
    """The defect, measured — not asserted away and not marked xfail."""

    def test_the_regime_feature_depends_on_rows_after_the_row_it_labels(
        self, smoothing_drift
    ):
        assert smoothing_drift.max() > DRIFT_EPS, (
            "With the fitted model held FIXED, the posterior for the same "
            "historical dates did not change when later observations were "
            f"added to the window (max drift {smoothing_drift.max():.2e}).\n\n"
            "If this now holds because the feature was switched to FILTERED "
            "posteriors, the train/serve mismatch is fixed: delete this file, "
            "drop section 3 of docs/EVALUATION_LIMITS.md, and remove the "
            "'exploratory' label from the regime feature.\n\n"
            "If it holds for any other reason, the diagnostic has stopped "
            "measuring what it claims."
        )

    def test_the_dependency_is_concentrated_at_the_end_of_the_window(
        self, smoothing_drift
    ):
        """The signature that identifies this as the backward pass.

        Smoothing revises a date's posterior using what came after it, so the
        revision is largest where least has come after — the newest rows. Old
        rows are already pinned. A drift spread evenly across the window would
        indicate something else entirely, and this assertion is what would
        catch that.
        """
        tail = smoothing_drift.tail(10).mean()
        head = smoothing_drift.head(100).mean()
        assert tail > head, (
            f"Drift is not concentrated at the window end (last-10 mean "
            f"{tail:.2e} vs first-100 mean {head:.2e}). Forward-backward "
            "smoothing revises recent rows most; an evenly-spread difference "
            "means this is measuring something other than the backward pass."
        )

    def test_the_dependency_is_small_and_affects_few_dates(self, smoothing_drift):
        """Bounds the SEVERITY, so the disclosure carries a magnitude.

        The looseness here is deliberate. This pins the order of magnitude —
        a few thousandths of a probability, on a handful of dates — without
        pinning a figure that a library version bump would break. A drift of
        order 0.1 would mean something structural changed and the disclosure
        in docs/EVALUATION_LIMITS.md would need rewriting, not relaxing.
        """
        n_moved = int((smoothing_drift > DRIFT_EPS).sum())
        assert smoothing_drift.max() < 0.05, (
            f"Smoothing drift reached {smoothing_drift.max():.2e}, far above "
            "the ~1e-3 this fixture has shown. The defect may have widened; "
            "re-measure before publishing the section 3 magnitude."
        )
        assert n_moved < len(smoothing_drift) // 4, (
            f"{n_moved} of {len(smoothing_drift)} dates moved, well beyond the "
            "handful expected at the window end. Re-check the localisation "
            "claim in docs/EVALUATION_LIMITS.md section 3."
        )

    def test_the_diagnostic_is_not_an_artefact_of_recomputation(self):
        """A genuinely causal column must NOT drift under the same treatment.

        Without this, the tests above would pass for any column recomputed on
        a longer window — including a perfectly causal one — and would prove
        nothing about smoothing.
        """
        returns = _returns()
        early, late = returns.index[EARLY_ROW], returns.index[-1]

        def panel(frame):
            wide = build_asset_features(
                frame, short_window=5, long_window=10, momentum_windows=[3])
            return melt_to_panel(wide, list(frame.columns))

        first, second = panel(returns.loc[:early]), panel(returns.loc[:late])
        shared = first.index.intersection(second.index)
        assert len(shared) > 0

        pd.testing.assert_frame_equal(
            first.loc[shared].sort_index(), second.loc[shared].sort_index(),
            check_like=True, obj="build_asset_features on a shared window",
        )

    def test_the_mismatch_is_confined_to_the_training_window(self):
        """The bound, kept adjacent to the disclosure.

        The posterior is smoothed WITHIN the window, but the window ends at
        tau. Nothing dated after tau can enter, because the engine slices
        every extras frame before the fit. The executable proof for the
        strategies that consume this column lives in
        `tests/test_phase4b_integration.py::TestNoLookahead`; this keeps the
        two facts side by side so a reader of the disclosure cannot conclude
        the out-of-sample results are contaminated.
        """
        market = _market_features()
        cutoff = market.index[EARLY_ROW]

        corrupted = market.copy()
        corrupted.loc[corrupted.index > cutoff] = 99.0

        clean_fit = fit_hmm(
            market.loc[:cutoff], min_regime_train_days=MIN_REGIME_TRAIN_DAYS)
        corrupt_fit = fit_hmm(
            corrupted.loc[:cutoff], min_regime_train_days=MIN_REGIME_TRAIN_DAYS)

        clean = predict_regime_posterior_series(clean_fit, market.loc[:cutoff])
        corrupt = predict_regime_posterior_series(
            corrupt_fit, corrupted.loc[:cutoff])

        pd.testing.assert_frame_equal(
            clean, corrupt,
            obj="regime posteriors at a rebalance dated before the corruption",
        )
