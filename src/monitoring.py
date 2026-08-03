"""
monitoring.py — offline, on-demand drift and health metrics.

Addresses: P2, P4 — non-stationarity is the project's second structural problem
(P2), so "have the inputs moved since the reference period?" is a question the
system must be able to answer. P4 because an evaluation that cannot be repeated
against a fixed reference is not an evaluation.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is an **offline, on-demand validation tool**. It compares a supplied
evaluation window against a **versioned reference distribution** stored in
`data/gold/monitoring_baseline.json`.

There is **no live schedule**. Drift alerts, retraining triggers, and incident
response are **not operational**. The Dagster schedule remains stopped, and
restarting it is gated on the four torn-artifact preconditions in
`docs/MODEL_GOVERNANCE.md` §8. Nothing here restarts it, and nothing here may
be described as live monitoring.

**Monitoring emits warnings. It never alters model behaviour.** No function in
this module writes a weight, refits an estimator, or changes an artifact any
strategy reads.

WHY A FIXED REFERENCE
---------------------
The reference window is pinned to the Phase 2/3 training period and stored in a
versioned artifact. Comparing against a moving trailing baseline would make
drift undetectable exactly when it matters: a slow regime shift moves the
baseline with the data, and the metric reports "no change" while the world
changes underneath it.

ON THE THRESHOLDS
-----------------
The PSI bands below (0.10 / 0.25) are the **conventional** credit-risk
thresholds. They are not derived from this project's data and carry no
statistical guarantee here. They are declared so a reader can disagree with a
specific number rather than with an unstated judgement.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("monitoring")

# Conventional PSI interpretation bands (see module docstring — conventional,
# not derived).
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

# Floor for empty bins. Without it a category present in one window and absent
# from the other sends PSI to infinity, which reports as catastrophic drift
# when the real finding is "one bin has no observations".
_EPSILON = 1e-6

DEFAULT_BINS = 10


def _as_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def population_stability_index(
    reference: Iterable[float],
    evaluation: Iterable[float],
    n_bins: int = DEFAULT_BINS,
) -> float:
    """PSI of `evaluation` against `reference`, using reference quantile bins.

    Addresses: P2 — the standard shift statistic,
    ``Σ (e_i − r_i) · ln(e_i / r_i)`` over matched bins.

    Bin edges come from the REFERENCE quantiles, never from the pooled data:
    binning on the union would let the evaluation window redefine the bins it
    is being judged against, which suppresses exactly the shift being looked
    for. Returns 0.0 when either window is empty or the reference is constant —
    there is no shift to measure, and a NaN here would propagate silently into
    a health report.
    """
    ref, ev = _as_array(reference), _as_array(evaluation)
    if ref.size == 0 or ev.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    ev_share = np.histogram(ev, bins=edges)[0] / ev.size
    ref_share = np.clip(ref_share, _EPSILON, None)
    ev_share = np.clip(ev_share, _EPSILON, None)
    return float(np.sum((ev_share - ref_share) * np.log(ev_share / ref_share)))


def interpret_psi(value: float) -> str:
    """Conventional band for a PSI value — see the module docstring."""
    if value < PSI_MODERATE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate_shift"
    return "significant_shift"


def categorical_shift(
    reference: Sequence[str], evaluation: Sequence[str]
) -> dict[str, object]:
    """Distribution shift over labels — used for the regime timeline.

    Addresses: P3 — a regime detector spending materially more time in its
    lower-mean-return state than in the reference period is the single most
    interpretable early signal this system can emit.
    """
    ref = pd.Series(list(reference), dtype="object")
    ev = pd.Series(list(evaluation), dtype="object")
    labels = sorted(set(ref.dropna()) | set(ev.dropna()))
    if not labels or ref.empty or ev.empty:
        return {"labels": labels, "reference_share": {}, "evaluation_share": {},
                "psi": 0.0, "interpretation": "stable"}

    ref_share = {k: float((ref == k).mean()) for k in labels}
    ev_share = {k: float((ev == k).mean()) for k in labels}
    r = np.clip(np.array([ref_share[k] for k in labels]), _EPSILON, None)
    e = np.clip(np.array([ev_share[k] for k in labels]), _EPSILON, None)
    psi = float(np.sum((e - r) * np.log(e / r)))
    return {
        "labels": labels,
        "reference_share": ref_share,
        "evaluation_share": ev_share,
        "psi": psi,
        "interpretation": interpret_psi(psi),
    }


def feature_missingness(frame: pd.DataFrame) -> dict[str, float]:
    """Fraction missing per column.

    Addresses: P4 — `ml_features_etf.parquet` carries 3,659 LEADING NaN after
    the deep-window change, and nothing consuming it noticed for two days. A
    missingness number that is REPORTED cannot repeat that.
    """
    if frame.empty:
        return {}
    return {str(c): float(frame[c].isna().mean()) for c in frame.columns}


def allocation_concentration(weights: pd.Series) -> dict[str, float]:
    """Herfindahl index, effective breadth, and the largest position.

    Addresses: P1 — concentration is the visible symptom of estimation error
    in a mean-variance optimizer, and the reason the weight cap exists.
    `effective_n` is 1/HHI: the number of equally-weighted positions carrying
    the same concentration, which is easier to reason about than HHI itself.
    """
    w = weights.to_numpy(dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0 or w.sum() <= 0:
        return {"herfindahl": 0.0, "effective_n": 0.0, "max_weight": 0.0}
    w = w / w.sum()
    hhi = float(np.sum(w**2))
    return {
        "herfindahl": hhi,
        "effective_n": float(1.0 / hhi) if hhi > 0 else 0.0,
        "max_weight": float(w.max()),
    }


def cap_binding_rate(
    weights_by_date: pd.DataFrame, max_weight: float, tolerance: float = 1e-6
) -> dict[str, float]:
    """How often the weight cap is the binding constraint.

    Addresses: P1 — on a small universe the cap can determine the allocation
    outright, in which case a change in the covariance model cannot express
    itself. A rise in this rate means the optimizer has progressively less room
    to act on any view at all, which is a health signal about the SYSTEM rather
    than about the market.

    Args:
        weights_by_date: long frame with `Date`, `asset`, `weight`.
        max_weight: the per-asset cap in force.
        tolerance: SLSQP converges to the boundary, it does not land on it.
    """
    if weights_by_date.empty:
        return {"position_rate": 0.0, "date_rate": 0.0, "mean_positions_at_cap": 0.0}
    at_cap = weights_by_date["weight"] >= max_weight - tolerance
    per_date = weights_by_date.assign(at_cap=at_cap).groupby("Date")["at_cap"].sum()
    return {
        "position_rate": float(at_cap.mean()),
        "date_rate": float((per_date > 0).mean()),
        "mean_positions_at_cap": float(per_date.mean()),
    }


def turnover_summary(weights_by_date: pd.DataFrame) -> dict[str, float]:
    """Turnover between consecutive rebalances: ``Σ|w_t − w_{t-1}|``.

    Addresses: P4 — turnover is where a good gross signal becomes a bad net
    one. Phase 4B's whole finding was a strategy with the best gross Sharpe of
    the comparison losing it to trading costs, so a turnover distribution that
    drifts upward matters even when returns look unchanged.
    """
    if weights_by_date.empty:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "n_rebalances": 0}
    wide = (
        weights_by_date.pivot_table(index="Date", columns="asset", values="weight")
        .sort_index()
        .fillna(0.0)
    )
    if len(wide) < 2:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "n_rebalances": int(len(wide))}
    turnover = wide.diff().abs().sum(axis=1).iloc[1:]
    return {
        "mean": float(turnover.mean()),
        "median": float(turnover.median()),
        "max": float(turnover.max()),
        "n_rebalances": int(len(wide)),
    }


def fallback_rate(regime_frame: pd.DataFrame) -> dict[str, float]:
    """Share of rebalances where the regime model did NOT converge.

    Addresses: P2, P4 — a non-converged fit is not an error, it is a
    documented degradation: the neutral posterior resolves to the defensive
    sub-strategy. But a RISING fallback rate means the system is increasingly
    defaulting rather than deciding, and that is invisible in returns.
    """
    if regime_frame.empty or "converged" not in regime_frame:
        return {"fallback_rate": 0.0, "n_rebalances": 0}
    converged = regime_frame["converged"].astype(bool)
    return {
        "fallback_rate": float((~converged).mean()),
        "n_rebalances": int(len(converged)),
    }


def compare_distributions(
    reference: Mapping[str, Sequence[float]],
    evaluation: Mapping[str, Sequence[float]],
    n_bins: int = DEFAULT_BINS,
) -> dict[str, dict]:
    """PSI per key, for any two matched collections of numeric series."""
    report: dict[str, dict] = {}
    for name in sorted(set(reference) & set(evaluation)):
        psi = population_stability_index(reference[name], evaluation[name], n_bins)
        report[name] = {"psi": psi, "interpretation": interpret_psi(psi)}
    only_reference = sorted(set(reference) - set(evaluation))
    only_evaluation = sorted(set(evaluation) - set(reference))
    if only_reference or only_evaluation:
        # A feature that exists on one side only is a schema change, not drift.
        # Silently intersecting would hide it, so it is reported as its own
        # finding rather than folded into the PSI table.
        report["__schema_mismatch__"] = {
            "missing_from_evaluation": only_reference,
            "unexpected_in_evaluation": only_evaluation,
        }
    return report


def summarize_reference(values: Iterable[float], n_bins: int = DEFAULT_BINS) -> dict:
    """Compact, storable description of a reference distribution.

    Stores quantile EDGES rather than raw observations: the baseline artifact
    must be committable and reviewable, and it must not become a second copy of
    the market data (which is licence-restricted and DVC-managed).
    """
    array = _as_array(values)
    if array.size == 0:
        return {"count": 0, "edges": [], "mean": None, "std": None}
    edges = np.unique(np.quantile(array, np.linspace(0, 1, n_bins + 1)))
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "edges": [float(e) for e in edges],
    }


def psi_from_reference_summary(
    summary: Mapping[str, object], evaluation: Iterable[float]
) -> float:
    """PSI against a STORED reference summary rather than raw reference data.

    This is what makes the baseline artifact self-sufficient: a reviewer can
    re-run the comparison from the committed JSON without holding the original
    training window.
    """
    edges = list(summary.get("edges") or [])
    ev = _as_array(evaluation)
    if len(edges) < 2 or ev.size == 0 or not summary.get("count"):
        return 0.0
    bounded = np.array(edges, dtype=float)
    bounded[0], bounded[-1] = -np.inf, np.inf
    # The reference share per bin is uniform BY CONSTRUCTION: the edges are
    # reference quantiles, so each interior bin holds ~1/n of the reference.
    n_bins = len(bounded) - 1
    ref_share = np.clip(np.full(n_bins, 1.0 / n_bins), _EPSILON, None)
    ev_share = np.clip(np.histogram(ev, bins=bounded)[0] / ev.size, _EPSILON, None)
    return float(np.sum((ev_share - ref_share) * np.log(ev_share / ref_share)))


def build_warnings(report: Mapping[str, object]) -> list[dict[str, str]]:
    """Turn a comparison report into explicit, human-readable warnings.

    Warnings are the ONLY output of this module that a reader should act on,
    and acting means investigating — never an automatic retrain, and never a
    change of allocation. Nothing consumes this list programmatically.
    """
    warnings: list[dict[str, str]] = []

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            interpretation = node.get("interpretation")
            if interpretation in ("moderate_shift", "significant_shift"):
                warnings.append({
                    "metric": path,
                    "severity": "warning" if interpretation == "moderate_shift" else "alert",
                    "detail": f"PSI = {node.get('psi'):.4f} ({interpretation})",
                })
            if node.get("missing_from_evaluation") or node.get("unexpected_in_evaluation"):
                warnings.append({
                    "metric": path,
                    "severity": "alert",
                    "detail": (
                        f"schema mismatch — missing "
                        f"{node.get('missing_from_evaluation')}, unexpected "
                        f"{node.get('unexpected_in_evaluation')}"
                    ),
                })
            for key, value in node.items():
                if isinstance(value, dict):
                    walk(value, f"{path}.{key}" if path else str(key))

    walk(report, "")
    return sorted(warnings, key=lambda w: (w["severity"] != "alert", w["metric"]))
