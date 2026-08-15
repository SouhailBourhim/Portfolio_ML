"""
test_global_2004_q1.py — Q1's protocol constraints, made executable.

Q1 is run ONCE from a clean committed revision. That makes its correctness a
pre-run concern: there is no "run it again and see" if the artifact turns out
malformed, because a rerun is permitted for a technical defect but never for
an unfavourable result. So the machinery is tested on synthetic data here, and
the artifact's SHAPE is asserted separately once it exists.

Four protocol rules are locked in:

  1. TWO INDEPENDENT VERDICTS, never collapsed. `economically_material` and
     `statistical_evidence` answer different questions — is the difference big
     enough to matter, and can the data distinguish it from zero. A single
     `wins` field would hide which one was actually met, which is precisely the
     ambiguity this project has spent its history removing.
  2. NO MULTIPLE-TESTING CORRECTION. Q1 is one pre-specified hypothesis; DSR,
     Reality Check and SPA belong to Q2's search.
  3. NO SILENT ALIGNMENT. Mismatched indexes are a caller error, because
     repairing them would change which days are compared.
  4. ABSOLUTE materiality. 0.05 Sharpe POINTS, not 5%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_global_2004_q1 as q1  # noqa: E402
from metrics import paired_block_bootstrap  # noqa: E402
from strategies import MaxSharpe, RegimeConditionalStrategy  # noqa: E402
from utils import load_params  # noqa: E402

ARTIFACT = ROOT / "data" / "gold" / "global_2004_q1_results.json"


def _artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip("Q1 artifact absent — run src/run_global_2004_q1.py.")
    return json.loads(ARTIFACT.read_text())


# ── The comparison is fixed by the protocol, not by configuration ────────────

class TestTheComparisonIsPreSpecified:
    def test_candidate_and_comparator_are_the_frozen_pair(self):
        assert q1.CANDIDATE == "regime_conditional"
        assert q1.COMPARATOR == "max_sharpe"

    def test_materiality_margin_is_absolute_sharpe_points(self):
        """0.05 POINTS. A relative form was proposed and overruled: it is
        undefined when the comparator is at or below zero and unstable near it."""
        assert q1.MATERIAL_MARGIN == 0.05

    def test_build_pair_returns_exactly_those_two_strategies(self):
        candidate, comparator = q1._build_pair(load_params())
        assert isinstance(candidate, RegimeConditionalStrategy)
        assert isinstance(comparator, MaxSharpe)

    def test_build_pair_returns_fresh_instances_each_call(self):
        """RegimeConditionalStrategy carries a per-instance regime_log; sharing
        one across runs would blend unrelated histories into one diagnostic."""
        first, _ = q1._build_pair(load_params())
        second, _ = q1._build_pair(load_params())
        assert first is not second
        assert first.regime_log is not second.regime_log


# ── No silent alignment ──────────────────────────────────────────────────────

class TestIdenticalIndexesAreRequired:
    def test_mismatched_indexes_raise_rather_than_align(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 0.01, 300), index=idx)
        b = pd.Series(rng.normal(0, 0.01, 299), index=idx[:-1])

        with pytest.raises(ValueError, match="identical date indexes"):
            paired_block_bootstrap(a, b, block_len=21, n_boot=50, seed=0)

    def test_nan_bearing_series_raise(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 0.01, 300), index=idx)
        b = a.copy()
        b.iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN-free"):
            paired_block_bootstrap(a, b, block_len=21, n_boot=50, seed=0)


# ── The bootstrap is seeded, paired and null-centred ─────────────────────────

class TestPairedInferenceContract:
    def _pair(self, n=500, seed=1):
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(seed)
        shared = rng.normal(0, 0.009, n)          # common market factor
        a = pd.Series(shared + rng.normal(0.0004, 0.003, n), index=idx)
        b = pd.Series(shared + rng.normal(0.0000, 0.003, n), index=idx)
        return a, b

    def test_same_seed_reproduces_exactly(self):
        a, b = self._pair()
        kw = dict(block_len=21, n_boot=300, alpha=0.10, seed=0)
        assert paired_block_bootstrap(a, b, **kw) == paired_block_bootstrap(a, b, **kw)

    def test_p_value_is_not_the_fraction_of_positive_draws(self):
        """The two are different quantities and the artifact reports both.

        Conflating them is the standard bootstrap-testing error: one is a tail
        probability under a recentred null, the other is descriptive.
        """
        a, b = self._pair()
        r = paired_block_bootstrap(a, b, block_len=21, n_boot=500, alpha=0.10, seed=0)
        assert r["p_value_no_outperformance"] != pytest.approx(
            1.0 - r["prob_sharpe_diff_positive"], abs=1e-9
        )

    def test_p_value_is_never_exactly_zero(self):
        """+1 smoothing: a bootstrap p should not assert certainty the
        resample count cannot support."""
        a, b = self._pair()
        b = b - 0.01           # make the candidate win overwhelmingly
        r = paired_block_bootstrap(a, b, block_len=21, n_boot=300, alpha=0.10, seed=0)
        assert r["p_value_no_outperformance"] > 0.0


# ── The artifact's shape is the protocol ─────────────────────────────────────

class TestQ1ArtifactShape:
    def test_reports_two_independent_booleans_and_no_collapsed_verdict(self):
        verdicts = _artifact()["verdicts"]
        assert isinstance(verdicts["economically_material"], bool)
        assert isinstance(verdicts["statistical_evidence"], bool)

        forbidden = {"wins", "winner", "beats", "success", "verdict", "passed"}
        present = forbidden & set(k.lower() for k in verdicts)
        assert not present, (
            f"Q1 must not collapse its two verdicts into {sorted(present)}. "
            "Economic materiality and statistical evidence are independent."
        )

    def test_no_multiple_testing_correction_is_applied(self):
        """No DSR/RC/SPA VALUE may appear — checked on keys, not on prose.

        Deliberately inspects the key set rather than grepping the serialized
        artifact: the `multiple_testing.reason` field EXPLAINS that DSR,
        Reality Check and SPA are not computed, and a substring search cannot
        tell an explanation from a result. That distinction bit this test on
        its first run, and it is the same false-positive class that got a
        significance wording-guard deleted earlier in this project.
        """
        art = _artifact()
        assert art["multiple_testing"]["applied"] is False

        def _keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k.lower()
                    yield from _keys(v)
            elif isinstance(node, list):
                for item in node:
                    yield from _keys(item)

        banned = ("deflated_sharpe", "dsr", "reality_check", "spa_p", "hansen", "white_p")
        offenders = [
            k for k in _keys(art) if any(b in k for b in banned)
        ]
        assert not offenders, (
            f"Multiple-testing fields present in the Q1 artifact: {offenders}. "
            "Q1 is a single pre-specified hypothesis; the correction belongs "
            "to Q2."
        )

    def test_records_the_usd_numeraire_and_no_hedge_concept(self):
        prov = _artifact()["provenance"]
        assert prov["base_currency"] == "USD"
        assert prov["currency_converted"] is False
        assert "not applicable" in prov["hedge_status"]

    def test_records_the_residual_outer_selection_limitation(self):
        limitations = " ".join(_artifact()["limitations"]).lower()
        assert "outer selection" in limitations
        assert "universe" in limitations

    def test_reports_every_required_economic_field(self):
        art = _artifact()
        for side in ("candidate", "comparator"):
            block = art[side]
            for field in (
                "net_sharpe", "net_annual_return", "max_drawdown", "avg_turnover",
                "total_cost_fraction", "fallback_count", "effective_models",
            ):
                assert field in block, f"{side}.{field} missing"

    def test_paired_inference_uses_the_frozen_bootstrap_settings(self):
        pi = _artifact()["paired_inference"]
        boot = load_params()["phase5"]["bootstrap"]
        assert pi["n_boot"] == boot["n_boot"] == 2000
        assert pi["block_len"] == boot["block_len"] == 21
        assert pi["seed"] == boot["seed"]
        assert "recentred on" in pi["p_value_definition"]

    def test_the_test_segment_came_from_the_frozen_split_helper(self):
        seg = _artifact()["provenance"]["test_segment"]
        assert seg["test_frac"] == load_params()["phase5"]["test_frac"] == 0.35
        assert "split_train_test" in seg["split_helper"]

    def test_verdicts_are_consistent_with_the_reported_numbers(self):
        """The booleans must be derivable from the artifact's own figures.

        A stakeholder-facing flag that does not follow from the numbers beside
        it is the failure mode `test_run_dashboard_data` exists to prevent.
        """
        art = _artifact()
        diff = art["observed_difference"]["net_sharpe_diff"]
        p = art["paired_inference"]["one_sided_null_centred_p_value"]
        alpha = art["paired_inference"]["ci_alpha"]

        assert art["verdicts"]["economically_material"] == (diff >= 0.05)
        assert art["verdicts"]["statistical_evidence"] == (p < alpha)

    def test_generated_from_a_clean_revision(self):
        rev = _artifact()["provenance"]["git_revision"]
        assert not rev.endswith("-dirty"), (
            "Q1's artifact was produced from a dirty tree, so it does not "
            "identify the code that produced it."
        )


class TestCostModel:
    def test_every_instrument_carries_the_etf_rate(self):
        """All ten are liquid US-listed ETFs (rule E1), so one rate applies.

        Regression on a real failure: build_cost_vector classifies by the
        RELEASED universes' asset lists and refused six of global_2004's
        tickers outright rather than defaulting them — correct, since a silent
        default is a silent cost misstatement. The fix is an explicit override
        to the SAME frozen rate, not a new cost assumption.
        """
        from backtest import build_cost_vector
        from global_universe import load_global_config

        cfg = load_global_config()
        etf_bps = float(load_params()["backtest"]["costs_bps"]["etf"])
        vector = build_cost_vector(
            cfg["tickers"], etf_cost_bps=etf_bps, bvc_cost_bps=999.0,
            overrides={t: etf_bps for t in cfg["tickers"]},
        )
        assert len(vector) == 10
        assert (vector == etf_bps).all(), vector.to_dict()

    def test_artifact_records_the_cost_model_explicitly(self):
        cost = _artifact()["cost_model"]
        assert cost["one_way_bps_per_instrument"] == (
            load_params()["backtest"]["costs_bps"]["etf"]
        )
        assert len(cost["applies_to"]) == 10
