"""
test_global_universe.py — The `global_2004` experiment, made un-driftable.

Three things are locked in here, each of which has already gone wrong once
somewhere in this project:

  1. THE FROZEN INSTRUMENT SET. The pre-registration is the protocol; the code
     is an implementation of it. A test that parses the frozen document and
     compares it to the shipped config is what makes "frozen" mean something —
     otherwise the freeze is a promise in prose, and §17.11's lesson is that a
     claim living in two places drifts.

  2. THE LAG-DOMINANCE GATE. Two earlier versions of this gate were wrong in
     opposite directions: a ratio that failed a CLEAN universe, and an absolute
     bound that would have passed the KNOWN-BAD control. The regression tests
     below assert the accepted rule does both things right — passes both USD
     universes AND fails all four BVC assets — because a gate is only worth
     having if it discriminates, and neither predecessor did.

  3. THE CALENDAR. `global_2004` deliberately does NOT use
     `clean.align_calendars`, because a business-day grid on a single-calendar
     universe invents ~9.7 rows per year of US market holidays and fills them
     forward into exact-zero returns. That is a subtle, silent contamination of
     the covariance input, so it is asserted rather than trusted.

Offline: the correlation tests read committed Gold artifacts and skip when
absent; nothing here hits the network.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from global_universe import build_global_silver, load_global_config, measure_lag_dominance  # noqa: E402

GOLD = ROOT / "data" / "gold"
PREREG = ROOT / "docs" / "GLOBAL_UNIVERSE_PREREGISTRATION.md"

# The four Casablanca-listed assets. `docs/NONSYNC_COVARIANCE.md` measured the
# stale-price signature on exactly these, with a clean US-listed control.
BVC_ASSETS = ["IAM.CS", "ATW.CS", "CIH.CS", "BCP.CS"]


def _load(name: str) -> pd.DataFrame:
    path = GOLD / name
    if not path.is_file():
        pytest.skip(f"{name} absent — run `dvc pull` or the global_2004 runner.")
    return pd.read_parquet(path)


# ── 1. The freeze is executable ──────────────────────────────────────────────

class TestTheInstrumentSetIsFrozen:
    def test_config_matches_the_pre_registered_tickers(self):
        """The shipped config must equal the frozen document, exactly and in order.

        Parses the fenced ticker block out of §2 rather than trusting a copy.
        If someone edits either side, this fails — which is the entire point of
        calling the set "frozen".
        """
        text = PREREG.read_text(encoding="utf-8")
        match = re.search(r"```\n(SPY[^`]*?)\n```", text)
        assert match, "Could not find the frozen ticker block in §2."
        pre_registered = match.group(1).split()

        configured = load_global_config()["tickers"]
        assert configured == pre_registered, (
            "The implemented instrument set has drifted from the frozen "
            f"protocol.\n  pre-registered: {pre_registered}\n  configured:     {configured}\n"
            "Changing it requires a dated amendment in §10.3, in its own commit, "
            "BEFORE any affected result is calculated."
        )

    def test_the_set_is_exactly_ten_instruments(self):
        assert len(load_global_config()["tickers"]) == 10

    def test_bam_macro_block_is_excluded(self):
        """The BAM exclusion is economic (a USD universe), and load-bearing.

        Retaining TAUX_DIR would push the first fully dense feature row to
        2017-01-04 and discard twelve of the twenty-one years the universe
        exists to provide.
        """
        cfg = load_global_config()
        assert set(cfg["macro_exclude"]) == {"TAUX_DIR", "EURMAD", "USDMAD"}
        assert "TAUX_DIR" not in cfg["macro_retain"]


# ── 2. The gate discriminates — the whole reason it was replaced ─────────────

class TestLagDominanceGate:
    """D_i = max(|rho_i(-1)|, |rho_i(+1)|) - |rho_i(0)|; require max_i D_i <= 0."""

    def test_passes_the_global_2004_universe(self):
        result = measure_lag_dominance(_load("log_returns_global.parquet"))
        assert result["max_lag_dominance"] <= 0.0, result["failing_assets"]
        assert result["failing_assets"] == []

    def test_passes_the_etf_2017_clean_control(self):
        """A universe already known to be free of the defect must pass.

        Without this, a gate could be trivially strict and nobody would notice
        until it rejected something good.
        """
        result = measure_lag_dominance(_load("log_returns_etf.parquet"))
        assert result["max_lag_dominance"] <= 0.0, result["failing_assets"]

    def test_fails_every_bvc_asset_in_full_2021(self):
        """The known-bad control. THIS is the test the rejected repair failed.

        An absolute bound |rho(1)| <= 0.20 was proposed and rejected because
        the BVC block's largest lag correlation is only ~0.09 — it would have
        passed here, making the gate worthless. Lag dominance fails all four.
        """
        result = measure_lag_dominance(_load("log_returns.parquet"))
        per_asset = result["per_asset"]

        for asset in BVC_ASSETS:
            assert asset in per_asset, f"{asset} missing from full_2021."
            assert per_asset[asset]["lag_dominance"] > 0.0, (
                f"{asset} should FAIL lag dominance — it is part of the "
                "documented stale-price block."
            )
            assert not per_asset[asset]["passes"]

        assert set(BVC_ASSETS).issubset(set(result["failing_assets"]))

    def test_discriminates_per_asset_not_per_universe(self):
        """The US-listed assets INSIDE full_2021 must pass while BVC fails.

        A universe-level verdict would be a blunter instrument: this proves the
        statistic isolates the defective instruments rather than condemning
        every asset that shares a file with them.
        """
        result = measure_lag_dominance(_load("log_returns.parquet"))
        us_listed = [c for c in result["per_asset"] if c not in BVC_ASSETS]
        assert us_listed, "full_2021 should contain US-listed assets besides SPY."
        for asset in us_listed:
            assert result["per_asset"][asset]["passes"], (
                f"{asset} is US-listed and should pass lag dominance."
            )

    def test_the_rejected_absolute_bound_would_have_passed_the_bad_block(self):
        """Locks in WHY the absolute bound was rejected, not just that it was.

        If someone later proposes `max |rho(1)| <= 0.20` again, this test is the
        counter-example, with the numbers attached.
        """
        result = measure_lag_dominance(_load("log_returns.parquet"))
        worst_bvc_lag = max(
            result["per_asset"][a]["abs_corr_ref_leads"] for a in BVC_ASSETS
        )
        assert worst_bvc_lag < 0.20, (
            "The BVC block's largest lag correlation is below 0.20, which is "
            "why an absolute bound at that level cannot detect it."
        )
        assert result["max_lag_dominance"] > 0.0, (
            "...while lag dominance does detect it."
        )

    def test_statistic_is_symmetric_in_lead_and_lag(self):
        """Reversing which series is shifted must not change the verdict.

        The gate takes max(|rho(-1)|, |rho(+1)|) precisely so it cannot be
        evaded by ordering the pair the other way round.
        """
        df = _load("log_returns_global.parquet")
        result = measure_lag_dominance(df)
        for asset, v in result["per_asset"].items():
            expected = max(v["abs_corr_ref_leads"], v["abs_corr_asset_leads"]) - v["abs_corr_same_day"]
            assert v["lag_dominance"] == pytest.approx(expected, abs=1e-4), asset


# ── 3. The calendar defect cannot come back ──────────────────────────────────

class TestSingleCalendarAlignment:
    def test_no_synthetic_business_days_are_introduced(self, tmp_path):
        """A single-calendar universe must not gain market-holiday rows.

        Regression on the real defect: `clean.align_calendars` turned 5,468
        observed dates into 5,672 by expanding to every business day, then
        forward-filled all columns across the ~204 US market holidays, turning
        each into an exact-zero return in every asset at once.
        """
        # Two weeks of business days with a mid-week holiday removed from both
        # series — a market closure, not a per-asset gap.
        idx = pd.bdate_range("2020-01-01", periods=15)
        holiday = idx[5]
        observed = idx.drop(holiday)
        prices = pd.DataFrame(
            {"SPY": np.linspace(100, 114, len(observed)),
             "GLD": np.linspace(50, 57, len(observed))},
            index=observed,
        )
        prices.index.name = "Date"

        returns, coverage = build_global_silver(
            prices, ffill_limit=5, out_path=tmp_path / "silver.parquet"
        )

        assert holiday not in returns.index, (
            "A market-wide closure was resurrected as a synthetic row."
        )
        assert coverage["forward_filled_cells"] == 0, (
            "Nothing should be carried forward when no instrument was missing."
        )
        assert (returns == 0.0).sum().sum() == 0, (
            "No exact-zero returns should be manufactured by alignment."
        )

    def test_real_universe_has_no_forward_filled_cells(self):
        """On the committed artifact, the fix holds: zero cells filled."""
        path = GOLD / "global_2004_readiness.json"
        if not path.is_file():
            pytest.skip("readiness artifact absent — run the global_2004 runner.")
        import json

        coverage = json.loads(path.read_text())["coverage"]
        assert coverage["forward_filled_cells"] == 0
        assert coverage["max_zero_return_share"] < 0.02


# ── 4. Completed evidence stays frozen, and the protocol is not a trigger ────

class TestFrozenEvidenceAndProtocolLineage:
    """Locks in the 2026-08-15 lineage migration.

    What went wrong: `docs/GLOBAL_UNIVERSE_PREREGISTRATION.md` was a DVC
    DEPENDENCY of the global_2004 stages. Committing amendment 2 to that
    document invalidated `global_2004_data` and `global_2004_q1` as upstreams,
    so `dvc repro global_2004_q2` silently regenerated two pieces of
    already-frozen evidence before Q2 even started. Q2's own guard caught it
    (the regenerated Q1 carried a dirty revision) and no candidate was ever
    evaluated, but the evidence had already been overwritten.

    A living protocol document is not a computational input. Each artifact
    records the document's hash INTERNALLY, which is the honest way to state
    which protocol governed a run, and the drift guard is
    `test_config_matches_the_pre_registered_tickers` above — it fails visibly
    instead of rebuilding evidence.
    """

    COMPLETED_STAGES = ("global_2004_data", "global_2004_q1")
    PROTOCOL_DOC = "docs/GLOBAL_UNIVERSE_PREREGISTRATION.md"

    def _stages(self) -> dict:
        path = ROOT / "dvc.yaml"
        if not path.is_file():
            pytest.skip("dvc.yaml absent")
        return yaml.safe_load(path.read_text())["stages"]

    def test_completed_stages_are_frozen(self):
        stages = self._stages()
        for name in self.COMPLETED_STAGES:
            assert name in stages, f"{name} missing from dvc.yaml"
            assert stages[name].get("frozen") is True, (
                f"{name} is run-once evidence and must be frozen so Q2 "
                "consumes its pinned output rather than rebuilding it."
            )

    def test_q2_is_not_frozen(self):
        """The other direction: freezing everything would be useless."""
        stages = self._stages()
        if "global_2004_q2" in stages:
            assert stages["global_2004_q2"].get("frozen") is not True

    @pytest.mark.parametrize(
        "stage", ["global_2004_data", "global_2004_q1", "global_2004_q2"]
    )
    def test_protocol_document_is_not_a_dvc_dependency(self, stage):
        stages = self._stages()
        if stage not in stages:
            pytest.skip(f"{stage} not defined")
        deps = stages[stage].get("deps") or []
        assert self.PROTOCOL_DOC not in deps, (
            f"{stage} depends on the protocol document. A later amendment "
            "would then invalidate this stage and regenerate frozen evidence "
            "— exactly the 2026-08-15 failure. Record the document hash inside "
            "the result artifact instead."
        )

    def test_result_artifacts_still_record_the_protocol_hash(self):
        """Removing the DVC trigger must not lose the provenance.

        Q1 records it. The readiness artifact does NOT — adding it would
        require re-running frozen evidence, so the gap is asserted here as a
        known, deliberate limitation rather than left to be discovered.
        """
        q1_path = GOLD / "global_2004_q1_results.json"
        if not q1_path.is_file():
            pytest.skip("Q1 artifact absent")
        sources = json.loads(q1_path.read_text())["provenance"]["source_artifacts"]
        assert self.PROTOCOL_DOC in sources
        assert sources[self.PROTOCOL_DOC], "protocol hash recorded but empty"
