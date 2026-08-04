"""test_reality_check.py — the correction must reject when it should.

A multiple-testing correction that never rejects is indistinguishable from a
broken one, and it would be the most flattering possible bug for this project:
every result already says "no outperformance established", so a test stuck at
p = 1.0 would agree with every conclusion and never be questioned.

So the suite is built around a **positive control**: a candidate with a genuine
edge, buried in a set of null candidates, MUST be detected. Only then do the
null cases mean anything.

The other property under test is the one that motivates the whole exercise:
the p-value must RISE with the number of searched candidates. If searching 240
configurations were as cheap as searching one, there would be nothing to
correct.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metrics import reality_check


DATES = pd.bdate_range("2022-01-03", periods=750)


def _series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=DATES)


def _null_set(n_candidates: int, seed: int = 0, correlation: float = 0.0) -> tuple[dict, pd.Series]:
    """Candidates with no edge, optionally sharing a common component.

    `correlation` mimics the real situation: configurations differing by one
    hyperparameter on one data window are nearly the same strategy repeated.
    """
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(0.0004, 0.010, len(DATES))
    common = rng.normal(0.0, 0.010, len(DATES))
    candidates = {}
    for k in range(n_candidates):
        idio = rng.normal(0.0, 0.010, len(DATES))
        mixed = correlation * common + np.sqrt(max(1 - correlation**2, 0.0)) * idio
        candidates[f"cand_{k:03d}"] = _series(0.0004 + mixed)
    return candidates, _series(benchmark)


class TestPositiveControl:
    """If this fails, every null result below is meaningless."""

    def test_a_genuine_edge_is_detected_among_null_candidates(self):
        candidates, benchmark = _null_set(20, seed=1)
        # One candidate with a large, persistent edge: +40 bps/day over the
        # benchmark's mean, on 750 days. Nothing subtle.
        winner = benchmark.to_numpy() + 0.004
        candidates["cand_winner"] = _series(winner)

        result = reality_check(candidates, benchmark, n_boot=500, seed=0)
        assert result["best_candidate"] == "cand_winner"
        assert result["reality_check_p_value"] < 0.05, (
            f"a candidate beating the benchmark by 40bps/day for 750 days was not "
            f"detected (RC p = {result['reality_check_p_value']}) — the correction "
            f"is too conservative to be useful"
        )
        assert result["spa_p_value"] < 0.05

    def test_the_interpretation_string_follows_the_verdict(self):
        candidates, benchmark = _null_set(10, seed=2)
        candidates["cand_winner"] = _series(benchmark.to_numpy() + 0.004)
        result = reality_check(candidates, benchmark, n_boot=500, seed=0)
        assert "outperforms the benchmark after" in result["interpretation"]


class TestNullCases:
    def test_pure_null_candidates_are_not_flagged(self):
        candidates, benchmark = _null_set(50, seed=3)
        result = reality_check(candidates, benchmark, n_boot=500, seed=0)
        assert result["reality_check_p_value"] > 0.05
        assert "No evidence" in result["interpretation"]

    def test_some_candidates_beat_the_benchmark_by_luck_yet_the_set_does_not(self):
        """The entire point of the correction, as a test.

        With 50 null candidates, several will beat the benchmark on the sample.
        A naive per-candidate test would call the best of them significant.
        """
        candidates, benchmark = _null_set(50, seed=4)
        result = reality_check(candidates, benchmark, n_boot=500, seed=0)
        assert result["n_candidates_beating_benchmark"] > 0, (
            "fixture is not exercising the case: no candidate beat the benchmark"
        )
        assert result["best_differential"] > 0
        assert result["reality_check_p_value"] > 0.05


class TestSearchSizeIsPenalised:
    def test_the_p_value_rises_with_the_number_of_candidates(self):
        """If breadth were free there would be nothing to correct."""
        _, benchmark = _null_set(1, seed=5)
        p_values = []
        for n_candidates in (2, 20, 200):
            candidates, bench = _null_set(n_candidates, seed=5)
            p_values.append(
                reality_check(candidates, bench, n_boot=400, seed=0)["reality_check_p_value"]
            )
        assert p_values[0] <= p_values[-1], (
            f"searching 200 candidates was no more penalised than searching 2: {p_values}"
        )

    def test_correlated_candidates_are_penalised_less_than_independent_ones(self):
        """240 near-identical configs are not 240 independent bets.

        This is why the bootstrap uses the SAME block draws for every
        candidate: it lets the correction see that the search was narrower than
        its raw count suggests.
        """
        independent, bench_i = _null_set(40, seed=6, correlation=0.0)
        correlated, bench_c = _null_set(40, seed=6, correlation=0.98)
        p_independent = reality_check(independent, bench_i, n_boot=400, seed=0)["reality_check_p_value"]
        p_correlated = reality_check(correlated, bench_c, n_boot=400, seed=0)["reality_check_p_value"]
        assert p_correlated <= p_independent + 1e-9, (
            f"highly correlated candidates were penalised at least as hard as "
            f"independent ones ({p_correlated} vs {p_independent}) — the shared "
            f"block draws are not preserving cross-correlation"
        )


class TestSpaVersusRealityCheck:
    def test_spa_drops_hopeless_candidates_from_the_recentring(self):
        """Padding a search with terrible models must not help the good ones."""
        candidates, benchmark = _null_set(10, seed=7)
        for k in range(10):
            candidates[f"awful_{k}"] = _series(benchmark.to_numpy() - 0.02)
        result = reality_check(candidates, benchmark, n_boot=400, seed=0)
        assert result["spa_candidates_retained"] < result["n_candidates"], (
            "SPA retained every candidate including ones losing 200bps/day; the "
            "consistent-variant threshold is not being applied"
        )


class TestBothStatistics:
    @pytest.mark.parametrize("statistic", ["mean_return", "sharpe"])
    def test_each_statistic_detects_the_positive_control(self, statistic):
        candidates, benchmark = _null_set(15, seed=8)
        candidates["cand_winner"] = _series(benchmark.to_numpy() + 0.004)
        result = reality_check(candidates, benchmark, statistic=statistic,
                               n_boot=500, seed=0)
        assert result["best_candidate"] == "cand_winner"
        assert result["reality_check_p_value"] < 0.05

    def test_an_unknown_statistic_is_rejected(self):
        candidates, benchmark = _null_set(3, seed=9)
        with pytest.raises(ValueError, match="Unknown statistic"):
            reality_check(candidates, benchmark, statistic="calmar")


class TestInputDiscipline:
    def test_an_empty_candidate_set_is_refused(self):
        _, benchmark = _null_set(1, seed=10)
        with pytest.raises(ValueError, match="at least one candidate"):
            reality_check({}, benchmark)

    def test_a_misaligned_candidate_is_refused_not_aligned(self):
        """Silent alignment would change WHICH days the max is taken over."""
        candidates, benchmark = _null_set(2, seed=11)
        shifted = candidates["cand_000"].iloc[:-10]
        with pytest.raises(ValueError, match="not on the benchmark's index"):
            reality_check({"bad": shifted}, benchmark)

    def test_nan_is_refused(self):
        candidates, benchmark = _null_set(2, seed=12)
        candidates["cand_000"].iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            reality_check(candidates, benchmark)

    def test_results_are_reproducible_for_a_fixed_seed(self):
        candidates, benchmark = _null_set(12, seed=13)
        first = reality_check(candidates, benchmark, n_boot=300, seed=42)
        second = reality_check(candidates, benchmark, n_boot=300, seed=42)
        assert first == second


# ── The persisted artifact ───────────────────────────────────────────────────
class TestPersistedArtifact:
    """The correction must describe the search it claims to correct for.

    Same failure class as the fit-report runner: a correction computed over a
    candidate set that is not the searched one answers a different question,
    and would be indistinguishable from a correct one in the output.
    """

    RESULTS = Path(__file__).resolve().parents[1] / "data" / "gold" / "reality_check_results.json"
    SERIES = Path(__file__).resolve().parents[1] / "data" / "gold" / "reality_check_series.parquet"

    def _results(self) -> dict:
        if not self.RESULTS.is_file():
            pytest.skip("reality_check_results.json not present — run `dvc repro reality_check`.")
        return json.loads(self.RESULTS.read_text(encoding="utf-8"))

    def test_the_candidate_count_matches_the_reachable_search_space(self):
        """Correcting for fewer candidates than were reachable understates it."""
        params = yaml.safe_load((Path(__file__).resolve().parents[1] / "params.yaml")
                                .read_text(encoding="utf-8"))["phase5"]
        expected = (
            (len(list(itertools.product(*params["rf_grid"].values())))
             + len(list(itertools.product(*params["xgb_grid"].values()))))
            * len(params["shrink_grid"]) * len(params["penalty_grid"])
        )
        for universe, block in self._results()["universes"].items():
            assert block["n_candidates"] == expected, (
                f"{universe}: corrected for {block['n_candidates']} candidates but the "
                f"search could reach {expected}."
            )

    def test_every_candidate_shares_the_frozen_test_window(self):
        """A max taken over different periods is not a max over candidates."""
        if not self.SERIES.is_file():
            pytest.skip("reality_check_series.parquet not present.")
        frame = pd.read_parquet(self.SERIES)
        for universe, block in frame.groupby("universe"):
            spans = block.groupby("candidate")["Date"].agg(["min", "max", "count"])
            assert spans["min"].nunique() == 1, f"{universe}: candidates start on different dates"
            assert spans["max"].nunique() == 1, f"{universe}: candidates end on different dates"
            assert spans["count"].nunique() == 1, f"{universe}: candidates have different lengths"

    def test_both_benchmarks_are_present_in_the_series(self):
        if not self.SERIES.is_file():
            pytest.skip("reality_check_series.parquet not present.")
        frame = pd.read_parquet(self.SERIES)
        for universe, block in frame.groupby("universe"):
            names = set(block["candidate"])
            assert {"regime_conditional", "equal_weight"} <= names, (
                f"{universe} is missing a benchmark series"
            )

    def test_p_values_are_probabilities(self):
        for block in self._results()["universes"].values():
            for name, test in block["tests"].items():
                for key in ("reality_check_p_value", "spa_p_value"):
                    assert 0.0 < test[key] <= 1.0, f"{name}: {key} = {test[key]}"

    def test_the_artifact_states_why_the_candidate_count_is_not_the_ledger_count(self):
        """The choice to correct for 240 rather than 51 must travel with it."""
        payload = self._results()
        assert "candidate_set_note" in payload
        assert "reachable" in payload["candidate_set_note"].lower()

    def test_the_artifact_declares_the_weaker_statistic_as_weaker(self):
        note = self._results()["statistic_note"].lower()
        assert "ratio of moments" in note, (
            "the Sharpe statistic's weaker asymptotic justification must be stated "
            "with the result, not left for a reader to know"
        )


class TestPersistedArtifact:
    """The released artifact must report honestly, not flatteringly.

    These read the committed artifact rather than synthetic fixtures: the risk
    being guarded is not that the maths is wrong (the tests above cover that)
    but that the RESULT is presented in a way that invites over-reading.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _artifact(self) -> dict:
        path = self.ROOT / "data" / "gold" / "reality_check_results.json"
        if not path.is_file():
            pytest.skip("reality_check_results.json not present — run `dvc repro reality_check`.")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_primary_benchmark_is_pre_specified_in_code(self):
        """Chosen before the p-values were visible, not after."""
        import run_reality_check

        assert run_reality_check.PRIMARY_BENCHMARK == "regime_conditional"
        artifact = self._artifact()
        for universe in artifact["universes"].values():
            primary = [t for t in universe["tests"].values() if t["status"] == "primary"]
            exploratory = [t for t in universe["tests"].values() if t["status"] == "exploratory"]
            assert primary and exploratory, "both statuses must be represented"
            assert all(t["benchmark"] == "regime_conditional" for t in primary)

    def test_every_test_reports_rc_and_spa_together(self):
        """Neither may be quoted alone; SPA is the more powerful by design."""
        for universe in self._artifact()["universes"].values():
            for test in universe["tests"].values():
                assert "reality_check_p_value" in test
                assert "spa_p_value" in test
                assert "spa_candidates_retained" in test, (
                    "the retained count must ship with the p-values: without it a "
                    "reader cannot tell whether an RC/SPA gap comes from the "
                    "trimming rule or from studentization alone"
                )

    def test_the_outer_multiplicity_is_disclosed(self):
        artifact = self._artifact()
        note = artifact["outer_multiplicity_note"]
        assert "Eight outer comparisons" in note
        assert "exploratory" in note

    def test_exploratory_results_are_not_dressed_as_primary(self):
        for universe in self._artifact()["universes"].values():
            for test in universe["tests"].values():
                if test["status"] == "exploratory":
                    assert "not evidence that the ML layer adds value" in test["status_reason"]

    def test_the_candidate_count_is_the_reachable_space_not_the_ledger(self):
        """Correcting for 51 recorded trials would understate the multiplicity."""
        for universe in self._artifact()["universes"].values():
            for test in universe["tests"].values():
                assert test["n_candidates"] == 240, (
                    f"expected the 240 reachable configurations, got "
                    f"{test['n_candidates']}"
                )

    def test_the_candidate_grid_matches_params_yaml(self):
        """The artifact's count must follow the configured search, not a literal."""
        import run_reality_check

        params = yaml.safe_load((self.ROOT / "params.yaml").read_text(encoding="utf-8"))
        phase5 = params["phase5"]
        expected = (
            len(list(itertools.product(*phase5["rf_grid"].values())))
            + len(list(itertools.product(*phase5["xgb_grid"].values())))
        ) * len(phase5["shrink_grid"]) * len(phase5["penalty_grid"])
        assert len(run_reality_check.candidate_configs(params)) == expected
