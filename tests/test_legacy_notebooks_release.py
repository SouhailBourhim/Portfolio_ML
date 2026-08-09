"""Release-safety controls for the refreshed historical notebooks."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
EXECUTABLE = [
    "phase1_eda.ipynb",
    "phase2_backtest.ipynb",
    "phase3_features.ipynb",
    "phase4_regime_covariance.ipynb",
    "phase4b_adaptive_ml_signals.ipynb",
    "phase4c_cost_aware.ipynb",
    "phase5_oos_evaluation.ipynb",
]
ARCHIVED = "deep_morocco_data_expansion.ipynb"


def _load(name: str):
    return nbformat.read(NOTEBOOK_DIR / name, as_version=4)


def _source(name: str) -> str:
    return "\n".join(cell.source for cell in _load(name).cells)


@pytest.mark.parametrize("name", EXECUTABLE)
def test_refreshed_notebook_is_executed_without_errors(name: str) -> None:
    notebook = _load(name)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code
    assert all(cell.execution_count is not None for cell in code)
    assert not [
        output
        for cell in code
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]


@pytest.mark.parametrize("name", EXECUTABLE)
def test_refreshed_notebook_declares_release_and_numeraires(name: str) -> None:
    notebook = _load(name)
    metadata = notebook.metadata["portfolio_ml"]
    assert metadata["safe_refresh"] is True
    assert metadata["release"] == "pfa-defense-ready-mad-currency"
    source = _source(name)
    assert "full_2021" in source and "MAD" in source and "Bank Al-Maghrib" in source
    assert "etf_2017" in source and "USD" in source and "novembre 2004" in source.lower()


@pytest.mark.parametrize("name", EXECUTABLE + [ARCHIVED])
def test_no_retired_headline_or_retracted_inference(name: str) -> None:
    source = _source(name)
    prohibited = [
        r"\+6[,.]2\s*%",
        r"1[,.]2363",
        r"1[,.]1644",
        r"statistically indistinguishable",
        r"statistiquement indiscern",
        r"returns are unitless",
        r"les rendements (?:sont|étant) sans unité",
    ]
    for pattern in prohibited:
        assert re.search(pattern, source, flags=re.IGNORECASE) is None, (name, pattern)


@pytest.mark.parametrize("name", EXECUTABLE)
def test_notebook_does_not_write_canonical_data_or_call_network(name: str) -> None:
    source = _source(name)
    prohibited = [
        ".to_parquet(",
        ".to_json(",
        "dvc repro",
        "requests.get(",
        "yfinance",
        "yf.download(",
        "subprocess.run(",
        "mlflow.start_run(",
    ]
    for token in prohibited:
        assert token not in source, (name, token)


def test_phase5_uses_final_forward_only_and_multiple_testing_protocol() -> None:
    source = _source("phase5_oos_evaluation.ipynb")
    assert "phase5_validation_protocol.json" in source
    assert "paired_comparison_results.json" in source
    assert "reality_check_results.json" in source
    assert "nested_walkforward_results.json" in source
    assert "PurgedKFold" not in source
    assert "240" in source


def test_heavy_recomputations_cross_check_canonical_gold() -> None:
    for name, artifact in {
        "phase2_backtest.ipynb": "phase2_hurdle.json",
        "phase4_regime_covariance.ipynb": "phase4_results.json",
    }.items():
        source = _source(name)
        assert artifact in source
        assert "matches canonical Gold" in source
        assert "assert winner.strategy_name" in source


def test_phase4b_uses_snapshot_bound_explanations_instead_of_duplicate_refits() -> None:
    notebook = _load("phase4b_adaptive_ml_signals.ipynb")
    source = _source("phase4b_adaptive_ml_signals.ipynb")
    assert notebook.metadata["portfolio_ml"]["artifact_driven"] is True
    assert "model_explanations.json" in source
    assert "run_backtest(" not in source
    assert ".fit(" not in source


def test_deep_morocco_is_archived_not_falsely_refreshed() -> None:
    notebook = _load(ARCHIVED)
    metadata = notebook.metadata["portfolio_ml"]
    assert metadata["execution_policy"] == "archived_not_executed"
    assert not [cell for cell in notebook.cells if cell.cell_type == "code"]
    source = " ".join(_source(ARCHIVED).split())
    assert "not release evidence" in source
    assert "outside the final DVC snapshot" in source
    assert "unadjusted" in source
