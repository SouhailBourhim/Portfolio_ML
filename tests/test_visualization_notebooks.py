"""Integrity checks for the defense-oriented visualization notebooks."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "phase6_currency_correction.ipynb": {
        "currency_manifest.json",
        "dashboard_showcase.json",
        "snapshot_manifest.json",
    },
    "phase7_model_decision_explainability.ipynb": {
        "model_explanations.json",
        "dashboard_regime.parquet",
        "snapshot_manifest.json",
    },
    "phase8_validation_and_statistical_evidence.ipynb": {
        "phase5_validation_protocol.json",
        "paired_comparison_results.json",
        "reality_check_results.json",
        "reality_check_series.parquet",
        "nested_walkforward_results.json",
        "snapshot_manifest.json",
    },
    "phase9_risk_cost_and_robustness.ipynb": {
        "dashboard_equity.parquet",
        "fit_report_summary.json",
        "fit_reports.parquet",
        "crisis_windows.json",
        "monitoring_baseline.json",
        "snapshot_manifest.json",
    },
    "phase10_global_2004_evidence.ipynb": {
        "global_2004_readiness.json",
        "global_2004_q1_results.json",
        "global_2004_q2_results.json",
        "global_2004_q2_series.parquet",
        "snapshot_manifest.json",
    },
}


def _load(name: str) -> nbformat.NotebookNode:
    return nbformat.read(ROOT / "notebooks" / name, as_version=4)


@pytest.mark.parametrize(("name", "artifacts"), NOTEBOOKS.items())
def test_notebook_is_artifact_driven_and_names_every_source(
    name: str, artifacts: set[str]
) -> None:
    notebook = _load(name)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert notebook.metadata["portfolio_ml"]["artifact_driven"] is True
    for artifact in artifacts:
        assert artifact in source
    assert "snapshot[\"git_dirty\"] is False" in source


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_executed_without_cell_errors(name: str) -> None:
    notebook = _load(name)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_does_not_launch_training_or_external_io(name: str) -> None:
    source = "\n".join(cell.source for cell in _load(name).cells)
    forbidden = (
        "run_backtest(",
        "run_phase4(",
        "run_phase5(",
        "fit_predict_expected_returns(",
        "GridSearchCV(",
        "RandomizedSearchCV(",
        "requests.get(",
        "yfinance",
        "fredapi",
    )
    assert not [token for token in forbidden if token in source]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_does_not_republish_retired_or_prohibited_claims(name: str) -> None:
    source = "\n".join(cell.source for cell in _load(name).cells)
    retired = (r"\+\s*6[,.]2\s*%", r"1[,.]2363", r"1[,.]1644")
    assert not [pattern for pattern in retired if re.search(pattern, source)]
    lowered = source.casefold()
    prohibited = (
        "statistiquement significatif",
        "statistiquement indiscernable",
        "statistically significant",
        "production-ready",
        "conseil d'investissement personnalisé",
    )
    assert not [phrase for phrase in prohibited if phrase in lowered]


def test_currency_notebook_keeps_the_two_numeraires_separate() -> None:
    source = "\n".join(
        cell.source
        for cell in _load("phase6_currency_correction.ipynb").cells
    )
    assert '== "MAD"' in source
    assert '== "USD"' in source
    assert "ne sont jamais agrégés" in source


def test_global_2004_notebook_preserves_the_licensed_interpretation() -> None:
    source = "\n".join(
        cell.source
        for cell in _load("phase10_global_2004_evidence.ipynb").cells
    )
    lowered = source.casefold()
    assert "deux évaluations distinctes mais statistiquement recouvrantes" in lowered
    assert "aucun test pairé individuel" in lowered
    assert "sharpe net observé inférieur" in lowered
    assert "indépendantes" not in lowered
    assert "masqué un avantage" not in lowered


def test_generator_and_generated_sources_are_in_sync(tmp_path: Path) -> None:
    before = {
        name: (ROOT / "notebooks" / name).read_bytes() for name in NOTEBOOKS
    }
    generator = ROOT / "scripts" / "build_visualization_notebooks.py"
    namespace = {"__name__": "not_main", "__file__": str(generator)}
    exec(compile(generator.read_text(encoding="utf-8"), str(generator), "exec"), namespace)
    rebuilt = {name: namespace[builder]() for name, builder in {
        "phase6_currency_correction.ipynb": "build_currency_notebook",
        "phase7_model_decision_explainability.ipynb": "build_explainability_notebook",
        "phase8_validation_and_statistical_evidence.ipynb": "build_validation_notebook",
        "phase9_risk_cost_and_robustness.ipynb": "build_risk_notebook",
        "phase10_global_2004_evidence.ipynb": "build_global_2004_notebook",
    }.items()}
    for name, notebook in rebuilt.items():
        # Execution outputs are intentionally absent from the deterministic source builder.
        for cell in notebook.cells:
            cell.metadata.pop("execution", None)
            if cell.cell_type == "code":
                cell.execution_count = None
                cell.outputs = []
        current = nbformat.reads(before[name].decode("utf-8"), as_version=4)
        for cell in current.cells:
            cell.metadata.pop("execution", None)
            if cell.cell_type == "code":
                cell.execution_count = None
                cell.outputs = []
        assert [
            (cell.cell_type, cell.source, cell.id) for cell in current.cells
        ] == [
            (cell.cell_type, cell.source, cell.id) for cell in notebook.cells
        ]
        assert current.metadata["portfolio_ml"] == notebook.metadata["portfolio_ml"]
