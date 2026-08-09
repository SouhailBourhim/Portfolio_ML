"""Refresh the eight historical notebooks against the final MAD/BAM release.

The source notebooks in the main worktree are user-owned.  This script operates only
on copies in the release worktree.  Seven notebooks remain executable; the deep-
Morocco experiment is archived because its input/results are outside the versioned
release snapshot and its unadjusted-price design is not comparable with current Gold.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

COMMON_NOTE = """
> **Actualisation sûre — release MAD/BAM (2026-08-09).** Ce notebook est exécuté sur
> les artefacts finaux versionnés. `full_2021` est exprimé en **MAD**, avec conversion
> causale des niveaux de prix ETF au taux de référence officiel Bank Al-Maghrib, sans
> couverture de change. `etf_2017` est un univers composé uniquement d'ETF en **USD**,
> commence en novembre 2004 et n'est pas modifié par cette correction. Les niveaux de
> Sharpe des deux univers ne sont donc pas comparés entre eux. Les écarts ponctuels sont
> descriptifs; le livrable est un prototype de recherche, ni un conseil, ni un système
> déployé.
""".strip()

PREFLIGHT = r'''
from pathlib import Path
import json

SAFE_ROOT = Path.cwd()
if not (SAFE_ROOT / "data" / "gold").exists():
    SAFE_ROOT = SAFE_ROOT.parent
GOLD_RELEASE = SAFE_ROOT / "data" / "gold"

snapshot = json.loads((GOLD_RELEASE / "snapshot_manifest.json").read_text())
currency = json.loads((GOLD_RELEASE / "currency_manifest.json").read_text())
assert snapshot["git_dirty"] is False, "The published snapshot itself must have clean provenance."
assert currency["universes"]["full_2021"]["base_currency"] == "MAD"
assert currency["universes"]["etf_2017"]["base_currency"] == "USD"
print("Published snapshot:", snapshot["git_commit"][:12])
print("full_2021: MAD, official BAM FX, unhedged")
print("etf_2017: USD-only, FX correction not applicable")
'''.strip()


def _source(cell: nbf.NotebookNode) -> str:
    return str(cell.source)


def _find(nb: nbf.NotebookNode, prefix: str) -> int:
    for i, cell in enumerate(nb.cells):
        if _source(cell).lstrip().startswith(prefix):
            return i
    raise KeyError(prefix)


def _set(nb: nbf.NotebookNode, prefix: str, text: str, *, cell_type: str = "markdown") -> None:
    i = _find(nb, prefix)
    nb.cells[i] = nbf.v4.new_markdown_cell(text) if cell_type == "markdown" else nbf.v4.new_code_cell(text)


def _insert_release_guard(nb: nbf.NotebookNode, original_hash: str) -> None:
    nb.cells = [c for c in nb.cells if "release-preflight" not in c.get("metadata", {}).get("tags", [])]
    note = nbf.v4.new_markdown_cell(COMMON_NOTE, metadata={"tags": ["release-preflight"]})
    guard = nbf.v4.new_code_cell(PREFLIGHT, metadata={"tags": ["release-preflight"]})
    nb.cells[1:1] = [note, guard]
    nb.metadata.setdefault("portfolio_ml", {})
    nb.metadata["portfolio_ml"].update(
        {
            "safe_refresh": True,
            "release": "pfa-defense-ready-mad-currency",
            "source_copy_sha256": original_hash,
            "canonical_inputs": "data/gold + snapshot_manifest.json + currency_manifest.json",
        }
    )


def _load(name: str) -> tuple[nbf.NotebookNode, str]:
    path = NOTEBOOKS / name
    raw = path.read_bytes()
    return nbf.reads(raw.decode("utf-8"), as_version=4), hashlib.sha256(raw).hexdigest()


def _write(name: str, nb: nbf.NotebookNode) -> None:
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbf.write(nb, NOTEBOOKS / name)


def refresh_phase1() -> None:
    name = "phase1_eda.ipynb"
    nb, digest = _load(name)
    _insert_release_guard(nb, digest)
    nb.cells[0].source = _source(nb.cells[0]).replace(
        "**Date ranges:** ETFs 2017–today | BVC 2021–today (medias24 free-tier limit) | BAM macro 2017–today",
        "**Date ranges:** ETF-only 2004-11→today | BVC-inclusive 2021-07→today | official BAM USD/MAD 2021-07→today",
    )
    load_i = _find(nb, "# Gold: log-returns")
    nb.cells[load_i].source += r'''

# Official BAM USD/MAD used by the final currency correction.  Keep the legacy
# macro frame for TAUX_DIR/EURMAD, but never visualize Yahoo's noisy USD/MAD quote.
bam_usd = pd.read_parquet(BRONZE / "bam_fx_reference.parquet")
bam_usd.index = pd.to_datetime(bam_usd.index)
official_col = "USDMAD" if "USDMAD" in bam_usd.columns else bam_usd.columns[0]
bam_raw = bam_raw.drop(columns=["USDMAD"], errors="ignore").join(
    bam_usd[[official_col]].rename(columns={official_col: "USDMAD"}), how="outer"
).sort_index()
'''
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = _source(cell).replace("ETFs 2017–today", "ETF-only 2004-11→today")
            cell.source = _source(cell).replace("ETF-only analysis could go back to 2017", "ETF-only analysis goes back to November 2004")
        elif cell.cell_type == "code":
            cell.source = _source(cell).replace("2017-today history", "2004-11-today history")
            cell.source = _source(cell).replace("(2017–today)", "(2004-11–today)")
            cell.source = _source(cell).replace("USD/MAD change 2017→today", "USD/MAD change over official BAM coverage")
    _write(name, nb)


def refresh_phase2() -> None:
    name = "phase2_backtest.ipynb"
    nb, digest = _load(name)
    _insert_release_guard(nb, digest)
    intro = _source(nb.cells[0])
    intro = intro.replace("the 5-ETF universe covers 2017→today including both crises.", "the 5-ETF universe covers November 2004→today, including 2008, 2020 and 2022.")
    intro = intro.replace(
        "**Currency:** BVC returns are MAD-denominated, ETF returns USD. Returns are unitless so the\n  arithmetic is valid, but portfolio results embed an **unhedged USD/MAD exposure**. This is\n  documented, not corrected — an FX-hedging model is out of scope.",
        "**Numéraire:** `full_2021` is consistently MAD after causal conversion of ETF price levels\n  with official BAM USD/MAD rates; it remains unhedged. `etf_2017` is consistently USD.",
    )
    nb.cells[0].source = intro
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = _source(cell).replace("5 ETFs, 2017→today", "5 ETFs, 2004-11→today")
            cell.source = _source(cell).replace("5 ETFs, 2017→", "5 ETFs, 2004-11→")
            cell.source = _source(cell).replace("Results embed unhedged USD/MAD exposure (documented §1).", "`full_2021` is MAD and unhedged; `etf_2017` is USD. Their Sharpe levels are not cross-compared.")
    nb.cells.append(nbf.v4.new_markdown_cell("## 8. Release cross-check — recomputation versus the canonical hurdle"))
    nb.cells.append(nbf.v4.new_code_cell(r'''
import json
canonical_hurdle = json.loads((SAFE_ROOT / "data" / "gold" / "phase2_hurdle.json").read_text())
for universe, results in RESULTS.items():
    winner = max(results, key=lambda r: annualized_sharpe(r.net_returns))
    measured = annualized_sharpe(winner.net_returns)
    expected = canonical_hurdle[universe]
    assert winner.strategy_name == expected["strategy"]
    assert abs(measured - expected["sharpe_net"]) < 5e-4
    print(f"{universe}: {winner.strategy_name} {measured:.4f} — matches canonical Gold")
'''.strip()))
    _write(name, nb)


def refresh_phase3() -> None:
    name = "phase3_features.ipynb"
    nb, digest = _load(name)
    _insert_release_guard(nb, digest)
    for cell in nb.cells:
        cell.source = _source(cell).replace("ETF 2017+", "ETF 2004+")
        cell.source = _source(cell).replace("ETF-only (2017+)", "ETF-only (2004-11+)")
    _write(name, nb)


def refresh_phase4() -> None:
    name = "phase4_regime_covariance.ipynb"
    nb, digest = _load(name)
    _insert_release_guard(nb, digest)
    intro = _source(nb.cells[0])
    intro = intro.replace("covers 2017→today", "covers November 2004→today")
    intro = intro.replace(
        "BVC returns are MAD-denominated, ETFs USD — an unhedged USD/MAD exposure is embedded and not\n  \"fixed\" here (out of scope).",
        "`full_2021` is consistently MAD after official BAM conversion and remains unhedged;\n  `etf_2017` is consistently USD.",
    )
    intro = intro.replace("Ledoit-Wolf fallback (observed for real on `IAM.CS` in `full_2021`, see §3).", "fallback path (tested, while the released snapshot records zero fallbacks across the evaluated fits).")
    nb.cells[0].source = intro
    _set(
        nb,
        "**Reading the numbers above:**",
        """**Reading the current run.** The ETF-only universe has substantially deeper history, while
`full_2021` begins in 2021 and is evaluated in MAD. The covariance/regime family does not replace
the classical winner on either released universe. This is an observed ranking, not a paired test
of `regime_conditional` against `max_sharpe`. The fixed 25% cap is near-determining with five ETF
assets, so the ETF result also measures the constraint as much as the estimator.""",
    )
    _set(
        nb,
        "## 8. Summary",
        """## 8. Summary — what Phase 4 establishes in the final release

- HMM regime detection and the covariance ablation ladder run through the same causal engine.
- On the released MAD/BAM snapshot, neither universe's best point estimate is a Phase 4 strategy;
  the canonical artifact below is the authority.
- Telemetry, not a label, establishes whether a fallback occurred; the release records zero
  fallbacks across the evaluated fits, without claiming that fallback paths can never fire.
- The result is descriptive and does not establish superiority or equivalence.""",
    )
    nb.cells.append(nbf.v4.new_code_cell(r'''
canonical = json.loads((ROOT / "data" / "gold" / "phase4_results.json").read_text())
for universe, results in RESULTS.items():
    winner = max(results, key=lambda r: annualized_sharpe(r.net_returns))
    measured = annualized_sharpe(winner.net_returns)
    assert winner.strategy_name == canonical[universe]["strategy"]
    assert abs(measured - canonical[universe]["sharpe_net"]) < 5e-4
    print(f"{universe}: {winner.strategy_name} {measured:.4f} — matches canonical Gold")
'''.strip()))
    _write(name, nb)


def refresh_phase4b() -> None:
    name = "phase4b_adaptive_ml_signals.ipynb"
    _, digest = _load(name)
    # Phase 4 already performs one complete independent recomputation.  Repeating all
    # HMM/DCC fits inside Phase 4B would add runtime without new evidence.  The final
    # explainability artifact contains exact additive RF/XGB attributions plus the
    # decision trace and is snapshot-bound, so use its audited presentation notebook.
    nb = nbf.read(NOTEBOOKS / "phase7_model_decision_explainability.ipynb", as_version=4)
    nb.cells[0].source = "# Phase 4B — Signaux ML adaptatifs : explicabilité et décision\n\n" + "\n".join(
        _source(nb.cells[0]).splitlines()[2:]
    )
    nb.metadata["title"] = "Phase 4B — Signaux ML adaptatifs : explicabilité et décision"
    nb.metadata.setdefault("portfolio_ml", {}).update(
        {
            "safe_refresh": True,
            "release": "pfa-defense-ready-mad-currency",
            "source_copy_sha256": digest,
            "canonical_inputs": "data/gold + snapshot_manifest.json + currency_manifest.json",
            "artifact_driven": True,
        }
    )
    _insert_release_guard(nb, digest)
    _write(name, nb)


def refresh_phase4c() -> None:
    name = "phase4c_cost_aware.ipynb"
    nb, digest = _load(name)
    nb.cells = [c for c in nb.cells if not _source(c).startswith("> ## ⚠️ Numbers below are PRE-CORRECTION")]
    _insert_release_guard(nb, digest)
    _set(
        nb,
        "# Phase 4C",
        """# Phase 4C — Cost-aware optimization and expected-return regularization

This notebook reads the versioned `phase4c_results.json`; it does not refit the 14-strategy
comparison. It tests how shrinkage, rank transforms and turnover penalties change gross return,
net return and turnover. The released result is negative: no Phase 4C variant displaces the
classical hurdle on either universe. That makes the lever diagnosis useful, but not evidence of
ML added value.""",
    )
    _set(
        nb,
        "**The three findings, stated plainly:**",
        """**How to read the levers.** Shrinkage and turnover penalties change the trade-off, but their
effects are model-specific. A penalty can reduce turnover while destroying the gross signal; a
shrinkage transform can improve a challenger without clearing the released hurdle. These are
mechanical observations from fixed configurations, not tuned economic conclusions.""",
    )
    _set(
        nb,
        "## 6. `etf_2017`",
        """## 6. `etf_2017` — same levers on the USD-only deep-history universe

This five-ETF universe begins in November 2004. Its 25% cap is near-determining, so changes in
expected-return modelling have limited room to alter allocations. The table is still reported;
the constraint caveat prevents interpreting a negative challenger result as a universal statement
about ML.""",
    )
    _set(
        nb,
        "Nothing clears",
        "The canonical table above determines the current ranking. No Phase 4C challenger displaces the released classical hurdle on this universe.",
    )
    _set(
        nb,
        "## 7. Summary",
        """## 7. Summary — final release

No Phase 4C variant displaces the classical hurdle on either universe. The useful contribution is
the diagnosis: expected-return regularization and transaction-cost penalties affect RF and XGB
differently, so a single global lever is not defensible. Later forward-only selection and the
240-candidate White/SPA correction do not establish challenger outperformance.""",
    )
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = _source(cell).replace("BEATS' if", "ABOVE' if").replace("does NOT beat", "is not above")
            cell.source = _source(cell).replace("(beats hurdle:", "(above hurdle:")
            cell.source = _source(cell).replace('print("full_2021 — change vs. the rf_signal baseline (net 1.062, turnover 0.885):")', 'print("full_2021 — change vs. the current rf_signal baseline:")')
    _write(name, nb)


def refresh_phase5() -> None:
    name = "phase5_oos_evaluation.ipynb"
    _, digest = _load(name)
    spec = importlib.util.spec_from_file_location("build_visualization_notebooks", ROOT / "scripts" / "build_visualization_notebooks.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nb = module.build_validation_notebook()
    nb.cells[0].source = _source(nb.cells[0]).replace(
        "# Phase 8 — Validation temporelle et correction du data snooping",
        "# Phase 5 — Évaluation hors échantillon, protocole final",
    )
    nb.metadata["title"] = "Phase 5 — Évaluation hors échantillon, protocole final"
    _insert_release_guard(nb, digest)
    _write(name, nb)


def archive_deep_morocco() -> None:
    name = "deep_morocco_data_expansion.ipynb"
    nb, digest = _load(name)
    nb.cells = [
        nbf.v4.new_markdown_cell(
            """# Deep Moroccan Data — archived exploratory experiment

This notebook is preserved for project history, but is **not release evidence** and is deliberately
not re-executed. Its Investing.com price files and result artifacts are outside the final DVC
snapshot; the prices are unadjusted for dividends and end in 2024. Its former wording about
statistical indistinguishability/significance and portfolio added value is retracted.

The safe replacement is the released dual-universe analysis: `etf_2017` uses dividend-adjusted USD
ETF history from November 2004, while `full_2021` uses BVC total returns plus ETF price levels
converted causally to MAD with official BAM rates. Any future deep-Morocco experiment must first
version its raw sources, apply corporate-action adjustments, define a numéraire, enter the DVC
graph, and rerun the forward-only/paired/White-SPA protocol."""
        ),
        nbf.v4.new_markdown_cell(
            """## Why execution is blocked

- canonical inputs absent from the release snapshot;
- unadjusted equity prices are not comparable with the dividend-corrected release;
- the experiment used a bounded five-trial DSR rather than the final reachable 240-candidate
  correction;
- re-running it would produce a fresh-looking notebook from superseded evidence.

The original file remains untouched in the main worktree and a byte-for-byte safety copy was made
before this release copy was archived."""
        ),
    ]
    nb.metadata.setdefault("portfolio_ml", {})
    nb.metadata["portfolio_ml"].update(
        {
            "safe_refresh": True,
            "execution_policy": "archived_not_executed",
            "source_copy_sha256": digest,
            "reason": "inputs and outputs are outside the final versioned snapshot",
        }
    )
    _write(name, nb)


def main() -> None:
    # This is a controlled migration from the preserved user copies, not a notebook
    # generator.  Refuse a second pass: several transformations intentionally replace
    # historical prose by semantic heading, and silently applying them twice would be
    # harder to audit than requiring a fresh copy from the recorded source hashes.
    already_refreshed = []
    for path in NOTEBOOKS.glob("*.ipynb"):
        notebook = nbf.read(path, as_version=4)
        if notebook.metadata.get("portfolio_ml", {}).get("safe_refresh"):
            already_refreshed.append(path.name)
    if already_refreshed:
        raise RuntimeError(
            "Refusing a second migration pass; restore the preserved source copies first: "
            + ", ".join(sorted(already_refreshed))
        )
    refresh_phase1()
    refresh_phase2()
    refresh_phase3()
    refresh_phase4()
    refresh_phase4b()
    refresh_phase4c()
    refresh_phase5()
    archive_deep_morocco()


if __name__ == "__main__":
    main()
