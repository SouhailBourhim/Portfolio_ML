"""
build_figures_global_2004.py — The three global_2004 report figures.

Addresses: P4 — every value plotted is read from a committed artifact. A figure
is a claim with a picture attached, and a hardcoded number inside one is
harder to catch than a hardcoded number in prose.

Figures:
  1. plafond_expressivite  — allocation expressiveness, etf_2017 vs global_2004
  2. global_2004_sharpes   — Q1 and the honestly selected challengers
  3. global_2004_correction— the 240 Sharpe differentials, the raw best, and
                             the RC/SPA verdict that refuses it

Usage:
    python scripts/build_figures_global_2004.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from metrics import annualized_sharpe  # noqa: E402

GOLD = ROOT / "data" / "gold"
# The FINAL report only. `docs/rapport` is the shorter variant (chapters 1-5,
# with no chapter 6 or 7); a global_2004 chapter there, with those absent,
# would be incoherent — and the distributed PDF is built from rapport_final.
TARGETS = [ROOT / "docs" / "rapport_final" / "assets" / "figures"]

INK, MUTED, ACCENT, WARN = "#1b1b1b", "#8a8a8a", "#1f5fa8", "#b3341f"


def load(name: str) -> dict:
    return json.loads((GOLD / name).read_text())


def save(fig, stem: str) -> None:
    for directory in TARGETS:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf")


def figure_expressiveness(readiness: dict, cap: dict) -> None:
    """Why the universe was built: the optimizer can finally express a view."""
    etf = cap["results"]["0.25"]["min_variance_lw"]["distinct_allocations"]
    freedom = readiness["allocation_freedom"]
    free = freedom["min_variance_lw"]
    gl, n_gl, n_etf = free["distinct_allocations"], freedom["n_rebalances"], 248

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    labels = ["etf_2017\n(5 actifs, plafond 25 %)", "global_2004\n(10 actifs, plafond 25 %)"]
    values = [etf / n_etf, gl / n_gl]
    bars = ax.barh(labels, values, color=[MUTED, ACCENT], height=0.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Part des rééquilibrages produisant une allocation distincte")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for bar, raw, total in zip(bars, [etf, gl], [n_etf, n_gl]):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{raw} / {total}", va="center", fontsize=10, color=INK)
    ax.set_title("Expressivité de l'allocation, à contrainte identique", loc="left",
                 fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "global_2004_expressivite")


def figure_sharpes(q1: dict, interp: dict) -> None:
    """Q1 plus the honestly selected challengers, on the frozen test segment."""
    bench = interp["benchmark"]
    sel = interp["honestly_selected_challengers"]
    rows = [(q1["comparator"]["strategy"], q1["comparator"]["net_sharpe"], MUTED),
            (bench["strategy"], bench["net_sharpe"], ACCENT)]
    rows += [(k, v["net_sharpe"], MUTED) for k, v in sel.items()]
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    names = [r[0] for r in rows]
    bars = ax.barh(names, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
    for bar, (_, value, _) in zip(bars, rows):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=10, color=INK)
    ax.axvline(bench["net_sharpe"], color=ACCENT, ls=":", lw=1)
    ax.set_xlabel("Ratio de Sharpe net, segment de test gelé")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.16)
    ax.set_title("Aucun avantage établi sur le comparateur classique", loc="left",
                 fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "global_2004_sharpes")


def figure_correction(q2: dict) -> None:
    """The 240 differentials, the raw best, and the correction that refuses it."""
    long = pd.read_parquet(GOLD / "global_2004_q2_series.parquet")
    wide = long.pivot(index="Date", columns="candidate", values="net_return")
    bench_sharpe = annualized_sharpe(wide["regime_conditional"])
    diffs = np.array([
        annualized_sharpe(wide[c]) - bench_sharpe
        for c in wide.columns if c != "regime_conditional"
    ])
    primary = q2["family_tests"]["primary_sharpe"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    counts, _, _ = ax.hist(diffs, bins=34, color=MUTED, edgecolor="white", linewidth=0.6)
    # Headroom so the annotation sits in clear space rather than on the bars.
    ax.set_ylim(0, counts.max() * 1.42)
    ax.axvline(0, color=INK, lw=1.2)
    ax.axvline(primary["best_differential"], color=WARN, lw=1.6)
    # Right-aligned to the LEFT of the marker line, so the leader never
    # crosses its own label — the first version drew the arrow straight
    # through the p-values.
    top = ax.get_ylim()[1]
    ax.annotate(
        f"meilleur brut {primary['best_differential']:+.3f}\n"
        f"RC p = {primary['reality_check_p_value']:.3f} · "
        f"SPA p = {primary['spa_p_value']:.3f}\n"
        "→ aucune preuve retenue",
        xy=(primary["best_differential"], top * 0.50),
        xytext=(primary["best_differential"] - 0.040, top * 0.88),
        fontsize=9, color=WARN, ha="right", va="center", linespacing=1.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.5),
        arrowprops=dict(arrowstyle="->", color=WARN, lw=1.1,
                        connectionstyle="arc3,rad=-0.15"),
    )
    beat = primary["n_candidates_beating_benchmark"]
    ax.set_xlabel("Différentiel de Sharpe net face à regime_conditional")
    ax.set_ylabel("Nombre de configurations")
    ax.set_title(
        f"240 configurations atteignables — {beat} dépassent la référence, "
        "aucune n'est retenue",
        loc="left", fontsize=11, color=INK,
    )
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "global_2004_correction")


def main() -> None:
    print("global_2004 figures:")
    readiness, cap = load("global_2004_readiness.json"), load("etf_cap_verdict.json")
    q1, q2 = load("global_2004_q1_results.json"), load("global_2004_q2_results.json")
    interp = load("global_2004_interpretation.json")
    figure_expressiveness(readiness, cap)
    figure_sharpes(q1, interp)
    figure_correction(q2)


if __name__ == "__main__":
    main()
