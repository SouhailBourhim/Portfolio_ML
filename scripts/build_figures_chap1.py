"""Figure for Chapter 1 (contexte et cadrage) of the internship report.

The activity bars are the project's real commit history, read from git — not a
plan redrawn after the fact. The phase bands above them are the milestone dates
recorded in the project log; they are declared here because git alone cannot
name them.
"""
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "rapport" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREY, ORANGE = "#1F3864", "#595959", "#C77700"

START, END = pd.Timestamp("2026-06-18"), pd.Timestamp("2026-08-01")

# (libellé, début, fin, couleur) — jalons consignés dans le journal de projet.
PHASES = [
    ("Cadrage, étude du domaine, choix des sources", "2026-06-18", "2026-06-28", GREY),
    ("Phase 1 — infrastructure de données",          "2026-06-29", "2026-07-02", BLUE),
    ("Phase 2 — Markowitz et moteur de backtest",    "2026-07-03", "2026-07-10", BLUE),
    ("Phase 3 — variables ML causales",              "2026-07-11", "2026-07-20", BLUE),
    ("Phases 4, 4B, 4C — régimes, signaux, coûts",   "2026-07-20", "2026-07-21", BLUE),
    ("Phase 5 — évaluation hors échantillon",        "2026-07-22", "2026-07-23", BLUE),
    ("Études complémentaires (5 pistes)",            "2026-07-23", "2026-07-30", ORANGE),
    ("Phases 6+7 — tableau de bord et API",          "2026-07-25", "2026-07-27", BLUE),
    ("Consolidation, rapport et livrables",          "2026-07-31", "2026-08-01", GREY),
]

plt.rcParams.update({
    "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def commits_per_day():
    """Real commit dates from git; empty Series if git is unavailable."""
    try:
        out = subprocess.run(["git", "log", "--format=%ad", "--date=short"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:
        return pd.Series(dtype=int)
    days = pd.to_datetime([d for d in out.split() if d])
    return pd.Series(1, index=days).groupby(level=0).sum().sort_index()


def chronologie():
    counts = commits_per_day()
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(8.6, 4.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.5, 1],
                                               "hspace": 0.12})

    for i, (label, d0, d1, col) in enumerate(PHASES):
        y = len(PHASES) - 1 - i
        a, b = pd.Timestamp(d0), pd.Timestamp(d1) + pd.Timedelta(days=1)
        ax.barh(y, (b - a).days, left=a, height=0.62, color=col, alpha=0.9, lw=0)
        ax.text(b + pd.Timedelta(days=1), y, label, va="center", fontsize=7.2,
                color="#333333")
    ax.set_ylim(-0.8, len(PHASES) - 0.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Déroulement du stage : six semaines et demie, du 18 juin au "
                 "1\\textsuperscript{er} août 2026".replace("\\textsuperscript{er}", "ᵉʳ"),
                 fontsize=10, color=BLUE, pad=8)

    if len(counts):
        axb.bar(counts.index, counts.values, width=0.85, color=BLUE, alpha=0.75, lw=0)
        axb.set_ylabel("commits\npar jour", fontsize=7.2, color=GREY)
        axb.tick_params(labelsize=7.2)
        peak = counts.idxmax()
        axb.annotate(f"{int(counts.max())}", (peak, counts.max()), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=6.8, color=BLUE)
    axb.set_xlim(START - pd.Timedelta(days=1), END + pd.Timedelta(days=1))
    axb.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axb.grid(alpha=0.15, lw=0.5, axis="y")
    for a_ in (ax, axb):
        a_.axvline(START, color=GREY, lw=0.8, ls=":")
        a_.axvline(END, color=GREY, lw=0.8, ls=":")
    axb.set_xlabel("")
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.savefig(OUT / "chronologie.pdf")
    plt.close(fig)
    print(f"  chronologie.pdf   ({int(counts.sum())} commits sur {len(counts)} jours)")


if __name__ == "__main__":
    print("figure chapitre 1 :")
    chronologie()
