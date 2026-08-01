"""Figures for Chapter 4 (modélisation et évaluation) of the internship report.

Two of the three are deliberately schematic — they explain a protocol, not a
result, and inventing data for them would be worse than drawing them. The third
(the regime timeline) is read from the committed Gold artifact, like every other
data figure in the report.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
OUT = ROOT / "docs" / "rapport" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREY, RED = "#1F3864", "#595959", "#B03000"
TRAIN, EARN, PURGE = "#1F3864", "#C77700", "#B03000"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ── 1. Walk-forward: what the engine actually does ───────────────────────────
def walkforward():
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    n, h = 5, 0.44
    for i in range(n):
        y = n - 1 - i
        tau = 3.0 + i * 1.6
        ax.add_patch(Rectangle((0, y - h / 2), tau, h, fc=TRAIN, ec="none", alpha=0.88))
        ax.add_patch(Rectangle((tau, y - h / 2), 1.6, h, fc=EARN, ec="none", alpha=0.88))
        ax.plot([tau, tau], [y - h / 2 - 0.14, y + h / 2 + 0.14], color="white", lw=1.4)
        ax.plot([tau, tau], [y - h / 2 - 0.14, y + h / 2 + 0.14], color=RED, lw=1.0)
        ax.text(-0.25, y, f"$\\tau_{i+1}$", ha="right", va="center", fontsize=8.5,
                color=GREY)
    ax.text(1.5, n - 1, "données connues", ha="center", va="center", fontsize=7.2,
            color="white", weight="bold")
    ax.annotate("décision prise ici,\navec ce qui précède seulement",
                xy=(3.0, n - 1 + h / 2 + 0.10), xytext=(1.1, n - 0.30),
                fontsize=7.2, color=RED, ha="left",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.annotate("performance encaissée ensuite", xy=(5.6, n - 2),
                xytext=(6.6, n - 1.55), fontsize=7.2, color=EARN, ha="left",
                arrowprops=dict(arrowstyle="-", color=EARN, lw=0.8))
    ax.set_xlim(-0.9, 11.4)
    ax.set_ylim(-0.75, n + 0.55)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("temps  $\\longrightarrow$", labelpad=2)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Backtest à fenêtre glissante : le modèle est réentraîné à chaque "
                 "décision", fontsize=9.8, color=BLUE)
    ax.legend(handles=[Patch(fc=TRAIN, label="fenêtre d'entraînement (élargie à chaque pas)"),
                       Patch(fc=EARN, label="mois suivant : la performance est subie, pas choisie")],
              frameon=False, fontsize=7.4, loc="lower center",
              bbox_to_anchor=(0.5, -0.30), ncol=2)
    fig.savefig(OUT / "walkforward.pdf")
    plt.close(fig)
    print("  walkforward.pdf")


# ── 2. Purged + embargoed cross-validation ───────────────────────────────────
def validation_purgee():
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(7.6, 3.9),
                                     gridspec_kw={"height_ratios": [1, 1, 1.25]})
    T, h = 10.0, 0.5

    # (a) the frozen split
    a0.add_patch(Rectangle((0, -h / 2), 0.65 * T, h, fc="#9AA9C4", ec="none"))
    a0.add_patch(Rectangle((0.65 * T, -h / 2), 0.35 * T, h, fc="#4A4A4A", ec="none"))
    a0.text(0.325 * T, 0, "entraînement + validation (65 %)", ha="center", va="center",
            fontsize=7.4, color="white", weight="bold")
    a0.text(0.825 * T, 0, "test gelé (35 %)", ha="center", va="center", fontsize=7.4,
            color="white", weight="bold")
    a0.set_title("① Le segment de test est mis de côté avant toute calibration",
                 fontsize=8.8, color=BLUE, loc="left")

    # (b) naive K-fold — the test fold touches its training data
    for x0, x1, col in [(0, 3.8, TRAIN), (3.8, 5.6, "#4A4A4A"), (5.6, 10, TRAIN)]:
        a1.add_patch(Rectangle((x0, -h / 2), x1 - x0, h, fc=col, ec="none",
                               alpha=0.88 if col == TRAIN else 1))
    a1.text(4.7, 0, "test", ha="center", va="center", fontsize=7.4, color="white",
            weight="bold")
    a1.text(1.9, 0, "entraînement", ha="center", va="center", fontsize=7.2,
            color="white", weight="bold")
    a1.annotate("la veille du test sert\nà l'entraîner",
                xy=(3.8, h / 2 + 0.02), xytext=(0.3, 0.72), fontsize=7.2, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    a1.set_title("② Validation croisée standard : les blocs se touchent — fuite",
                 fontsize=8.8, color=BLUE, loc="left")

    # (c) purged + embargoed
    for x0, x1, col in [(0, 3.35, TRAIN), (3.35, 3.8, PURGE), (3.8, 5.6, "#4A4A4A"),
                        (5.6, 6.15, PURGE), (6.15, 10, TRAIN)]:
        a2.add_patch(Rectangle((x0, -h / 2), x1 - x0, h, fc=col, ec="none",
                               alpha=0.88 if col == TRAIN else 1))
    a2.text(1.7, 0, "entraînement", ha="center", va="center", fontsize=7.2,
            color="white", weight="bold")
    a2.text(4.7, 0, "test", ha="center", va="center", fontsize=7.4, color="white",
            weight="bold")
    a2.annotate("purge", xy=(3.57, -h / 2), xytext=(2.5, -1.05), fontsize=7.2,
                color=PURGE, arrowprops=dict(arrowstyle="->", color=PURGE, lw=0.9))
    a2.annotate("embargo", xy=(5.88, -h / 2), xytext=(6.3, -1.05), fontsize=7.2,
                color=PURGE, arrowprops=dict(arrowstyle="->", color=PURGE, lw=0.9))
    a2.set_title("③ Validation croisée purgée et sous embargo : les blocs sont séparés",
                 fontsize=8.8, color=BLUE, loc="left")

    for ax in (a0, a1, a2):
        ax.set_xlim(-0.2, T + 0.2)
        ax.set_ylim(-1.35, 1.35)
        ax.axis("off")
    a2.text(T / 2, -1.3, "temps  $\\longrightarrow$", ha="center", fontsize=7.6,
            color=GREY)
    fig.subplots_adjust(hspace=0.55)
    fig.savefig(OUT / "validation_purgee.pdf")
    plt.close(fig)
    print("  validation_purgee.pdf")


# ── 3. What the regime detector actually outputs ─────────────────────────────
def regimes_timeline():
    reg = pd.read_parquet(GOLD / "dashboard_regime.parquet")
    reg = reg[reg.universe == "etf_2017"].sort_values("Date").reset_index(drop=True)
    crises = json.loads((GOLD / "crisis_windows.json").read_text())["crises"]

    warm_end = reg.loc[~reg.converged, "Date"].max() if (~reg.converged).any() else None

    fig, ax = plt.subplots(figsize=(9.0, 2.9))
    for c in crises.values():
        s0, s1 = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
        ax.axvspan(s0, s1, color=RED, alpha=0.16, lw=0, zorder=1)
        ax.annotate(c["label"].replace("Global Financial Crisis", "Crise 2008")
                    .replace("EU sovereign debt crisis", "Dette eur.")
                    .replace("Q4 2018 selloff", "T4 2018")
                    .replace("COVID-19 crash", "COVID")
                    .replace("2022 rate shock", "Taux 2022"),
                    (s0 + (s1 - s0) / 2, 1.06), ha="center", fontsize=6.6, color=RED)

    # One bar per rebalance decision: the dispatch is discrete, so draw it discretely.
    w = 32  # days; the rebalance step is month-end, so the bars abut
    for _, r in reg.iterrows():
        bear = r.regime == "bear"
        ax.bar(r.Date, 0.55, bottom=0.12, width=w,
               color=RED if bear else "#3C6BA5",
               alpha=1.0 if r.converged else 0.35, lw=0, zorder=3)
    if warm_end is not None:
        ax.annotate("amorçage : sans historique suffisant,\nla posture par défaut est défensive",
                    (reg.Date.min(), 0.70), xytext=(reg.Date.min(), 0.86),
                    fontsize=6.6, color=GREY,
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.7))

    ax.set_ylim(0, 1.22)
    ax.set_yticks([])
    ax.set_xlim(reg.Date.min() - pd.Timedelta(days=140),
                reg.Date.max() + pd.Timedelta(days=140))
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Ce que produit le détecteur : un régime par décision, "
                 "estimé sans connaître la suite", fontsize=9.8, color=BLUE, pad=16)
    ax.legend(handles=[Patch(fc="#3C6BA5", label="régime « haussier » → optimiseur offensif"),
                       Patch(fc=RED, label="régime « baissier » → optimiseur défensif"),
                       Patch(fc=RED, alpha=0.20, label="crises (dates externes, fixées à l\'avance)")],
              frameon=False, fontsize=7.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.savefig(OUT / "regimes_timeline.pdf")
    plt.close(fig)
    n_bear = int((reg.regime == "bear").sum())
    print(f"  regimes_timeline.pdf   ({len(reg)} rééquilibrages, {n_bear} baissiers)")


if __name__ == "__main__":
    print("figures chapitre 4 :")
    walkforward(); validation_purgee(); regimes_timeline()
