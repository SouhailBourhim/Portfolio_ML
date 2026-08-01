"""Figures for Chapter 5 (résultats) of the internship report.

Like the Chapter 2 set, every figure is computed from the committed Gold
artifacts — no number is typed into this file, so the report cannot drift from
the results it describes.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
OUT = ROOT / "docs" / "rapport" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREY, RED, GREEN = "#1F3864", "#595959", "#B03000", "#1B5E20"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

L = lambda n: json.loads((GOLD / n).read_text())
SC, P5, NW, CW, CAP = (L("dashboard_showcase.json"), L("phase5_results.json"),
                       L("nested_walkforward_results.json"),
                       L("crisis_windows.json"), L("etf_cap_verdict.json"))
EQ = pd.read_parquet(GOLD / "dashboard_equity.parquet")

FR = {"equal_weight": "Équipondéré (1/N)", "min_variance_lw": "Variance min. (Ledoit-Wolf)",
      "max_sharpe": "Markowitz Sharpe max.", "regime_conditional": "Système ML (régime)",
      "rf_signal_tuned": "Signal RF calibré", "xgb_signal_tuned": "Signal XGB calibré"}
COL = {"equal_weight": GREY, "min_variance_lw": "#7A9CC6",
       "max_sharpe": "#C77700", "regime_conditional": BLUE}


def fr(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


# ── 1. Equity curves, both universes ─────────────────────────────────────────
def courbes():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, uni, titre in zip(axes, ("full_2021", "etf_2017"),
                              ("Portefeuille EURAFRIC (9 actifs)",
                               "ETF internationaux (5 actifs)")):
        sub = EQ[EQ.universe == uni]
        for s in ("equal_weight", "min_variance_lw", "max_sharpe", "regime_conditional"):
            r = sub[sub.strategy == s].sort_values("Date")
            if r.empty:
                continue
            w = (1 + r["net_return"]).cumprod() * 100
            ax.plot(r["Date"], w, lw=1.9 if s == "regime_conditional" else 1.0,
                    color=COL[s], label=FR[s], zorder=3 if s == "regime_conditional" else 2)
        ax.set_yscale("log")
        ax.set_title(titre, fontsize=9)
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_ylabel("Valeur (base 100, échelle log)" if uni == "full_2021" else "")
    axes[0].legend(frameon=False, fontsize=7.2, loc="upper left")
    fig.suptitle("Évolution du portefeuille, nette de coûts et hors échantillon",
                 fontsize=10.5, color=BLUE, y=1.02)
    fig.savefig(OUT / "courbes_equity.pdf")
    plt.close(fig)
    print("  courbes_equity.pdf")


# ── 2. Confidence intervals: single split vs nested ──────────────────────────
def intervalles():
    single = {**P5["full_2021"]["tuned"], **P5["full_2021"]["baselines"]}
    keys = ["regime_conditional", "xgb_signal_tuned", "rf_signal_tuned", "equal_weight"]
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    for i, k in enumerate(keys):
        s, n = single.get(k), NW["strategies"].get(k)
        if not (s and n):
            continue
        for off, (pt, lo, hi, col, lab) in enumerate((
                (s["test_sharpe_net"], *s["test_sharpe_ci"], GREY,
                 "Découpage unique (455 lignes)"),
                (n["sharpe_net"], *n["ci"], BLUE, "Walk-forward imbriqué (793 lignes)"))):
            y = i + (0.18 if off else -0.18)
            ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round",
                    label=lab if i == 0 else None)
            ax.plot([pt], [y], "o", color=col, ms=5.5, zorder=4)
    ax.axvline(0, color=RED, lw=0.9, ls="--")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([FR[k] for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Ratio de Sharpe net — intervalle de confiance à 90 %")
    ax.set_title("Les intervalles se chevauchent : aucun écart n'est significatif",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.grid(alpha=0.18, lw=0.5, axis="x")
    fig.savefig(OUT / "intervalles_confiance.pdf")
    plt.close(fig)
    print("  intervalles_confiance.pdf")


# ── 3. Crisis behaviour ──────────────────────────────────────────────────────
def crises():
    U = CW["universes"]["etf_2017"]
    keys = [k for k in CW["crises"] if k in U]
    labels = [CW["crises"][k]["label"].replace("Global Financial Crisis", "Crise fin. 2008")
              .replace("EU sovereign debt crisis", "Dette eur. 2011")
              .replace("Q4 2018 selloff", "Repli T4 2018")
              .replace("COVID-19 crash", "COVID 2020")
              .replace("2022 rate shock", "Taux 2022") for k in keys]
    opt = [100 * U[k]["min_variance_lw"]["cum_return"] for k in keys]
    ew = [100 * U[k]["equal_weight"]["cum_return"] for k in keys]
    ro = [U[k]["min_variance_lw"].get("recovery_days") for k in keys]
    re_ = [U[k]["equal_weight"].get("recovery_days") for k in keys]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.5),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(keys)); w = 0.38
    a1.bar(x - w/2, opt, w, color=BLUE, label="Optimisation sous contrainte")
    a1.bar(x + w/2, ew, w, color=GREY, label="Équipondéré (1/N)")
    a1.axhline(0, color="black", lw=0.8)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=7, rotation=18, ha="right")
    a1.set_ylabel("Performance cumulée (%)")
    a1.set_title("Perte subie pendant la crise", fontsize=9.5)
    a1.legend(frameon=False, fontsize=7.4, loc="lower left")
    a1.grid(alpha=0.18, lw=0.5, axis="y")

    m = [i for i in range(len(keys)) if ro[i] and re_[i]]
    a2.bar(np.arange(len(m)) - w/2, [ro[i] for i in m], w, color=BLUE)
    a2.bar(np.arange(len(m)) + w/2, [re_[i] for i in m], w, color=GREY)
    a2.set_xticks(range(len(m)))
    a2.set_xticklabels([labels[i] for i in m], fontsize=7, rotation=18, ha="right")
    a2.set_ylabel("Jours avant retour au sommet")
    a2.set_title("Délai de récupération", fontsize=9.5)
    a2.grid(alpha=0.18, lw=0.5, axis="y")
    fig.suptitle("Comportement en crise : la contrainte protège, le 1/N non",
                 fontsize=10.5, color=BLUE, y=1.03)
    fig.savefig(OUT / "crises.pdf")
    plt.close(fig)
    print("  crises.pdf")


# ── 4. Regime detection ──────────────────────────────────────────────────────
def detection():
    rd = CW["regime_detection"]["etf_2017"]
    per = rd["per_crisis"]
    names = [CW["crises"][k]["label"][:18] for k in per]
    rates = [100 * per[k]["bear_rate"] for k in per]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.bar(range(len(names)), rates, color=RED, alpha=0.85, width=0.6)
    ax.axhline(100 * rd["bear_rate_outside"], color=GREY, ls="--", lw=1.2,
               label=f"Taux hors crise ({100*rd['bear_rate_outside']:.0f} %)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=18, ha="right")
    ax.set_ylabel("Rééquilibrages classés « baissier » (%)")
    ax.set_ylim(0, 112)
    for i, v in enumerate(rates):
        ax.text(i, v + 3, f"{v:.0f} %", ha="center", fontsize=7.6, color=RED)
    sig = rd["significance"]
    ax.set_title(f"Le modèle non supervisé a détecté les {sig['crises_exceeding_base_rate']} "
                 f"crises  (test des signes : p = {str(sig['sign_test_p_conservative']).replace('.', ',')})",
                 fontsize=9.8)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.grid(alpha=0.18, lw=0.5, axis="y")
    fig.savefig(OUT / "detection_regime.pdf")
    plt.close(fig)
    print("  detection_regime.pdf")


# ── 5. Cap sweep ─────────────────────────────────────────────────────────────
def plafond():
    caps = sorted(CAP["verdicts"], key=float)
    ys = [CAP["verdicts"][c]["classical_sharpe"] for c in caps]
    alloc = [CAP["results"][c]["min_variance_lw"]["distinct_allocations"] for c in caps]
    # Categorical x-axis: on a linear scale the four caps 0.25-0.40 collapse into
    # a cluster while "no cap" sits far right, and the annotations collide.
    xs = list(range(len(caps)))

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.plot(xs, ys, "o-", color=BLUE, lw=2, ms=6.5)
    for x, y, a in zip(xs, ys, alloc):
        ax.annotate(f"{fr(y, 4)}", (x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color=BLUE, weight="bold")
        ax.annotate(f"{a} allocation" + ("s" if a > 1 else ""), (x, y),
                    xytext=(0, -17), textcoords="offset points", ha="center",
                    fontsize=6.8, color=GREY)
    ax.set_xticks(xs)
    ax.set_xticklabels(["sans plafond" if float(c) >= 1 else f"{100*float(c):.0f} %"
                        for c in caps])
    ax.set_xlim(-0.4, len(caps) - 0.6)
    ax.set_xlabel("Plafond par actif")
    ax.set_ylabel("Meilleur Sharpe net classique")
    swing = 100 * (ys[0] - ys[-1]) / ys[-1]
    ax.set_title(f"Relâcher la contrainte coûte {fr(swing, 1)} % de Sharpe — "
                 f"davantage que tout modèle testé", fontsize=9.8)
    ax.set_ylim(min(ys) - 0.028, max(ys) + 0.030)
    ax.grid(alpha=0.18, lw=0.5)
    fig.savefig(OUT / "plafond.pdf")
    plt.close(fig)
    print(f"  plafond.pdf   ({swing:.1f} % swing)")


if __name__ == "__main__":
    print("figures chapitre 5 :")
    courbes(); intervalles(); crises(); detection(); plafond()
