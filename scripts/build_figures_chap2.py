"""Generate the figures for Chapter 2 of the internship report.

Every figure is computed from the project's OWN committed Gold data — not
illustrative sketches — so the report shows the reader the actual phenomena the
system was built to address.

Outputs vector PDFs into docs/rapport/assets/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
OUT = ROOT / "docs" / "rapport" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREY, RED, AMBER = "#1F3864", "#595959", "#B03000", "#C77700"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

RET = pd.read_parquet(GOLD / "log_returns_etf.parquet")
FEA = pd.read_parquet(GOLD / "ml_features_etf.parquet")
TD = 252

# Crisis windows — the same external S&P 500 peak-to-trough dates used in the
# crisis-window study, so the report stays internally consistent.
CRISES = [("2007-10-09", "2009-03-09", "Crise financière\n2008"),
          ("2011-04-29", "2011-10-03", "Dette\neuropéenne"),
          ("2020-02-19", "2020-03-23", "COVID"),
          ("2022-01-03", "2022-10-12", "Choc de taux\n2022")]


def shade(ax, label=True):
    for a, b, name in CRISES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=RED, alpha=0.10, lw=0)
        if label:
            ax.annotate(name, xy=(pd.Timestamp(a), ax.get_ylim()[1]),
                        xytext=(0, -2), textcoords="offset points",
                        fontsize=6.5, color=RED, ha="left", va="top")


# ── 1. Efficient frontier ────────────────────────────────────────────────────
def frontiere():
    simple = np.exp(RET) - 1.0
    mu = simple.mean().to_numpy() * TD
    cov = simple.cov().to_numpy() * TD
    n = len(mu)
    rng = np.random.default_rng(0)

    w = rng.dirichlet(np.ones(n), 8000)
    r = w @ mu
    v = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.scatter(100 * v, 100 * r, s=3, c=GREY, alpha=0.16, lw=0,
               label="Portefeuilles possibles (tirage aléatoire)")

    # Efficient frontier = upper-left envelope, swept from the minimum-variance
    # portfolio RIGHTWARD. A running maximum over volatility-sorted points gives
    # exactly that; sweeping the other way (as a first attempt did) traces only
    # the high-risk tail and leaves the curve visibly detached from the cloud.
    order = np.argsort(v)
    vs, rs = v[order], r[order]
    start = int(np.argmax(rs[:200]))
    fv, fr, best = [], [], -np.inf
    for vv, rr in zip(vs[start:], rs[start:]):
        if rr > best:
            best = rr
            fv.append(vv); fr.append(rr)
    ax.plot(100 * np.array(fv), 100 * np.array(fr), color=BLUE, lw=2.2,
            label="Frontière efficiente")
    ax.scatter([100 * vs[start]], [100 * rs[start]], s=34, facecolor="white",
               edgecolor=BLUE, zorder=6, lw=1.4)
    ax.annotate("variance minimale", (100 * vs[start], 100 * rs[start]),
                xytext=(6, -12), textcoords="offset points", fontsize=7.4,
                color=BLUE)

    ivol = np.sqrt(np.diag(cov))
    ax.scatter(100 * ivol, 100 * mu, s=46, c=RED, zorder=5,
               label="Actifs pris isolément")
    for i, name in enumerate(RET.columns):
        ax.annotate(name, (100 * ivol[i], 100 * mu[i]), xytext=(6, -2),
                    textcoords="offset points", fontsize=8, color=RED)

    ax.margins(x=0.09)
    ax.set_xlabel("Risque — volatilité annualisée (%)")
    ax.set_ylabel("Rendement annualisé espéré (%)")
    ax.set_title("Rendement contre risque : la frontière efficiente")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(alpha=0.18, lw=0.5)
    fig.savefig(OUT / "frontiere_efficiente.pdf")
    plt.close(fig)
    print("  frontiere_efficiente.pdf")


# ── 2. Volatility regimes ────────────────────────────────────────────────────
def volatilite():
    vol = 100 * FEA["MARKET_VOL_SHORT"].dropna()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.plot(vol.index, vol.values, color=BLUE, lw=0.7)
    ax.axhline(vol.mean(), color=GREY, ls="--", lw=0.9,
               label=f"Moyenne ({vol.mean():.0f} %)")
    ax.set_ylim(0, min(vol.max() * 1.05, 90))
    shade(ax)
    ax.set_ylabel("Volatilité annualisée (%)")
    ax.set_title("La volatilité se regroupe : les marchés changent de régime (P2)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.18, lw=0.5)
    fig.savefig(OUT / "volatilite_regimes.pdf")
    plt.close(fig)
    print("  volatilite_regimes.pdf")


# ── 3. Correlation breakdown, DECOMPOSED ─────────────────────────────────────
def correlations():
    """The naive average over all pairs HIDES the phenomenon on this universe.

    Among equities, correlation rises in a crisis exactly as theory predicts
    (0.80 -> 0.88, and 0.94 during COVID). But gold and long Treasuries
    DECOUPLE harder at the same moment (-0.03 -> -0.19), so averaging every
    pair together cancels the two effects and shows a spurious *fall*. The
    figure therefore plots the two families separately — which is both the
    honest presentation and the more informative one.
    """
    import itertools
    simple = np.exp(RET) - 1.0
    EQ, DIV = ["SPY", "QQQ", "EEM"], ["GLD", "TLT"]
    pairs_eq = list(itertools.combinations(EQ, 2))
    pairs_x = [(e, d) for e in EQ for d in DIV]
    W = 63

    def rolling_avg(pairs):
        s = sum(simple[a].rolling(W).corr(simple[b]) for a, b in pairs)
        return (s / len(pairs)).dropna()

    eq, cross = rolling_avg(pairs_eq), rolling_avg(pairs_x)
    m = np.zeros(len(eq), bool)
    for a, b, _ in CRISES:
        m |= (eq.index >= pd.Timestamp(a)) & (eq.index <= pd.Timestamp(b))

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(eq.index, eq.values, color=RED, lw=0.8,
            label="Entre actions (SPY, QQQ, EEM)")
    ax.plot(cross.index, cross.values, color=BLUE, lw=0.8,
            label="Actions contre or et obligations (GLD, TLT)")
    ax.axhline(eq[~m].mean(), color=RED, ls="--", lw=0.8, alpha=0.6)
    ax.axhline(cross[~m].mean(), color=BLUE, ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0, color=GREY, lw=0.6)
    shade(ax, label=False)
    for a, b, name in CRISES:
        ax.annotate(name.replace(chr(10), " "), xy=(pd.Timestamp(a), 1.04),
                    fontsize=6.2, color=RED, ha="left", va="bottom")
    ax.set_ylim(-0.78, 1.20)
    ax.set_ylabel("Corrélation glissante (63 j)")
    ax.set_title("En crise, les actions se resserrent — et les valeurs refuges "
                 "s'en détachent (P3)")
    ax.legend(frameon=True, framealpha=0.93, edgecolor="none",
              fontsize=7.6, loc="lower left")
    ax.grid(alpha=0.18, lw=0.5)
    fig.savefig(OUT / "correlations_crise.pdf")
    plt.close(fig)
    print(f"  correlations_crise.pdf   actions {eq[~m].mean():.2f}->{eq[m].mean():.2f} | "
          f"refuges {cross[~m].mean():.2f}->{cross[m].mean():.2f}")


# ── 4. The four problems ─────────────────────────────────────────────────────
def quatre_problemes():
    items = [
        ("P1", "Covariance bruitée",
         "L'optimiseur traite des estimations imprécises comme exactes et amplifie le bruit.",
         "Rétrécissement Ledoit-Wolf, EWMA, DCC-GARCH"),
        ("P2", "Non-stationnarité",
         "Les paramètres estimés en période calme ne décrivent pas la tension qui suit.",
         "Détection de régimes par chaîne de Markov cachée"),
        ("P3", "Rupture en crise",
         "Au sein d'une classe d'actifs, tout chute ensemble : 0,79 vers 0,94 entre actions en mars 2020.",
         "Covariance dynamique et posture défensive par régime"),
        ("P4", "Surapprentissage",
         "Biais d'anticipation et sélection répétée font paraître excellente une stratégie sans valeur.",
         "Moteur sans fuite, K-Fold purgé, Sharpe déflaté, intervalles"),
    ]
    import textwrap
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    WB, GAP = 2.28, 2.45
    x0 = (10 - (3 * GAP + WB)) / 2
    for i, (tag, title, prob, sol) in enumerate(items):
        x = x0 + GAP * i
        ax.add_patch(FancyBboxPatch((x, 4.15), WB, 4.7,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc="#EEF2F8", ec=BLUE, lw=1.1))
        ax.text(x + WB / 2, 8.32, tag, ha="center", fontsize=13, color=BLUE, weight="bold")
        ax.text(x + WB / 2, 7.70, title, ha="center", fontsize=8.2, color=BLUE, weight="bold")
        ax.text(x + WB / 2, 5.85, "\n".join(textwrap.wrap(prob, 31)), ha="center",
                va="center", fontsize=6.3, color="#333333", linespacing=1.4)
        ax.annotate("", xy=(x + WB / 2, 3.10), xytext=(x + WB / 2, 4.05),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1))
        ax.add_patch(FancyBboxPatch((x, 1.35), WB, 1.7,
                                    boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc="white", ec=GREY, lw=0.9))
        ax.text(x + WB / 2, 2.2, "\n".join(textwrap.wrap(sol, 31)), ha="center",
                va="center", fontsize=6.3, color="#222222", linespacing=1.4)
    ax.text(5.0, 9.32, "Les quatre échecs de l'optimisation classique",
            ha="center", fontsize=11, color=BLUE, weight="bold")
    ax.text(5.0, 0.55, "Réponse apportée par le système",
            ha="center", fontsize=8.2, color=GREY, style="italic")
    fig.savefig(OUT / "quatre_problemes.pdf")
    plt.close(fig)
    print("  quatre_problemes.pdf")


if __name__ == "__main__":
    print("figures chapitre 2 :")
    frontiere()
    volatilite()
    correlations()
    quatre_problemes()
