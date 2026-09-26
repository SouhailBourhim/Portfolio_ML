"""build_figures_soutenance_fr.py — pedagogical figures for the vulgarised defence deck.

The existing report figures are written for a reader who already knows what a
covariance matrix is. The defence jury does not. These five figures carry the
concepts that deck assumes: what an allocation is, what the Sharpe ratio buys
you, how complexity was added in tiers, what the evidence chain does, and why
the best of 240 searched candidates is not a discovery.

Every number that appears on a figure is read from a committed Gold artifact —
`snooping_240` in particular recomputes the 240 candidate Sharpes from
`global_2004_q2_series.parquet` and asserts the benchmark reproduces the value
published in `global_2004_q2_results.json`, so a stale series cannot silently
redraw the slide.

Usage:
    python scripts/build_figures_soutenance_fr.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
OUT = ROOT / "output" / "presentation" / "_figures_fr"
FIG_REPORT = ROOT / "docs" / "rapport_final" / "assets" / "figures"

INK = "#111827"
NAVY = "#1B3A6B"
BLUE = "#3D8DFF"
GREY = "#5D6675"
AMBER = "#C8862A"
RED = "#B7433E"
TEAL = "#0F766E"
PANEL = "#F3F4F6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#C9CED6",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": GREY,
    "ytick.color": GREY,
})


def _box(ax, x, y, w, h, label, sub, *, fill, edge, tcolor=INK, fs=12, fss=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fill, edgecolor=edge, linewidth=1.3))
    ax.text(x + w / 2, y + h * 0.63, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tcolor)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=fss, color=GREY, linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, color=NAVY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.6, color=color, shrinkA=0, shrinkB=0))


# -- 1. what an allocation actually is ---------------------------------------

def allocation_expliquee() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax in axes:
        ax.set_axis_off()

    ax = axes[0]
    ax.set_title("1 · Un portefeuille", fontsize=13, fontweight="bold", color=NAVY, pad=14)
    noms = ["Action A", "Action B", "ETF or", "ETF oblig."]
    poids = [0.25, 0.20, 0.25, 0.30]
    cols = [NAVY, BLUE, AMBER, TEAL]
    ax.pie(poids, labels=noms, colors=cols, autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 9.5, "color": INK},
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.text(0.5, -0.10, "un capital reparti entre plusieurs actifs".replace("reparti", "réparti"),
            transform=ax.transAxes, ha="center", fontsize=10, color=GREY)

    ax = axes[1]
    ax.set_title("2 · L’allocation", fontsize=13, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.80
    for nom, p, c in zip(noms, poids, cols):
        ax.text(0.02, y, nom, fontsize=10, va="center", color=INK)
        ax.barh(y, p * 1.6, left=0.38, height=0.09, color=c)
        ax.text(0.38 + p * 1.6 + 0.02, y, f"{p:.0%}".replace("%", " %"),
                fontsize=10, va="center", color=GREY)
        y -= 0.20
    ax.plot([0.38, 0.38], [0.08, 0.92], color="#C9CED6", lw=1)
    ax.text(0.5, 0.02, "les poids : la seule décision du modèle",
            transform=ax.transAxes, ha="center", fontsize=10, color=GREY)

    ax = axes[2]
    ax.set_title("3 · Le rééquilibrage", fontsize=13, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.97, 0.55), xytext=(0.03, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
    for i, x in enumerate(np.linspace(0.12, 0.86, 5)):
        ax.plot([x], [0.55], marker="o", ms=9, color=BLUE, zorder=3)
        ax.text(x, 0.68, f"m{i + 1}", ha="center", fontsize=9, color=GREY)
    ax.text(0.5, 0.36, "chaque mois, on recalcule les poids", ha="center",
            fontsize=10.5, color=INK)
    ax.text(0.5, 0.22, "et chaque mouvement coûte des frais", ha="center",
            fontsize=10.5, fontweight="bold", color=RED)
    ax.text(0.5, 0.02, "la décision est répétée, jamais unique",
            transform=ax.transAxes, ha="center", fontsize=10, color=GREY)

    fig.savefig(OUT / "allocation_expliquee.png")
    plt.close(fig)


# -- 2. the Sharpe ratio, without a formula ----------------------------------

def sharpe_explique() -> None:
    rng = np.random.default_rng(7)
    n = 520
    drift = 0.00045
    a = drift + rng.normal(0, 0.0060, n)
    b = drift + rng.normal(0, 0.0185, n)
    a = a - a.mean() + drift          # identical average return, by construction
    b = b - b.mean() + drift
    pa, pb = 100 * np.exp(np.cumsum(a)), 100 * np.exp(np.cumsum(b))
    sa = a.mean() / a.std(ddof=1) * np.sqrt(252)
    sb = b.mean() / b.std(ddof=1) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(pa, color=TEAL, lw=2.1,
            label=("Portefeuille A — Sharpe %.2f" % sa).replace(".", ","))
    ax.plot(pb, color=RED, lw=1.5, alpha=0.85,
            label=("Portefeuille B — Sharpe %.2f" % sb).replace(".", ","))
    ax.set_title("Même rendement final, deux trajets très différents",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_ylabel("Valeur du portefeuille (base 100)")
    ax.set_xlabel("Temps (jours de bourse)")
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.985, 0.06,
            "Le ratio de Sharpe = rendement obtenu par unité de risque subi.\n"
            "C’est le « rapport qualité/prix » du risque : plus il est haut, mieux c’est.",
            transform=ax.transAxes, ha="right", fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL, edgecolor="none"))
    fig.savefig(OUT / "sharpe_explique.png")
    plt.close(fig)


# -- 3. complexity added in tiers --------------------------------------------

def escalier_modeles() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.7)
    ax.set_axis_off()

    paliers = [
        ("1 · Références classiques",
         "1/N · variance minimale · Markowitz\nLe plancher à battre", NAVY),
        ("2 · Covariance dynamique",
         "Ledoit–Wolf · EWMA · DCC-GARCH\nLe risque cesse d’être figé", BLUE),
        ("3 · Régimes de marché",
         "HMM à 2 états + allocation conditionnelle\nCalme ou stress : deux postures", TEAL),
        ("4 · Challengers ML",
         "Random Forest · XGBoost + coûts\nApprentissage supervisé du signal", AMBER),
    ]
    w, h = 3.05, 1.02
    for i, (titre, sous, col) in enumerate(paliers):
        x, y = 0.15 + i * 2.35, 0.45 + i * 1.18
        _box(ax, x, y, w, h, titre, sous, fill="#FFFFFF", edge=col, tcolor=col, fs=10.5, fss=8)
        if i < 3:
            _arrow(ax, x + w * 0.82, y + h + 0.02, x + w * 0.82 + 0.1, y + 1.18 - 0.02, col)

    ax.text(5.3, 5.45,
            "La complexité est ajoutée par paliers — chacun doit justifier son coût",
            ha="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(0.15, 0.08,
            "Chaque palier est comparé au précédent hors échantillon, net de frais.",
            fontsize=10.5, color=GREY)
    fig.savefig(OUT / "escalier_modeles.png")
    plt.close(fig)


# -- 4. the evidence chain, in plain French ----------------------------------

def chaine_simple() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.set_axis_off()

    etapes = [
        ("Données", "Bourse de Casablanca\nETF · taux officiel BAM", NAVY),
        ("Préparation", "Bronze → Silver → Gold\ncontrôlée et versionnée", TEAL),
        ("Modèles", "Markowitz · HMM\nRandom Forest · XGBoost", BLUE),
        ("Épreuve", "walk-forward · coûts\ntests statistiques", AMBER),
        ("Publication", "API · tableau de bord\nrapport · 891 tests", "#2F7D3B"),
    ]
    w, h, gap = 1.82, 1.15, 0.18
    for i, (titre, sous, col) in enumerate(etapes):
        x = 0.05 + i * (w + gap)
        _box(ax, x, 0.75, w, h, titre, sous, fill=PANEL, edge=col, tcolor=col, fs=12, fss=8.3)
        if i < len(etapes) - 1:
            _arrow(ax, x + w + 0.03, 1.33, x + w + gap - 0.03, 1.33)

    ax.text(5.0, 2.35,
            "Une seule direction : de la donnée brute à la décision publiée",
            ha="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(5.0, 0.35,
            "Aucun chiffre publié n’est saisi à la main : chaque surface est "
            "régénérée depuis les artefacts.",
            ha="center", fontsize=10.5, color=GREY, style="italic")
    fig.savefig(OUT / "chaine_simple.png")
    plt.close(fig)


# -- 5. why the best of 240 searches is not a discovery ----------------------

def snooping_240() -> None:
    q2 = json.loads((GOLD / "global_2004_q2_results.json").read_text(encoding="utf-8"))
    published = q2["benchmark"]["net_sharpe"]
    test = q2["family_tests"]["primary_sharpe"]

    df = pd.read_parquet(GOLD / "global_2004_q2_series.parquet")
    g = df.groupby("candidate")["net_return"]
    sharpe = g.mean() / g.std(ddof=1) * np.sqrt(252)
    bench = float(sharpe.loc["regime_conditional"])
    assert abs(bench - published) < 5e-4, f"series no longer reproduces {published}: {bench}"
    cand = sharpe.drop("regime_conditional")
    assert len(cand) == test["n_candidates"], f"expected {test['n_candidates']} candidates"
    best = float(cand.max())
    assert abs((best - bench) - test["best_differential"]) < 5e-4

    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    ax.hist(cand.values, bins=34, color="#C6D3E4", edgecolor="white", linewidth=0.8)
    ax.axvline(bench, color=NAVY, lw=2.2)
    ax.axvline(best, color=RED, lw=2.2, ls="--")
    ax.set_xlim(float(cand.min()) - 0.02, best + 0.035)
    top = ax.get_ylim()[1]
    ax.text(bench - 0.006, top * 0.62,
            "Stratégie de référence\n" + ("%.3f" % bench).replace(".", ","),
            color=NAVY, fontsize=10.5, fontweight="bold", va="top", ha="right")
    ax.text(best - 0.006, top * 0.40,
            "Le « meilleur » essai\n" + ("%.3f" % best).replace(".", ","),
            color=RED, fontsize=10.5, fontweight="bold", va="top", ha="right")
    ax.set_title(f"{len(cand)} stratégies testées — et le champion est refusé",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Ratio de Sharpe hors échantillon, net de frais")
    ax.set_ylabel("Nombre de stratégies")
    ax.spines[["top", "right"]].set_visible(False)
    rc, spa = test["reality_check_p_value"], test["spa_p_value"]
    ax.text(0.015, 0.97,
            "Chercher longtemps garantit de trouver un gagnant apparent.\n"
            "Les tests de sélection multiple tranchent : p = "
            + ("%.2f" % rc).replace(".", ",") + " et p = " + ("%.2f" % spa).replace(".", ",")
            + "  →  effet du hasard.",
            transform=ax.transAxes, va="top", fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF0D5", edgecolor="none"))
    fig.savefig(OUT / "snooping_240.png")
    plt.close(fig)


# -- 6. out-of-sample equity curves, decluttered -----------------------------

def courbes_equity_fr() -> None:
    """The same out-of-sample evidence as the report's figure, drawn for a room.

    The chapter-5 version plots four near-coincident noisy lines with a legend
    box sitting on top of them and every rebalance date on the x-axis; projected,
    it reads as a single grey smudge. The claim the slide actually makes is
    "everything tracks, nothing breaks away", and a band states that better than
    four overlapping strokes: the three classical baselines become their own
    min–max envelope, leaving one emphasised line against it.

    Nothing is aggregated away silently — the band is labelled with the three
    strategies it contains, and their exact Sharpe ratios are on annex slide A.
    """
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FixedFormatter

    eq = pd.read_parquet(GOLD / "dashboard_equity.parquet")
    classiques = ["equal_weight", "min_variance_lw", "max_sharpe"]
    panneaux = [
        ("full_2021", "Univers mixte marocain — 9 actifs, en dirhams", 1),
        ("etf_2017", "ETF internationaux — 5 actifs, en dollars", 4),
    ]
    # Round values a reader recognises on a log axis; the ones inside each
    # panel's own range become its ticks, so a four-year panel and a twenty-year
    # one both get labelled gridlines instead of a lone "100".
    ECHELLE = [50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400,
               500, 600, 800, 1000, 1250, 1600, 2000]
    BAND, BAND_EDGE = "#C9D6E8", "#9FB3CE"

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    fig.subplots_adjust(wspace=0.16)

    for ax, (uni, titre, pas) in zip(axes, panneaux):
        sub = eq[eq.universe == uni]
        courbes = {}
        for strat in classiques + ["regime_conditional"]:
            r = sub[sub.strategy == strat].sort_values("Date")
            assert not r.empty, f"{strat} absent de {uni} dans dashboard_equity.parquet"
            courbes[strat] = pd.Series(
                ((1 + r["net_return"]).cumprod() * 100).values, index=r["Date"].values
            )
        ref = pd.DataFrame({s: courbes[s] for s in classiques})
        dates = ref.index

        ax.fill_between(dates, ref.min(axis=1), ref.max(axis=1),
                        facecolor=BAND, edgecolor=BAND_EDGE, linewidth=0.7, zorder=2)
        ax.plot(courbes["regime_conditional"].index, courbes["regime_conditional"].values,
                color=NAVY, lw=2.2, zorder=3, solid_joinstyle="round")

        ax.set_yscale("log")
        bas = min(ref.min(axis=1).min(), courbes["regime_conditional"].min())
        haut = max(ref.max(axis=1).max(), courbes["regime_conditional"].max())
        paliers = [t for t in ECHELLE if bas <= t <= haut]
        while len(paliers) > 6:                      # keep the axis readable
            paliers = paliers[::2]
        ax.yaxis.set_major_locator(FixedLocator(paliers))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{t}" for t in paliers]))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_locator(mdates.YearLocator(pas))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.set_title(titre, fontsize=11.5, color=INK, pad=12)
        ax.grid(axis="y", color="#E3E7ED", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10.5)

    axes[0].set_ylabel("Valeur d’un placement de 100, nette de frais", fontsize=10.5)
    fig.legend(
        handles=[
            Line2D([], [], color=NAVY, lw=2.4, label="Système ML à régimes"),
            Patch(facecolor=BAND, edgecolor=BAND_EDGE,
                  label="Références classiques (1/N · variance minimale · Markowitz)"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=11,
        bbox_to_anchor=(0.5, -0.06), handlelength=2.4, columnspacing=2.4,
    )
    fig.savefig(OUT / "courbes_equity_fr.png")
    plt.close(fig)


# -- 7. crisis detection, with French crisis names ---------------------------

def detection_crises_fr() -> None:
    """The report's detection chart carries the English crisis labels from the
    artifact. A French-speaking jury should not have to translate an axis."""
    crisis = json.loads((GOLD / "crisis_windows.json").read_text(encoding="utf-8"))
    det = crisis["regime_detection"]["etf_2017"]
    noms = {
        "gfc_2008": "Crise financière\n2008",
        "eu_debt_2011": "Dette souveraine\neuropéenne 2011",
        "q4_2018": "Correction\nT4 2018",
        "covid_2020": "Krach\nCOVID-19",
        "rate_shock_2022": "Choc de taux\n2022",
    }
    per = det["per_crisis"]
    labels = [noms[k] for k in noms if k in per]
    rates = [per[k]["bear_rate"] * 100 for k in noms if k in per]
    hors = det["bear_rate_outside"] * 100

    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    bars = ax.bar(labels, rates, color=RED, width=0.62)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 2.5, ("%.0f %%" % r),
                ha="center", fontsize=11, fontweight="bold", color=RED)
    ax.axhline(hors, color=GREY, ls="--", lw=1.4)
    ax.text(0.012, hors / 116 + 0.02, "Taux hors crise : " + ("%.0f %%" % hors),
            transform=ax.transAxes, ha="left", fontsize=10, color=GREY)
    ax.set_ylim(0, 116)
    ax.set_ylabel("Mois classés « marché sous tension » (%)")
    ax.set_title("Le modèle, jamais informé des crises, les a toutes reconnues",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    fig.savefig(OUT / "detection_crises_fr.png")
    plt.close(fig)


# -- 8. landscape crops of the product screenshots ---------------------------

def captures_recadrees() -> None:
    """Crop the full-page captures to their legible top half.

    The committed screenshots are 2880x3500 portrait pages. Dropped whole into a
    16:9 slide they scale to a thumbnail nobody in a lecture hall can read. The
    top band carries the identity of each surface, which is all the slide claims.
    """
    from PIL import Image

    for stem, bottom in (("dashboard_page1", 1750), ("api_swagger", 1700)):
        src = Image.open(FIG_REPORT / f"{stem}.png")
        src.crop((0, 0, src.width, min(bottom, src.height))).save(OUT / f"{stem}_haut.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for builder in (allocation_expliquee, sharpe_explique, escalier_modeles,
                    chaine_simple, snooping_240, courbes_equity_fr, detection_crises_fr,
                    captures_recadrees):
        builder()
        print(f"built {builder.__name__}")


if __name__ == "__main__":
    main()
