"""Build the artifact-driven figures used only by the final PFA report.

The existing report already contains the market and modelling figures.  This
builder adds the cross-cutting evidence that became available after the MAD/BAM
release: currency correctness, claim revisions, the proof chain, multiple-
testing correction, exact explainability, and industrial delivery coverage.

Every result-shaped number is read from ``data/gold``.  Historical superseded
values are deliberately labelled and exist only in the revision figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
OUT = ROOT / "docs" / "rapport_final" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1F4E79"
NAVY = "#17365D"
TEAL = "#0F6B66"
GREEN = "#2E7D32"
RED = "#B03000"
AMBER = "#B97900"
GREY = "#5F6368"
LIGHT = "#F4F7FA"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    }
)


def load(name: str):
    return json.loads((GOLD / name).read_text())


def fr(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def rounded_box(ax, xy, width, height, title, body, color=BLUE, title_size=9):
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        facecolor=color,
        alpha=0.09,
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=color)
    ax.text(x + width / 2, y + height * 0.29, body, ha="center", va="center", fontsize=7.2, color=GREY, linespacing=1.3)


def architecture_globale() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    columns = [
        (0.2, "SOURCES", [
            ("BVC", "Cours + dividendes"),
            ("Yahoo/FRED", "ETF + macro global"),
            ("Bank Al-Maghrib", "USD/MAD officiel"),
        ]),
        (2.75, "DONNEES", [
            ("Bronze", "Brut, persistant"),
            ("Silver", "Calendriers, MAD, contrats"),
            ("Gold", "Rendements, features, preuves"),
        ]),
        (5.3, "RECHERCHE", [
            ("Backtest", "Walk-forward causal"),
            ("Modeles", "Markowitz, HMM, RF, XGB"),
            ("Selection", "Forward-only + embargo"),
        ]),
        (7.85, "VALIDATION", [
            ("Comparaison pairee", "Bootstrap par blocs"),
            ("Data snooping", "White RC + Hansen SPA"),
            ("Gouvernance", "Cartes, provenance, telemetry"),
        ]),
        (10.4, "PUBLICATION", [
            ("Dashboard", "Deux vues, memes artefacts"),
            ("API REST", "Lecture seule, contrat devise"),
            ("Rapport", "Faits generes + controle PDF"),
        ]),
    ]

    colors = [GREY, TEAL, BLUE, AMBER, GREEN]
    for col_idx, (x, label, items) in enumerate(columns):
        ax.text(x + 0.9, 7.55, label, ha="center", fontsize=9, fontweight="bold", color=colors[col_idx])
        for i, (title, body) in enumerate(items):
            y = 5.75 - i * 1.78
            rounded_box(ax, (x, y), 1.8, 1.15, title, body, colors[col_idx], 8.3)
        if col_idx < len(columns) - 1:
            ax.annotate("", xy=(x + 2.43, 4.78), xytext=(x + 1.88, 4.78), arrowprops=dict(arrowstyle="->", lw=1.5, color=NAVY))

    ax.text(6, 0.42, "Une seule direction de preuve : source -> decision -> validation -> surface publiee", ha="center", color=NAVY, fontsize=10, fontweight="bold")
    ax.set_title("Architecture fonctionnelle du prototype et chaine de preuve", fontsize=13, fontweight="bold", color=NAVY, pad=14)
    fig.savefig(OUT / "architecture_globale.pdf")
    plt.close(fig)


def revisions_resultat() -> None:
    showcase = load("dashboard_showcase.json")
    current = float(showcase["universes"]["full_2021"]["headline_lift_pct"])
    stages = [
        ("Mesure initiale", 14.3, "Prix BVC hors dividendes"),
        ("Rendement total", 6.2, "Dividendes reintegres"),
        ("Numeraire MAD", current, "Taux BAM officiel"),
    ]
    fig, ax = plt.subplots(figsize=(9.3, 4.7))
    x = np.arange(3)
    y = [row[1] for row in stages]
    ax.axhline(0, color="black", lw=0.9)
    ax.plot(x, y, color=NAVY, lw=2.2)
    for i, (label, value, cause) in enumerate(stages):
        color = RED if i == 2 else GREY
        ax.scatter(i, value, s=145, color=color, zorder=3)
        ax.text(i, value + (2.7 if value >= 0 else -3.8), f"{fr(value)} %", ha="center", fontsize=14, fontweight="bold", color=color)
        if i < 2:
            ax.text(i, value + 5.3, "HISTORIQUE - SUPERSEDED", ha="center", fontsize=7, color=GREY)
        ax.text(i, -18.4, label, ha="center", fontsize=9, fontweight="bold")
        ax.text(i, -20.8, cause, ha="center", fontsize=7.7, color=GREY)
    ax.annotate("changement de signe", xy=(2, current), xytext=(1.15, -3.5), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), color=RED, fontweight="bold")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-24, 23)
    ax.set_xticks([])
    ax.set_ylabel("Ecart ponctuel regime - meilleur classique (%)")
    ax.set_title("Le meme systeme, mesure trois fois : la qualite des donnees change le verdict", color=NAVY, fontweight="bold")
    fig.savefig(OUT / "revisions_resultat.pdf")
    plt.close(fig)


def numeraire() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, valid in zip(axes, [False, True]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        color = GREEN if valid else RED
        ax.set_title("Apres : portefeuille en MAD" if valid else "Avant : addition de deux devises", color=color, fontweight="bold")
        rounded_box(ax, (0.4, 6.6), 3.9, 1.55, "4 actions BVC", "rendements en MAD", TEAL)
        rounded_box(ax, (5.7, 6.6), 3.9, 1.55, "5 ETF internationaux", "convertis en MAD" if valid else "rendements en USD", GREEN if valid else RED)
        if valid:
            rounded_box(ax, (5.9, 4.5), 3.5, 1.0, "Conversion causale", "niveaux ETF x USD/MAD BAM", BLUE, 8)
            ax.annotate("", xy=(7.65, 5.55), xytext=(7.65, 6.55), arrowprops=dict(arrowstyle="->", color=BLUE))
        ax.annotate("", xy=(5, 3.1), xytext=(2.35, 6.55), arrowprops=dict(arrowstyle="->", color=GREY))
        ax.annotate("", xy=(5, 3.1), xytext=(7.65, 4.45 if valid else 6.55), arrowprops=dict(arrowstyle="->", color=GREY))
        rounded_box(ax, (2.1, 1.55), 5.8, 1.55, "P&L agregé", "un seul numeraire, change realise inclus" if valid else "variance et covariances FX absentes", color, 9)
        ax.text(5, 0.55, "non couvert : aucune position forward n'est modelisee" if valid else "un rendement est sans dimension, mais un P&L est une somme d'argent", ha="center", fontsize=7.5, color=GREY)
    fig.suptitle("Correction du numeraire : rendre la performance realisable par un investisseur marocain", fontsize=12, fontweight="bold", color=NAVY)
    fig.savefig(OUT / "numeraire.pdf")
    plt.close(fig)


def chaine_preuve() -> None:
    layers = [
        ("Contrats de donnees", "schemas Pandera, quality gate FX", "valeurs impossibles et trous longs"),
        ("Lignage DVC", "20 etapes et dependances explicites", "artefacts perimes"),
        ("Moteur causal", "decoupe a tau, poids appliques a tau+1", "look-ahead"),
        ("Provenance", "hashes, numeraire, dates, commit propre", "resultat dechire"),
        ("Validation statistique", "pairee + RC/SPA sur 240 candidats", "data snooping"),
        ("Faits canoniques", "README, cartes, rapport generes", "derive de formulation"),
        ("Controle rendu", "PDF relu apres compilation", "ancien chiffre encore visible"),
    ]
    fig, ax = plt.subplots(figsize=(10.3, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, len(layers) + 0.6)
    ax.axis("off")
    for i, (name, mechanism, catches) in enumerate(layers):
        y = len(layers) - i - 0.4
        rounded_box(ax, (0.35, y - 0.38), 3.15, 0.8, name, mechanism, BLUE if i < 4 else TEAL, 8.5)
        ax.annotate("", xy=(4.2, y), xytext=(3.55, y), arrowprops=dict(arrowstyle="->", color=GREY))
        rounded_box(ax, (4.25, y - 0.38), 5.25, 0.8, "Bloque / rend visible", catches, GREEN, 8.1)
    ax.set_title("Sept couches de controle : du fichier brut a la phrase lue par le jury", fontsize=12, color=NAVY, fontweight="bold")
    fig.savefig(OUT / "chaine_preuve.pdf")
    plt.close(fig)


def protocoles() -> None:
    nested = load("nested_walkforward_results.json")
    phase5 = load("phase5_results.json")["full_2021"]
    start = pd.Timestamp(nested["provenance"]["data_range"]["start"])
    end = pd.Timestamp(nested["provenance"]["data_range"]["end"])
    nested_start = pd.Timestamp(nested["design"]["oos_start"])
    single_start = pd.Timestamp(phase5["test_start"])

    fig, ax = plt.subplots(figsize=(10.3, 3.8))
    ax.barh(2, (nested_start - start).days, left=start, height=0.45, color=BLUE, alpha=0.65, label="selection / entrainement")
    ax.barh(2, (end - nested_start).days, left=nested_start, height=0.45, color=AMBER, alpha=0.82, label="OOS concatene")
    ax.barh(0.8, (single_start - start).days, left=start, height=0.45, color=BLUE, alpha=0.65)
    ax.barh(0.8, (end - single_start).days, left=single_start, height=0.45, color=RED, alpha=0.72, label="test gele")
    for fold in nested["folds"]:
        boundary = pd.Timestamp(fold["oos_start"])
        ax.axvline(boundary, ymin=0.55, ymax=0.78, color="white", lw=1.2)
    ax.set_yticks([0.8, 2])
    ax.set_yticklabels(["Decoupage unique\nPhase 5", f"Walk-forward imbrique\n{nested['design']['n_folds']} re-selections"])
    ax.set_xlim(start, end)
    ax.grid(alpha=0.2, axis="x")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    ax.set_title("Deux protocoles, deux fenetres OOS : le classement ne peut pas etre attribue au protocole seul", color=NAVY, fontweight="bold")
    fig.savefig(OUT / "protocoles.pdf")
    plt.close(fig)


def multiple_testing() -> None:
    reality = load("reality_check_results.json")
    rows = []
    for universe, payload in reality["universes"].items():
        for key, result in payload["tests"].items():
            benchmark, statistic = key.split("__")
            if benchmark == "regime_conditional":
                rows.append((universe, statistic, result["reality_check_p_value"], result["spa_p_value"], result["spa_candidates_retained"]))
    df = pd.DataFrame(rows, columns=["universe", "stat", "rc", "spa", "retained"])
    labels = [f"{u}\n{'rendement' if s == 'mean_return' else 'Sharpe'}" for u, s in zip(df.universe, df.stat)]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    width = 0.34
    ax.bar(x - width / 2, df.rc, width, label="White Reality Check", color=BLUE)
    ax.bar(x + width / 2, df.spa, width, label="Hansen SPA", color=TEAL)
    ax.axhline(0.05, color=RED, ls="--", lw=1.2, label="seuil descriptif 5 %")
    for i, retained in enumerate(df.retained):
        ax.text(i, max(df.rc.iloc[i], df.spa.iloc[i]) + 0.035, f"SPA retient {retained}/240", ha="center", fontsize=7, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("p-value unilaterale")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(alpha=0.22, axis="y")
    ax.set_title("Correction du data snooping face au comparateur primaire pre-specifie", color=NAVY, fontweight="bold")
    fig.savefig(OUT / "multiple_testing.pdf")
    plt.close(fig)


def explicabilite() -> None:
    explanations = load("model_explanations.json")
    full = explanations["universes"]["full_2021"]
    challengers = full["challengers"]
    primary = full["primary"]

    fig, axes = plt.subplots(1, 2, figsize=(11.1, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    features = [row["feature"] for row in challengers["rf_signal_tuned"]["global_importance_permutation"]]
    rf_map = {row["feature"]: row["mse_increase"] for row in challengers["rf_signal_tuned"]["global_importance_permutation"]}
    xgb_map = {row["feature"]: row["mse_increase"] for row in challengers["xgb_signal_tuned"]["global_importance_permutation"]}
    rf = np.array([rf_map[f] for f in features]); rf = rf / rf.sum()
    xgb = np.array([xgb_map[f] for f in features]); xgb = xgb / xgb.sum()
    y = np.arange(len(features)); h = 0.36
    axes[0].barh(y - h / 2, rf, h, label="Random Forest", color=BLUE)
    axes[0].barh(y + h / 2, xgb, h, label="XGBoost", color=TEAL)
    axes[0].set_yticks(y); axes[0].set_yticklabels(features, fontsize=8); axes[0].invert_yaxis()
    axes[0].set_xlabel("importance par permutation, normalisee")
    axes[0].legend(frameon=False)
    axes[0].set_title("Challengers : importance globale")

    weights = pd.Series(primary["weights"]).sort_values(ascending=True)
    colors = [RED if value <= 1e-10 else (AMBER if abs(value - 0.25) < 1e-8 else GREEN) for value in weights]
    axes[1].barh(weights.index, weights.values, color=colors)
    axes[1].axvline(0.25, color=RED, ls="--", lw=1, label="plafond 25 %")
    axes[1].set_xlabel("poids publie")
    decision = primary["decision"]
    axes[1].set_title(f"Systeme primaire : trace deterministe\n{decision['selected_regime']} -> {decision['selected_sub_optimizer']}")
    axes[1].legend(frameon=False)
    fig.suptitle("Deux explications adaptees a deux objets differents", fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "explicabilite.pdf")
    plt.close(fig)


def strategie_cout() -> None:
    results = load("phase4c_results.json")["full_2021"]
    rows = []
    for name, metrics in results["per_strategy"].items():
        rows.append((name, metrics["avg_turnover"], metrics["sharpe_net"], metrics["sharpe_gross"]))
    df = pd.DataFrame(rows, columns=["strategy", "turnover", "net", "gross"])
    fig, ax = plt.subplots(figsize=(10.3, 5.4))
    for row in df.itertuples():
        group = "ML" if "signal" in row.strategy else ("Regime/dynamique" if row.strategy in {"regime_conditional", "dcc_garch", "min_variance_ewma"} else "Classique")
        color = {"ML": RED, "Regime/dynamique": AMBER, "Classique": BLUE}[group]
        ax.scatter(row.turnover, row.net, s=68, color=color, edgecolor="white", linewidth=0.5)
        ax.annotate(row.strategy, (row.turnover, row.net), xytext=(4, 3), textcoords="offset points", fontsize=6.7)
    ax.axhline(results["phase4_hurdle"]["sharpe_net"], color=GREY, ls="--", lw=1, label="meilleur classique")
    ax.set_xlabel("rotation moyenne par reequilibrage")
    ax.set_ylabel("Sharpe net annualise")
    ax.grid(alpha=0.22)
    handles = [patches.Patch(color=c, label=g) for g, c in {"Classique": BLUE, "Regime/dynamique": AMBER, "ML": RED}.items()]
    ax.legend(handles=handles, frameon=False)
    ax.set_title("Performance nette et intensite de trading sur full_2021 (MAD)", color=NAVY, fontweight="bold")
    fig.savefig(OUT / "strategie_cout.pdf")
    plt.close(fig)


def industrialisation() -> None:
    rows = [
        ("Reproductibilite", "DVC + lock 284 dependances + R2", "livre", GREEN),
        ("Tests", "781 tests hors ligne + CI", "livre", GREEN),
        ("Explicabilite", "trace HMM + attributions arbres exactes", "livre", GREEN),
        ("Gouvernance", "3 cartes modele + politique challenger", "livre", GREEN),
        ("Service", "API REST read-only + dashboard", "prototype", BLUE),
        ("Conteneurisation", "Docker Compose, volumes read-only", "livre", GREEN),
        ("Monitoring", "PSI et baseline offline", "pret, non actif", AMBER),
        ("Execution marche", "ordres, capacite, impact", "hors perimetre", GREY),
        ("Validation independante", "revue par une seconde ligne", "a realiser", RED),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, len(rows) + 0.4)
    ax.axis("off")
    for i, (area, evidence, status, color) in enumerate(rows):
        y = len(rows) - i - 0.5
        ax.add_patch(patches.FancyBboxPatch((0.25, y - 0.34), 9.45, 0.68, boxstyle="round,pad=0.02", facecolor=color, alpha=0.08, edgecolor=color, linewidth=0.9))
        ax.text(0.48, y, area, va="center", fontsize=8.5, fontweight="bold", color=color)
        ax.text(3.05, y, evidence, va="center", fontsize=7.8, color=GREY)
        ax.text(9.35, y, status.upper(), va="center", ha="right", fontsize=7.6, fontweight="bold", color=color)
    ax.set_title("Lecture industrielle : ce qui est livre, pret, ou volontairement hors perimetre", fontsize=12, color=NAVY, fontweight="bold")
    fig.savefig(OUT / "industrialisation.pdf")
    plt.close(fig)


def main() -> None:
    builders = [
        architecture_globale,
        revisions_resultat,
        numeraire,
        chaine_preuve,
        protocoles,
        multiple_testing,
        explicabilite,
        strategie_cout,
        industrialisation,
    ]
    for builder in builders:
        builder()
        print(f"built {builder.__name__}")


if __name__ == "__main__":
    main()
