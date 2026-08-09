"""Figures for Chapter 5 (résultats) of the internship report.

Like the Chapter 2 set, every figure is computed from the committed Gold
artifacts — no number is typed into this file, so the report cannot drift from
the results it describes.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenance import require_current_artifact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
OUT = ROOT / "docs" / "rapport" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "docs" / "rapport" / "assets" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

BLUE, GREY, RED, GREEN = "#1F3864", "#595959", "#B03000", "#1B5E20"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

L = lambda n: json.loads((GOLD / n).read_text())

# The nested walk-forward artifact is loaded through the provenance gate, not
# with a bare read. It went stale invisibly once — produced 2026-07-28 from a
# MIXED-CURRENCY full_2021, it survived the base-currency correction and would
# have put a sign-flipped conclusion beside rebuilt numbers. The gate refuses it
# on CONTENT (numeraire absent or not MAD, or any recorded source hash no longer
# matching the tree), so the failure is detected instead of remembered.
NW = require_current_artifact(
    GOLD / "nested_walkforward_results.json",
    expect_universe="full_2021",
    expect_base_currency="MAD",
    root=ROOT,
)
SC, P5, CW, CAP = (L("dashboard_showcase.json"), L("phase5_results.json"),
                   L("crisis_windows.json"), L("etf_cap_verdict.json"))
PAIRED = L("paired_comparison_results.json")
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
    # NOT "aucun écart n'est significatif": these are MARGINAL intervals, and
    # concluding non-significance from their overlap is the same unlicensed
    # inference as concluding significance from non-overlap. The difference is
    # tested separately, by the paired bootstrap (tab:paires).
    ax.set_title("Intervalles marginaux : ils quantifient l'incertitude,\n"
                 "ils ne testent pas la différence entre stratégies",
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


# ── 6. Paired-comparison table (LaTeX, generated) ────────────────────────────
# This table was hand-typed once. Eight rows of literals in a report whose every
# other number is derived is exactly the drift the convention exists to prevent:
# the next Phase 5 run would move the artifact and leave the report asserting the
# old figures with no test able to notice. The prose immediately after the table
# quotes three of the same numbers, so those are emitted as macros rather than
# left as a second copy.
CAND_FR = {"rf_signal_tuned": "RF", "xgb_signal_tuned": "XGB"}
CAND_PROSE = {"rf_signal_tuned": "signal RF calibré",
              "xgb_signal_tuned": "signal XGB calibré"}
BENCH_FR = {"regime_conditional": "régimes", "equal_weight": "1/N"}
# Table cells are terse; the surrounding prose needs a readable noun phrase.
BENCH_PROSE = {"regime_conditional": "la stratégie à régimes",
               "equal_weight": "l'équipondéré"}
UNI_FR = {"etf_2017": r"\texttt{etf\_2017}", "full_2021": r"\texttt{full\_2021}"}
CARDINAUX = {1: "une", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq",
             6: "six", 7: "sept", 8: "huit", 9: "neuf", 10: "dix"}


def _signe(x: float, n: int = 3) -> str:
    """Signed French decimal for math mode: 0.0087 → ``+0{,}009``.

    ASCII hyphen, not U+2212: these strings land inside ``$...$`` where TeX
    already renders ``-`` as a proper minus, and a literal Unicode minus would
    depend on the math font covering it.
    """
    return ("+" if x >= 0 else "-") + f"{abs(x):.{n}f}".replace(".", "{,}")


def _mot(n: int) -> str:
    return CARDINAUX.get(n, str(n))


def tableau_paires():
    comps = PAIRED["comparisons"]
    # "Establishes outperformance" is the artifact's own bar, recomputed here so
    # the caption cannot claim a clean sweep that the data no longer supports.
    etablies = [c for c in comps
                if c["sharpe_diff_ci"][0] > 0 and c["p_value_no_outperformance"] < 0.05]
    verdict = (f"\\emph{{Aucune}} des {_mot(len(comps))} comparaisons n'établit de "
               "surperformance." if not etablies else
               f"{_mot(len(etablies)).capitalize()} comparaison(s) sur {_mot(len(comps))} "
               "établissent une surperformance.")

    lignes, n_boot = [], comps[0]["n_boot"]
    for uni in ("etf_2017", "full_2021"):
        bloc = [c for c in comps if c["universe"] == uni]
        lignes.append(f"    \\multirow{{{len(bloc)}}}{{*}}{{{UNI_FR[uni]}}}")
        for c in bloc:
            lo, hi = c["sharpe_diff_ci"]
            lignes.append(
                f"      & {CAND_FR[c['candidate']]} vs {BENCH_FR[c['benchmark']]}"
                f" & ${_signe(c['sharpe_diff'])}$"
                f" & $[{_signe(lo)};\\ {_signe(hi)}]$"
                f" & {fr(c['p_value_no_outperformance'], 3)} \\\\")
        if uni == "etf_2017":
            lignes.append("    \\addlinespace")

    # The prose highlights the case the paired test exists to catch: a candidate
    # ahead on point estimate whose paired interval still spans zero.
    cas = max((c for c in comps if c["sharpe_diff"] > 0
               and c["sharpe_diff_ci"][0] < 0 < c["sharpe_diff_ci"][1]),
              key=lambda c: P5[c["universe"]]["tuned"][c["candidate"]]["test_sharpe_net"])
    p5 = P5[cas["universe"]]
    plus_proche = min(comps, key=lambda c: c["p_value_no_outperformance"])

    macros = {
        "paireNb": _mot(len(comps)),
        "paireCasUnivers": UNI_FR[cas["universe"]],
        "paireCasCandidat": CAND_PROSE[cas["candidate"]],
        "paireCasSharpe": fr(p5["tuned"][cas["candidate"]]["test_sharpe_net"], 3),
        "paireCasRefSharpe": fr(p5["baselines"][cas["benchmark"]]["test_sharpe_net"], 3),
        # Delivered WITHOUT surrounding $: the prose places them inside its own
        # math, and nesting $...$ inside $...$ closes the group early.
        "paireCasDelta": _signe(cas["sharpe_diff"]),
        "paireCasIC": (f"[{_signe(cas['sharpe_diff_ci'][0])};\\ "
                       f"{_signe(cas['sharpe_diff_ci'][1])}]"),
        # Braced comma: these two land inside $...$, where a bare "," is a list
        # separator and TeX inserts a space after it ("p = 0, 066").
        "paireCasP": f"{cas['p_value_no_outperformance']:.3f}".replace(".", "{,}"),
        "paireMinP": f"{plus_proche['p_value_no_outperformance']:.3f}".replace(".", "{,}"),
        "paireMinPCas": (f"{CAND_FR[plus_proche['candidate']]} contre "
                         f"{BENCH_PROSE[plus_proche['benchmark']]} sur "
                         f"{UNI_FR[plus_proche['universe']]}"),
    }

    # French thousands separator, applied to the number alone: joining first and
    # substituting commas afterwards also rewrites the caption's punctuation.
    n_boot_fr = f"{n_boot:,}".replace(",", "\\,")
    texte = "\n".join([
        "% GÉNÉRÉ par scripts/build_figures_chap5.py — ne pas éditer à la main.",
        "% Source : data/gold/paired_comparison_results.json + phase5_results.json",
        *(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()),
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\caption[Comparaisons pairées]{Comparaisons pairées sur la fenêtre de test",
        f"  gelée, nettes de coûts (bootstrap par blocs, {n_boot_fr} rééchantillons).",
        f"  {verdict}}}",
        "  \\label{tab:paires}",
        "  \\begin{tabular}{@{}llrrr@{}}",
        "    \\toprule",
        "    Univers & Comparaison & $\\Delta$Sharpe & IC 90~\\% & $p$ \\\\",
        "    \\midrule",
        *lignes,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])
    (TABLES / "paires.tex").write_text(texte, encoding="utf-8")
    print(f"  paires.tex    ({len(comps)} comparaisons, {len(etablies)} établie(s))")


if __name__ == "__main__":
    print("figures chapitre 5 :")
    courbes(); intervalles(); crises(); detection(); plafond(); tableau_paires()
