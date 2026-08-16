"""Build the defense-oriented visualization notebooks from canonical artifacts.

The notebooks are presentation layers: they never refit a model, select a
configuration, or mutate an artifact.  Result-shaped values are loaded at
execution time from the released DVC snapshot.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def _notebook(cells: list[nbf.NotebookNode], title: str) -> nbf.NotebookNode:
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{title}:{index}".encode()).hexdigest()[:8]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Portfolio ML",
                "language": "python",
                "name": "portfolio_ml",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "portfolio_ml": {
                "artifact_driven": True,
                "purpose": "defense_visualization",
                "title": title,
            },
        }
    )
    return notebook


COMMON_SETUP = r'''
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

ROOT = Path.cwd()
if not (ROOT / "data" / "gold").exists():
    ROOT = ROOT.parent
GOLD = ROOT / "data" / "gold"
BRONZE = ROOT / "data" / "bronze"

def load_json(name):
    return json.loads((GOLD / name).read_text(encoding="utf-8"))

snapshot = load_json("snapshot_manifest.json")
assert snapshot["git_dirty"] is False, "Le snapshot source doit être propre."

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (12, 5.5),
    "figure.dpi": 120,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "equal_weight": "#94a3b8",
    "min_variance_lw": "#14b8a6",
    "max_sharpe": "#2563eb",
    "regime_conditional": "#f59e0b",
    "dcc_garch": "#0f766e",
    "rf_signal": "#8b5cf6",
    "xgb_signal": "#ef4444",
    "rf_signal_tuned": "#8b5cf6",
    "xgb_signal_tuned": "#ef4444",
}

display(Markdown(
    f"**Snapshot canonique :** `{snapshot['git_commit'][:12]}`  "
    f"— généré le `{snapshot['generated_at_utc']}` — arbre source propre."
))
'''


def build_currency_notebook() -> nbf.NotebookNode:
    return _notebook(
        [
            _md(
                """
                # Phase 6 — Correction du numéraire MAD/USD

                **Question de soutenance :** pourquoi la correction de devise a-t-elle changé
                la conclusion du projet ?

                Ce notebook explique la correction sans relancer le pipeline. Les performances
                viennent exclusivement des artefacts Gold du release. La courbe du taux de change
                lit le Bronze officiel BAM versionné par DVC, car cette série source n'est pas
                dupliquée dans Gold.
                """
            ),
            _md(
                """
                ## Sources et règles de lecture

                - `currency_manifest.json` : politique de numéraire et contrôle qualité.
                - `dashboard_showcase.json` : résultats finaux publiés.
                - `bam_fx_reference.parquet` : observations officielles BAM, lecture seule.
                - `snapshot_manifest.json` : provenance du release.

                Les deux univers ne sont jamais agrégés : `full_2021` est en MAD non couvert,
                tandis que `etf_2017` reste en USD. Leurs niveaux de Sharpe ne constituent donc
                pas une comparaison directe entre univers.
                """
            ),
            _code(COMMON_SETUP),
            _code(
                r'''
currency = load_json("currency_manifest.json")
showcase = load_json("dashboard_showcase.json")

policy_rows = []
for universe, meta in currency["universes"].items():
    policy = meta["policy"]
    policy_rows.append({
        "univers": universe,
        "numéraire": meta["base_currency"],
        "conversion requise": policy["requires_conversion"],
        "série FX requise": policy["requires_fx"],
        "statut de couverture": meta["hedge_status"],
        "actifs étrangers": len(policy["assets_foreign"]),
        "actifs domestiques": len(policy["assets_domestic"]),
    })
policy_table = pd.DataFrame(policy_rows).set_index("univers")
display(policy_table)

assert policy_table.loc["full_2021", "numéraire"] == "MAD"
assert policy_table.loc["etf_2017", "numéraire"] == "USD"
assert policy_table.loc["full_2021", "conversion requise"]
assert not policy_table.loc["etf_2017", "conversion requise"]
'''
            ),
            _md("## Série officielle et contrôle qualité"),
            _code(
                r'''
fx = pd.read_parquet(BRONZE / "bam_fx_reference.parquet").sort_index()
fx.index = pd.to_datetime(fx.index)
quality = currency["universes"]["full_2021"]["fx_quality"]
thresholds = quality["thresholds"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].plot(fx.index, fx["USDMAD"], color="#2563eb", lw=1.5)
axes[0].set_title("Taux de référence officiel BAM — MAD pour 1 USD")
axes[0].set_ylabel("MAD / USD")
axes[0].set_xlabel("")

fx_log_return = np.log(fx["USDMAD"] / fx["USDMAD"].shift(1)).dropna()
sns.histplot(fx_log_return, bins=45, ax=axes[1], color="#14b8a6")
axes[1].axvline(0, color="black", lw=1)
axes[1].set_title("Distribution des variations journalières officielles")
axes[1].set_xlabel("Log-rendement du taux")
plt.tight_layout()
plt.show()

quality_table = pd.DataFrame(
    [
        {
            "contrôle": "Volatilité annualisée",
            "mesure": quality["annualised_volatility"],
            "seuil": thresholds["max_annualised_volatility"],
            "règle": "≤",
        },
        {
            "contrôle": "Autocorrélation retard 1",
            "mesure": quality["lag1_autocorrelation"],
            "seuil": thresholds["min_lag1_autocorrelation"],
            "règle": "≥",
        },
        {
            "contrôle": "Part de valeurs aberrantes",
            "mesure": quality["outlier_share"],
            "seuil": thresholds["max_outlier_share"],
            "règle": "≤",
        },
        {
            "contrôle": "Densité d'observation",
            "mesure": quality["observation_density"],
            "seuil": thresholds["min_observation_density"],
            "règle": "≥",
        },
    ]
)
quality_table["verdict"] = [
    m <= s if rule == "≤" else m >= s
    for m, s, rule in quality_table[["mesure", "seuil", "règle"]].itertuples(index=False)
]
display(quality_table.style.format({"mesure": "{:.4f}", "seuil": "{:.4f}"}))
assert quality["passed"] and quality_table["verdict"].all()
'''
            ),
            _md("## Effet sur la comparaison publiée"),
            _code(
                r'''
rows = []
for universe, result in showcase["universes"].items():
    classical = result["best_classical"]
    regime = result["strategies"]["regime_conditional"]
    rows.extend([
        {"univers": universe, "approche": classical["name"], "Sharpe net": classical["sharpe_net"], "famille": "Classique"},
        {"univers": universe, "approche": "regime_conditional", "Sharpe net": regime["sharpe_net"], "famille": "Comparateur pré-spécifié"},
    ])
comparison = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(11, 5.2))
sns.barplot(data=comparison, x="univers", y="Sharpe net", hue="famille", palette=["#2563eb", "#f59e0b"], ax=ax)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3)
ax.set_title("Résultats finaux par univers — chaque univers garde son numéraire")
ax.set_xlabel("")
ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
ax.set_ylim(0, comparison["Sharpe net"].max() * 1.16)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()

summary_rows = []
for universe, result in showcase["universes"].items():
    summary_rows.append({
        "univers": universe,
        "numéraire": currency["universes"][universe]["base_currency"],
        "meilleur classique": result["best_classical"]["name"],
        "Sharpe classique": result["best_classical"]["sharpe_net"],
        "Sharpe régime": result["strategies"]["regime_conditional"]["sharpe_net"],
        "écart observé (%)": result["headline_lift_pct"],
    })
summary = pd.DataFrame(summary_rows).set_index("univers")
display(summary.style.format({"Sharpe classique": "{:.4f}", "Sharpe régime": "{:.4f}", "écart observé (%)": "{:+.2f}"}))
'''
            ),
            _code(
                r'''
full = summary.loc["full_2021"]
etf = summary.loc["etf_2017"]
display(Markdown(f"""
### Ce que cette correction établit

- `full_2021` est valorisé en **{full['numéraire']}**, sans couverture de change, à partir du taux officiel BAM.
- Son écart ponctuel est **{full['écart observé (%)']:+.2f} %** : {full['Sharpe régime']:.4f} pour `regime_conditional` contre {full['Sharpe classique']:.4f} pour `{full['meilleur classique']}`.
- `etf_2017` reste un univers homogène en **{etf['numéraire']}** et son écart ponctuel est **{etf['écart observé (%)']:+.2f} %**.

### Ce que cela n'établit pas

- Ces écarts descriptifs ne constituent pas un test pairé entre `regime_conditional` et le meilleur classique.
- Les Sharpe des deux univers ne doivent pas être comparés comme s'ils décrivaient le même portefeuille, la même période ou le même numéraire.
- Le prototype n'exécute aucune couverture FX et ne constitue pas un conseil d'investissement.
"""))
'''
            ),
        ],
        "Correction du numéraire MAD/USD",
    )


def build_explainability_notebook() -> nbf.NotebookNode:
    return _notebook(
        [
            _md(
                """
                # Phase 7 — Du régime détecté à l'allocation expliquée

                **Question de soutenance :** peut-on reconstruire une décision et expliquer ce
                que les challengers RF/XGBoost ont appris, sans attribuer causalement les marchés
                aux modèles ?

                Le système `regime_conditional` est expliqué par une trace de décision. Les
                challengers sont expliqués par attribution exacte de leur fonction prédictive.
                """
            ),
            _code(COMMON_SETUP),
            _code(
                r'''
explanations = load_json("model_explanations.json")
regime = pd.read_parquet(GOLD / "dashboard_regime.parquet")
regime["Date"] = pd.to_datetime(regime["Date"])

policy = explanations["explanation_policy"]
display(pd.Series(policy, name="politique d'explication").to_frame())
assert policy["shap_package_used"] is False
'''
            ),
            _md("## Chronologie causale des régimes"),
            _code(
                r'''
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
for ax, (universe, frame) in zip(axes, regime.groupby("universe", sort=True)):
    frame = frame.sort_values("Date")
    ax.plot(frame["Date"], frame["bull_prob"], color="#2563eb", lw=1.4, label="P(bull)")
    ax.fill_between(frame["Date"], 0, 1, where=frame["regime"].eq("bear"), color="#ef4444", alpha=0.10, label="Décision bear")
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{universe} — probabilité bull à chaque réallocation")
    ax.legend(loc="upper left", ncol=2)
plt.tight_layout()
plt.show()
'''
            ),
            _md("## Dernière décision reconstruite de bout en bout"),
            _code(
                r'''
decision_rows = []
weight_rows = []
for universe, payload in explanations["universes"].items():
    primary = payload["primary"]
    decision_rows.append({
        "univers": universe,
        "date": primary["decision_date"],
        "régime": primary["decision"]["selected_regime"],
        "sous-optimiseur": primary["decision"]["selected_sub_optimizer"],
        "P(bull)": primary["hmm"]["posterior"]["bull"],
        "P(bear)": primary["hmm"]["posterior"]["bear"],
        "HMM convergé": primary["hmm"]["converged"],
        "fallback": primary["fallback_used"],
        "positions au plafond": primary["constraints"]["n_binding_at_cap"],
    })
    for asset, weight in primary["weights"].items():
        weight_rows.append({"univers": universe, "actif": asset, "poids": weight})

display(pd.DataFrame(decision_rows).set_index("univers").style.format({"P(bull)": "{:.4f}", "P(bear)": "{:.4f}"}))

weights = pd.DataFrame(weight_rows)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, (universe, frame) in zip(axes, weights.groupby("univers", sort=True)):
    sns.barplot(data=frame.sort_values("poids", ascending=False), x="actif", y="poids", color="#14b8a6", ax=ax)
    cap = explanations["universes"][universe]["primary"]["constraints"]["max_weight"]
    ax.axhline(cap, color="#ef4444", ls="--", label=f"plafond {cap:.0%}")
    ax.set_title(f"{universe} — allocation expliquée")
    ax.legend()
plt.tight_layout()
plt.show()
'''
            ),
            _md("## Importance globale des challengers"),
            _code(
                r'''
importance_rows = []
for universe, payload in explanations["universes"].items():
    for challenger, model in payload["challengers"].items():
        for item in model["global_importance_permutation"]:
            importance_rows.append({
                "univers": universe,
                "challenger": challenger,
                "variable": item["feature"],
                "augmentation MSE": item["mse_increase"],
                "rang": item["rank"],
            })
importance = pd.DataFrame(importance_rows)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, ((universe, challenger), frame) in zip(axes.flat, importance.groupby(["univers", "challenger"], sort=True)):
    frame = frame.sort_values("augmentation MSE")
    ax.barh(frame["variable"], frame["augmentation MSE"], color=COLORS.get(challenger, "#64748b"))
    ax.set_title(f"{universe} — {challenger}")
    ax.set_xlabel("Hausse de MSE après permutation")
plt.tight_layout()
plt.show()
'''
            ),
            _md("## Du classement prédit aux poids réellement détenus"),
            _code(
                r'''
link_rows = []
concordance_rows = []
for universe, payload in explanations["universes"].items():
    for challenger, model in payload["challengers"].items():
        link = model["ranking_to_weights"]
        concordance_rows.append({
            "univers": universe,
            "challenger": challenger,
            "concordance de Spearman": link["rank_concordance_spearman"],
            "raison si indéfinie": link["concordance_undefined_reason"],
        })
        for asset, item in link["per_asset"].items():
            link_rows.append({
                "univers": universe,
                "challenger": challenger,
                "actif": asset,
                "rendement prédit": item["predicted_expected_return"],
                "rang prédit": item["predicted_rank"],
                "poids": item["weight"],
                "rang poids": item["weight_rank"],
                "au plafond": item["at_cap"],
            })

concordance = pd.DataFrame(concordance_rows)
display(concordance.set_index(["univers", "challenger"]).style.format({"concordance de Spearman": "{:+.3f}"}, na_rep="non définie"))

links = pd.DataFrame(link_rows)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, ((universe, challenger), frame) in zip(axes.flat, links.groupby(["univers", "challenger"], sort=True)):
    sns.scatterplot(data=frame, x="rendement prédit", y="poids", hue="au plafond", s=110, ax=ax, palette={True: "#ef4444", False: "#2563eb"})
    labels = (
        frame.assign(
            x_label=frame["rendement prédit"].round(10),
            y_label=frame["poids"].round(6),
        )
        .groupby(["x_label", "y_label"], as_index=False)["actif"]
        .agg(" / ".join)
    )
    for row in labels.itertuples(index=False):
        ax.annotate(row.actif, (row.x_label, row.y_label), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.margins(x=0.14, y=0.12)
    ax.set_title(f"{universe} — {challenger}")
plt.tight_layout()
plt.show()
'''
            ),
            _md("## Attribution locale exacte — le cas le plus contradictoire"),
            _code(
                r'''
finite = concordance.dropna(subset=["concordance de Spearman"])
case = finite.loc[finite["concordance de Spearman"].idxmin()]
model = explanations["universes"][case["univers"]]["challengers"][case["challenger"]]
local = model["local_contributions"]
contrib = pd.Series(local["contributions"]).sort_values()

fig, ax = plt.subplots(figsize=(11, 5))
colors = np.where(contrib >= 0, "#14b8a6", "#ef4444")
ax.barh(contrib.index, contrib.values, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title(f"{case['univers']} — {case['challenger']} — actif {local['asset']}")
ax.set_xlabel("Contribution à la prédiction")
plt.tight_layout()
plt.show()

display(Markdown(f"""
**Reconstruction :** biais `{local['bias']:.8f}` + contributions = prédiction `{local['prediction']:.8f}`.<br>
Erreur de reconstruction : `{local['reconstruction_error']:.3e}` — méthode : {local['method']}.

La concordance classement→poids de ce cas vaut **{case['concordance de Spearman']:+.3f}**.
Ce nombre ne prouve pas un défaut du modèle : le poids final combine rendement prédit,
covariance et contraintes. Il montre précisément pourquoi une attribution prédictive ne suffit
pas à expliquer une allocation.
"""))
'''
            ),
            _md(
                """
                ### Ce que ce notebook établit

                - La décision HMM → sous-optimiseur → poids est reconstructible à une date donnée.
                - Les attributions RF/XGBoost sont additives et décrivent la fonction ajustée.
                - La contrainte de 25 % peut dominer le passage du classement prédit aux poids.

                ### Ce qu'il n'établit pas

                - Une contribution de variable n'est pas une cause économique.
                - Une décision expliquée n'est ni une recommandation ni une validation du modèle.
                """
            ),
        ],
        "Explication des décisions et challengers",
    )


def build_validation_notebook() -> nbf.NotebookNode:
    return _notebook(
        [
            _md(
                """
                # Phase 8 — Validation temporelle et correction du data snooping

                **Question de soutenance :** comment prouver que la sélection ne voit pas le futur,
                puis éviter de confondre le meilleur résultat observé parmi 240 essais avec une
                surperformance établie ?
                """
            ),
            _code(COMMON_SETUP),
            _code(
                r'''
protocol = load_json("phase5_validation_protocol.json")
paired = load_json("paired_comparison_results.json")
reality = load_json("reality_check_results.json")
nested = load_json("nested_walkforward_results.json")
phase5 = load_json("phase5_results.json")
rc_series = pd.read_parquet(GOLD / "reality_check_series.parquet")
rc_series["Date"] = pd.to_datetime(rc_series["Date"])

assert protocol["protocol"] == "purged_walk_forward"
assert paired["multiple_testing"]["status"] == "established"
'''
            ),
            _md("## Sélection forward-only : train, purge/embargo, validation"),
            _code(
                r'''
import matplotlib.dates as mdates

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, universe in zip(axes, sorted(protocol["universes"])):
    rf_folds = protocol["universes"][universe]["random_forest"]["folds"]
    xgb_folds = protocol["universes"][universe]["xgboost"]["folds"]
    assert rf_folds == xgb_folds, "Les deux sélecteurs doivent partager les mêmes frontières."
    for y, fold in enumerate(rf_folds, start=1):
        segments = [
            ("Entraînement", fold["train_start"], fold["train_end"], "#2563eb"),
            ("Purge / embargo", fold["embargo_start"], fold["embargo_end"], "#94a3b8"),
            ("Validation", fold["val_start"], fold["val_end"], "#f59e0b"),
        ]
        for label, start, end, color in segments:
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            ax.barh(y, (end - start).days + 1, left=mdates.date2num(start), color=color, alpha=0.9, label=label if y == 1 else None)
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["val_start"])
    ax.set_yticks(range(1, len(rf_folds) + 1), [f"Fold {i}" for i in range(1, len(rf_folds) + 1)])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title(f"{universe} — fenêtre croissante, jamais de date future dans train")
    ax.legend(loc="upper left", ncol=3)
plt.tight_layout()
plt.show()
'''
            ),
            _md("## Test pairé : distribution de la différence, pas juxtaposition d'intervalles marginaux"),
            _code(
                r'''
comparisons = pd.DataFrame(paired["comparisons"])
comparisons["label"] = comparisons["universe"] + " · " + comparisons["candidate"] + " vs " + comparisons["benchmark"]
comparisons = comparisons.sort_values(["universe", "sharpe_diff"])

fig, ax = plt.subplots(figsize=(12, 7))
y = np.arange(len(comparisons))
low = comparisons["sharpe_diff"] - comparisons["sharpe_diff_ci"].str[0]
high = comparisons["sharpe_diff_ci"].str[1] - comparisons["sharpe_diff"]
ax.errorbar(comparisons["sharpe_diff"], y, xerr=np.vstack([low, high]), fmt="o", color="#2563eb", ecolor="#94a3b8", capsize=4)
ax.axvline(0, color="#ef4444", lw=1.2, ls="--")
ax.set_yticks(y, comparisons["label"])
ax.set_xlabel("Différence de Sharpe pairée (candidat − benchmark)")
ax.set_title("Bootstrap par blocs sur les mêmes dates")
plt.tight_layout()
plt.show()

display(comparisons[["universe", "candidate", "benchmark", "sharpe_diff", "sharpe_diff_ci", "p_value_no_outperformance", "prob_sharpe_diff_positive"]].style.format({
    "sharpe_diff": "{:+.3f}",
    "p_value_no_outperformance": "{:.3f}",
    "prob_sharpe_diff_positive": "{:.3f}",
}))
'''
            ),
            _md("## White Reality Check et Hansen SPA — benchmark primaire fixé avant les résultats"),
            _code(
                r'''
test_rows = []
for universe, payload in reality["universes"].items():
    for name, result in payload["tests"].items():
        benchmark, statistic = name.split("__")
        test_rows.extend([
            {"univers": universe, "benchmark": benchmark, "statistique": statistic, "test": "White RC", "p-value": result["reality_check_p_value"], "statut": result["status"], "candidats SPA retenus": result["spa_candidates_retained"], "N": result["n_candidates"]},
            {"univers": universe, "benchmark": benchmark, "statistique": statistic, "test": "Hansen SPA", "p-value": result["spa_p_value"], "statut": result["status"], "candidats SPA retenus": result["spa_candidates_retained"], "N": result["n_candidates"]},
        ])
tests = pd.DataFrame(test_rows)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for ax, (status, frame) in zip(axes, tests.groupby("statut", sort=False)):
    labels = frame["univers"] + " · " + frame["benchmark"] + " · " + frame["statistique"]
    sns.barplot(data=frame.assign(label=labels), y="label", x="p-value", hue="test", ax=ax, palette=["#2563eb", "#f59e0b"])
    ax.axvline(0.05, color="#ef4444", ls="--", lw=1, label="repère 5 %")
    ax.set_title("Primaire" if status == "primary" else "Exploratoire")
    ax.set_ylabel("")
    ax.legend(title="")
plt.suptitle("Les résultats exploratoires contre 1/N ne démontrent pas la valeur de la couche ML", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()

display(tests.drop_duplicates(["univers", "benchmark", "statistique"])[["univers", "benchmark", "statistique", "statut", "N", "candidats SPA retenus"]])
'''
            ),
            _md("## Inflation de sélection : 240 configurations observées sur le test"),
            _code(
                r'''
def sharpe_from_log_returns(values):
    values = pd.Series(values).dropna()
    return np.sqrt(252) * values.mean() / values.std(ddof=1)

candidate_sharpes = (
    rc_series.groupby(["universe", "candidate"], sort=False)["net_return"]
    .apply(sharpe_from_log_returns)
    .rename("Sharpe test ex post")
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (universe, frame) in zip(axes, candidate_sharpes.groupby("universe", sort=True)):
    sns.histplot(frame["Sharpe test ex post"], bins=30, color="#94a3b8", ax=ax)
    selected = max(
        item["test_sharpe_net"]
        for item in phase5[universe]["tuned"].values()
    )
    ex_post_best = frame["Sharpe test ex post"].max()
    ax.axvline(selected, color="#2563eb", lw=2, label=f"sélection honnête : {selected:.3f}")
    ax.axvline(ex_post_best, color="#ef4444", lw=2, ls="--", label=f"meilleur ex post : {ex_post_best:.3f}")
    ax.set_title(universe)
    ax.legend()
plt.suptitle("Chercher directement sur le test gonfle le meilleur résultat apparent", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()
'''
            ),
            _md("## Walk-forward imbriqué : davantage d'OOS, mais un classement dépendant du protocole et de sa fenêtre"),
            _code(
                r'''
nested_rows = []
for strategy, result in nested["strategies"].items():
    nested_rows.append({
        "stratégie": strategy,
        "Sharpe net": result["sharpe_net"],
        "borne basse": result["ci"][0],
        "borne haute": result["ci"][1],
        "largeur IC": result["ci_width"],
    })
nested_df = pd.DataFrame(nested_rows).sort_values("Sharpe net")

fig, ax = plt.subplots(figsize=(11, 6))
y = np.arange(len(nested_df))
ax.errorbar(
    nested_df["Sharpe net"], y,
    xerr=np.vstack([nested_df["Sharpe net"] - nested_df["borne basse"], nested_df["borne haute"] - nested_df["Sharpe net"]]),
    fmt="o", capsize=4, color="#8b5cf6", ecolor="#94a3b8",
)
ax.set_yticks(y, nested_df["stratégie"])
ax.set_xlabel("Sharpe net et intervalle marginal")
ax.set_title(f"Walk-forward imbriqué — {nested['design']['n_oos_rows']} lignes OOS")
plt.tight_layout()
plt.show()

display(Markdown(f"""
**DSR = {nested['best_dsr_vs_search']:.4f} sur {nested['n_search_trials']} configurations.**<br>
La valeur est reportée sans adjectif : aucun seuil d'interprétation n'a été pré-spécifié.
Le classement observé dépend du protocole **et de la fenêtre OOS associée** ; le chevauchement
des intervalles marginaux n'établit ni supériorité ni équivalence.
"""))
'''
            ),
            _md(
                """
                ### Ce que ce notebook établit

                - La sélection est strictement antérieure à chaque validation.
                - Les différences sont évaluées de façon pairée sur les mêmes dates.
                - La correction White/SPA couvre les 240 configurations accessibles.
                - Aucun candidat n'établit de surperformance contre le comparateur primaire pré-spécifié.

                ### Ce qu'il n'établit pas

                - L'absence de rejet n'est pas une preuve d'équivalence.
                - Les comparaisons contre l'équipondéré restent exploratoires.
                - Huit comparaisons externes ont été lues ; cette multiplicité externe reste signalée.
                """
            ),
        ],
        "Validation temporelle et correction du data snooping",
    )


def build_risk_notebook() -> nbf.NotebookNode:
    return _notebook(
        [
            _md(
                """
                # Phase 9 — Risque, coûts, contraintes et observabilité

                **Question de soutenance :** que voit un validateur lorsqu'il quitte le seul
                ratio de Sharpe pour examiner trajectoire, drawdown, rotation, coûts, concentration
                et comportement dégradé ?
                """
            ),
            _code(COMMON_SETUP),
            _code(
                r'''
equity_returns = pd.read_parquet(GOLD / "dashboard_equity.parquet")
equity_returns["Date"] = pd.to_datetime(equity_returns["Date"])
fit_summary = load_json("fit_report_summary.json")
fit_reports = pd.read_parquet(GOLD / "fit_reports.parquet")
crises = load_json("crisis_windows.json")
monitoring = load_json("monitoring_baseline.json")
currency = load_json("currency_manifest.json")

assert monitoring["status"] == "offline_on_demand_reference"
'''
            ),
            _md("## Trajectoires nettes et drawdowns"),
            _code(
                r'''
paths = equity_returns.sort_values(["universe", "strategy", "Date"]).copy()
paths["equity_net"] = paths.groupby(["universe", "strategy"])["net_return"].transform(lambda s: np.exp(s.cumsum()))
paths["drawdown"] = paths.groupby(["universe", "strategy"])["equity_net"].transform(lambda s: s / s.cummax() - 1)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for row, universe in enumerate(sorted(paths["universe"].unique())):
    frame = paths[paths["universe"].eq(universe)]
    for strategy, series in frame.groupby("strategy"):
        axes[row, 0].plot(series["Date"], series["equity_net"], label=strategy, color=COLORS.get(strategy))
        axes[row, 1].plot(series["Date"], series["drawdown"], label=strategy, color=COLORS.get(strategy))
    axes[row, 0].set_title(f"{universe} — croissance de 1 unité ({currency['universes'][universe]['base_currency']})")
    axes[row, 1].set_title(f"{universe} — drawdown")
    axes[row, 0].legend(ncol=2, fontsize=8)
    axes[row, 1].legend(ncol=2, fontsize=8)
    axes[row, 1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
plt.tight_layout()
plt.show()
'''
            ),
            _md("## Rotation, coût et contrainte de poids"),
            _code(
                r'''
profile_rows = []
scenario_rows = []
for item in fit_summary["results"]:
    profile = item["execution_profile"]
    one_x = item["cost_sensitivity"]["scenarios"]["1x"]
    profile_rows.append({
        "univers": item["universe"],
        "stratégie": item["strategy_requested"],
        "Sharpe net 1x": one_x["net_sharpe"],
        "rotation moyenne": profile["avg_turnover"],
        "coût moyen (pb)": profile["avg_cost_drag_bps"],
        "dates avec plafond actif": profile["cap_binding_date_rate"],
        "allocation max": profile["max_allocation_observed"],
    })
    for label, scenario in item["cost_sensitivity"]["scenarios"].items():
        scenario_rows.append({
            "univers": item["universe"],
            "stratégie": item["strategy_requested"],
            "multiplicateur": scenario["cost_multiplier"],
            "Sharpe net": scenario["net_sharpe"],
            "rendement annualisé": scenario["annualized_return"],
        })
profiles = pd.DataFrame(profile_rows)
scenarios = pd.DataFrame(scenario_rows)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, (universe, frame) in zip(axes, profiles.groupby("univers", sort=True)):
    sns.scatterplot(data=frame, x="rotation moyenne", y="Sharpe net 1x", size="dates avec plafond actif", hue="stratégie", sizes=(80, 300), ax=ax, palette=COLORS, legend=False)
    for row in frame.itertuples(index=False):
        ax.annotate(row.stratégie, (getattr(row, "_3"), getattr(row, "_2")), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.margins(x=0.12, y=0.16)
    ax.set_title(universe)
plt.suptitle("Le coût économique augmente avec la rotation ; le plafond structure l'allocation", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()

display(profiles.set_index(["univers", "stratégie"]).style.format({
    "Sharpe net 1x": "{:.3f}",
    "rotation moyenne": "{:.3f}",
    "coût moyen (pb)": "{:.2f}",
    "dates avec plafond actif": "{:.1%}",
    "allocation max": "{:.1%}",
}))
'''
            ),
            _md("## Sensibilité descriptive aux coûts"),
            _code(
                r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for ax, (universe, frame) in zip(axes, scenarios.groupby("univers", sort=True)):
    for strategy, series in frame.groupby("stratégie"):
        series = series.sort_values("multiplicateur")
        ax.plot(series["multiplicateur"], series["Sharpe net"], marker="o", label=strategy, color=COLORS.get(strategy))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(universe)
    ax.set_xlabel("Multiplicateur du coût linéaire")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Sharpe net du chemin de poids fixe")
plt.suptitle("Re-pricing d'un chemin fixe — ce n'est pas une ré-optimisation", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()

validity = fit_summary["results"][0]["cost_sensitivity"]["validity_conditions"]
display(Markdown("**Conditions de validité enregistrées dans l'artefact :**\n\n" + "\n".join(f"- {item}" for item in validity)))
'''
            ),
            _md("## Fenêtres de crise : drawdown par stratégie"),
            _code(
                r'''
crisis_rows = []
for universe, windows in crises["universes"].items():
    for crisis_id, strategies in windows.items():
        for strategy, metrics in strategies.items():
            if metrics is not None:
                crisis_rows.append({
                    "univers": universe,
                    "crise": crises["crises"][crisis_id]["label"],
                    "stratégie": strategy,
                    "drawdown max": metrics["max_drawdown"],
                    "rendement": metrics["cum_return"],
                    "jours de reprise": metrics["recovery_days"],
                })
crisis_df = pd.DataFrame(crisis_rows)

for universe, frame in crisis_df.groupby("univers", sort=True):
    matrix = frame.pivot(index="crise", columns="stratégie", values="drawdown max")
    plt.figure(figsize=(12, max(3.5, 0.65 * len(matrix))))
    sns.heatmap(matrix, annot=True, fmt=".1%", cmap="RdYlGn", center=0, cbar_kws={"label": "drawdown"})
    plt.title(f"{universe} — pertes maximales dans les fenêtres externes")
    plt.tight_layout()
    plt.show()
'''
            ),
            _md("## Intégrité des modèles : comportement demandé et comportement effectif"),
            _code(
                r'''
telemetry = (
    fit_reports.groupby(["universe", "strategy"], as_index=False)
    .agg(
        réallocations=("Date", "size"),
        fallbacks=("fit_status", lambda s: s.eq("fallback").sum()),
        avertissements=("fit_status", lambda s: s.eq("warning").sum()),
        modèles_effectifs=("model_effective", lambda s: ", ".join(sorted(set(s)))),
    )
)
telemetry["taux fallback"] = telemetry["fallbacks"] / telemetry["réallocations"]
display(telemetry.set_index(["universe", "strategy"]).style.format({"taux fallback": "{:.2%}"}))

total_fits = len(fit_reports)
total_fallbacks = fit_reports["fit_status"].eq("fallback").sum()
assert total_fallbacks == 0
display(Markdown(f"**Mesure versionnée : {total_fallbacks} fallback sur {total_fits} fits enregistrés.** Ce zéro décrit ce snapshot et ces stratégies ; il ne signifie pas que les chemins de repli sont impossibles."))
'''
            ),
            _md("## Monitoring : prêt pour contrôle à la demande, pas en production"),
            _code(
                r'''
monitor_rows = []
for universe, payload in monitoring["universes"].items():
    metrics = payload["metrics"]
    monitor_rows.append({
        "univers": universe,
        "fenêtre de référence": f"{payload['reference_window']['start']} → {payload['reference_window']['end']}",
        "fenêtre d'évaluation": f"{payload['evaluation_window_available']['start']} → {payload['evaluation_window_available']['end']}",
        "rotation moyenne": metrics["turnover"]["mean"],
        "part bear": metrics["regime_share"].get("bear", np.nan),
        "plafond actif": metrics["cap_binding"]["date_rate"],
    })
display(pd.DataFrame(monitor_rows).set_index("univers").style.format({"rotation moyenne": "{:.3f}", "part bear": "{:.1%}", "plafond actif": "{:.1%}"}))
display(Markdown(f"**Statut :** `{monitoring['status']}`. {monitoring['operational_note']}"))
'''
            ),
            _md(
                """
                ### Ce que ce notebook établit

                - Les performances sont accompagnées de trajectoires, drawdowns, rotation et coûts.
                - Le coût est une revalorisation linéaire d'un chemin de poids fixe, avec limites explicites.
                - Aucun label publié ne cache un fallback sur le snapshot mesuré.
                - Les références de monitoring existent pour un contrôle hors ligne à la demande.

                ### Ce qu'il n'établit pas

                - Il n'existe ni monitoring live, ni exécution d'ordres, ni mesure de capacité/impact de marché.
                - Un scénario de coûts n'est pas une preuve de robustesse économique.
                - Le prototype reste non consultatif et hors production.
                """
            ),
        ],
        "Risque, coûts, contraintes et observabilité",
    )


def build_global_2004_notebook() -> nbf.NotebookNode:
    """Visualize the frozen global_2004 readiness, Q1 and Q2 evidence."""
    return _notebook(
        [
            _md(
                """
                # Phase 10 — `global_2004` : le test équitable des modèles ML

                **Question de soutenance :** que reste-t-il de la promesse ML lorsque les deux
                défauts d'identification des univers précédents sont retirés ?

                Ce notebook est une couche de lecture des artefacts gelés. Il ne réentraîne
                aucun modèle, ne relance aucun backtest et ne télécharge aucune donnée. Il montre
                successivement pourquoi l'univers a été créé, ce que Q1 teste, ce que Q2 corrige,
                et ce que les résultats autorisent réellement à conclure.
                """
            ),
            _md(
                """
                ## Sources, périmètre et règles de lecture

                - Prix ajustés de dix ETF cotés aux États-Unis : source publique Yahoo Finance,
                  figée dans Bronze et versionnée par DVC.
                - Variables macro retardées : FRED (`VIX`, taux US 10 ans, dollar, spread Baa).
                - Numéraire : **USD**, univers mono-devise, sans conversion MAD/USD.
                - Fenêtre exploitable : novembre 2004 → août 2026.
                - Les écarts ponctuels sont descriptifs. Seuls les tests pré-spécifiés répondent
                  aux hypothèses correspondantes.
                - Il s'agit de deux évaluations distinctes mais statistiquement recouvrantes :
                  `etf_2017` et `global_2004` partagent cinq instruments et une grande partie
                  des dates.
                """
            ),
            _code(COMMON_SETUP),
            _code(
                """
                readiness = load_json("global_2004_readiness.json")
                q1 = load_json("global_2004_q1_results.json")
                q2 = load_json("global_2004_q2_results.json")
                q2_series = pd.read_parquet(GOLD / "global_2004_q2_series.parquet")

                required = {
                    "data/gold/global_2004_readiness.json",
                    "data/gold/global_2004_q1_results.json",
                    "data/gold/global_2004_q2_results.json",
                    "data/gold/global_2004_q2_series.parquet",
                }
                missing = required.difference(snapshot["files"])
                assert not missing, f"Artefacts absents du snapshot : {sorted(missing)}"
                for artifact in (readiness, q1, q2):
                    assert not artifact["provenance"]["git_revision"].endswith("-dirty")

                display(pd.DataFrame({
                    "Artefact": ["Readiness", "Q1", "Q2", "Séries Q2"],
                    "Rôle": [
                        "qualité et expressivité, aucune performance",
                        "régimes contre Markowitz",
                        "famille RF/XGBoost contre régimes",
                        "241 séries nettes sur les mêmes dates",
                    ],
                    "Source": [
                        "global_2004_readiness.json",
                        "global_2004_q1_results.json",
                        "global_2004_q2_results.json",
                        "global_2004_q2_series.parquet",
                    ],
                }))
                """
            ),
            _md(
                """
                ## 1. Pourquoi construire un troisième univers ?

                L'ancien univers ETF utilisait cinq actifs avec un plafond de 25 %. Le portefeuille
                minimum-variance n'y produisait qu'une allocation distincte sur 248 rééquilibrages.
                Le nouvel univers conserve exactement le même plafond, mais porte dix actifs
                couvrant actions, obligations, crédit, immobilier et or. La question n'est pas
                encore « gagne-t-il ? », mais d'abord « l'optimiseur peut-il exprimer une vue ? ».
                """
            ),
            _code(
                """
                window = readiness["window"]
                freedom = readiness["allocation_freedom"]
                reference = readiness["reference_comparison"]

                readiness_table = pd.DataFrame({
                    "Mesure": [
                        "Période", "Jours de marché", "Actifs", "Cellules forward-fill",
                        "Max. dominance lead/lag", "Allocations distinctes MinVar-LW",
                        "Actifs au plafond en moyenne", "Positions effectives moyennes",
                    ],
                    "Valeur": [
                        f"{window['start']} → {window['end']}",
                        readiness["provenance"]["data_range"]["n_rows"],
                        readiness["provenance"]["data_range"]["n_assets"],
                        readiness["coverage"]["forward_filled_cells"],
                        readiness["lag_dominance"]["max_lag_dominance"],
                        f"{freedom['min_variance_lw']['distinct_allocations']} / {freedom['n_rebalances']}",
                        freedom["min_variance_lw"]["mean_assets_at_cap"],
                        freedom["min_variance_lw"]["mean_effective_positions"],
                    ],
                })
                display(readiness_table)

                labels = ["etf_2017", "global_2004"]
                shares = [reference["etf_2017_distinct_share"], freedom["min_variance_lw"]["distinct_share"]]
                counts = [
                    f"{reference['etf_2017_min_variance_lw_distinct']}/{reference['etf_2017_n_rebalances']}",
                    f"{freedom['min_variance_lw']['distinct_allocations']}/{freedom['n_rebalances']}",
                ]
                fig, ax = plt.subplots(figsize=(9, 4.6))
                bars = ax.bar(labels, shares, color=["#94a3b8", "#14b8a6"], width=0.58)
                ax.set_ylim(0, 1.12)
                ax.set_ylabel("Part des rééquilibrages avec une allocation distincte")
                ax.set_title("Le défaut d'expressivité est retiré, sous le même plafond de 25 %")
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.035,
                            count, ha="center", fontweight="bold")
                plt.show()
                """
            ),
            _md(
                """
                **Lecture autorisée :** `global_2004` retire le défaut d'expressivité de
                `etf_2017`. Cela rend l'expérience plus informative ; cela ne constitue encore
                aucune preuve de performance.
                """
            ),
            _md(
                """
                ## 2. Q1 — La stratégie à régimes bat-elle Markowitz ?

                Q1 compare une seule stratégie pré-spécifiée, `regime_conditional`, à
                `max_sharpe`. Les deux sont évaluées nettes de coûts sur le même segment final
                gelé. Le seuil économique exige un gain d'au moins **0,05 point de Sharpe** ; le
                test statistique est un bootstrap pairé à blocs sur les mêmes jours.
                """
            ),
            _code(
                """
                candidate, comparator = q1["candidate"], q1["comparator"]
                q1_table = pd.DataFrame([
                    {
                        "Stratégie": candidate["strategy"],
                        "Sharpe net": candidate["net_sharpe"],
                        "Rendement annualisé géométrique": candidate["net_geometric_annual_return"],
                        "Max drawdown": candidate["max_drawdown"],
                        "Rotation moyenne": candidate["avg_turnover"],
                        "Coût cumulé": candidate["total_cost_fraction"],
                    },
                    {
                        "Stratégie": comparator["strategy"],
                        "Sharpe net": comparator["net_sharpe"],
                        "Rendement annualisé géométrique": comparator["net_geometric_annual_return"],
                        "Max drawdown": comparator["max_drawdown"],
                        "Rotation moyenne": comparator["avg_turnover"],
                        "Coût cumulé": comparator["total_cost_fraction"],
                    },
                ]).set_index("Stratégie")
                display(q1_table.style.format({
                    "Sharpe net": "{:.4f}",
                    "Rendement annualisé géométrique": "{:.2%}",
                    "Max drawdown": "{:.2%}",
                    "Rotation moyenne": "{:.3f}",
                    "Coût cumulé": "{:.2%}",
                }))

                fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
                colors = [COLORS["regime_conditional"], COLORS["max_sharpe"]]
                for ax, field, title, fmt in (
                    (axes[0], "Sharpe net", "Sharpe net observé", "{:.3f}"),
                    (axes[1], "Rendement annualisé géométrique", "Rendement annualisé net", "{:.1%}"),
                    (axes[2], "Rotation moyenne", "Rotation moyenne", "{:.3f}"),
                ):
                    values = q1_table[field].to_numpy()
                    bars = ax.bar(["Régimes", "Max Sharpe"], values, color=colors)
                    ax.set_title(title)
                    for bar, value in zip(bars, values):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), fmt.format(value),
                                ha="center", va="bottom", fontweight="bold")
                plt.tight_layout()
                plt.show()
                """
            ),
            _code(
                """
                diff = q1["observed_difference"]["net_sharpe_diff"]
                lo, hi = q1["paired_inference"]["sharpe_diff_ci"]
                p_value = q1["paired_inference"]["one_sided_null_centred_p_value"]

                fig, ax = plt.subplots(figsize=(10, 2.8))
                ax.axvline(0, color="#111827", linewidth=1.2)
                ax.errorbar(diff, 0, xerr=[[diff-lo], [hi-diff]], fmt="o", color="#ef4444",
                            ecolor="#475569", capsize=7, markersize=9)
                ax.set_yticks([])
                ax.set_xlabel("Δ Sharpe net : régime − max_sharpe")
                ax.set_title(f"Q1 : Δ observé {diff:+.4f} · IC 90 % [{lo:+.4f}, {hi:+.4f}] · p={p_value:.4f}")
                plt.show()
                """
            ),
            _md(
                """
                **Conclusion Q1 :** aucune surperformance Sharpe de la stratégie à régimes n'est
                établie. L'écart observé est défavorable et l'intervalle traverse zéro ; cela ne
                démontre pas non plus la supériorité de `max_sharpe`. La stratégie à régimes
                affiche une rotation environ 4,3 fois plus élevée et un drawdown maximal légèrement
                moins sévère : elle ne perd donc pas sur toutes les dimensions du risque.
                """
            ),
            _md(
                """
                ## 3. Q2 — Le meilleur RF/XGBoost parmi 240 essais est-il crédible ?

                Q2 teste une **famille**, pas une configuration isolée. Toutes les combinaisons
                atteignables d'hyperparamètres et de leviers sont conservées. White Reality Check
                et Hansen SPA évaluent le meilleur différentiel après avoir intégré le fait qu'il
                a été choisi parmi 240 possibilités.
                """
            ),
            _code(
                """
                family_rows = []
                for endpoint, label in (("primary_sharpe", "Primaire : Sharpe"),
                                        ("secondary_mean_return", "Secondaire : rendement moyen")):
                    block = q2["family_tests"][endpoint]
                    family_rows.append({
                        "Critère": label,
                        "White RC p": block["reality_check_p_value"],
                        "Hansen SPA p": block["spa_p_value"],
                        "Candidats > référence": f"{block['n_candidates_beating_benchmark']}/{block['n_candidates']}",
                        "Meilleur différentiel observé": block["best_differential"],
                    })
                display(pd.DataFrame(family_rows).set_index("Critère").style.format({
                    "White RC p": "{:.4f}", "Hansen SPA p": "{:.4f}",
                    "Meilleur différentiel observé": "{:+.6f}",
                }))
                """
            ),
            _code(
                """
                wide = q2_series.pivot(index="Date", columns="candidate", values="net_return")
                benchmark_returns = wide["regime_conditional"]

                def net_sharpe(series):
                    return float(series.mean() / series.std(ddof=1) * np.sqrt(252))

                benchmark_sharpe = net_sharpe(benchmark_returns)
                candidate_columns = [c for c in wide.columns if c != "regime_conditional"]
                sharpe_diffs = pd.Series(
                    {name: net_sharpe(wide[name]) - benchmark_sharpe for name in candidate_columns},
                    name="Δ Sharpe vs régime",
                )
                primary = q2["family_tests"]["primary_sharpe"]
                assert len(sharpe_diffs) == q2["candidate_ledger"]["executed_count"] == 240
                assert int((sharpe_diffs > 0).sum()) == primary["n_candidates_beating_benchmark"] == 17
                assert np.isclose(sharpe_diffs.max(), primary["best_differential"], atol=1e-6)

                fig, ax = plt.subplots(figsize=(11, 5.2))
                ax.hist(sharpe_diffs, bins=28, color="#64748b", edgecolor="white", alpha=0.9)
                ax.axvline(0, color="#111827", linewidth=1.4, label="Référence à régimes")
                ax.axvline(sharpe_diffs.max(), color="#ef4444", linewidth=2,
                           label=f"Maximum observé : {sharpe_diffs.max():+.3f}")
                ax.set_xlabel("Δ Sharpe net candidat − régime")
                ax.set_ylabel("Nombre de configurations")
                ax.set_title("Le maximum séduisant est le produit d'une recherche de 240 configurations")
                ax.legend()
                plt.show()

                display(Markdown(
                    f"**Verdict familial :** White RC p = `{primary['reality_check_p_value']:.4f}`, "
                    f"Hansen SPA p = `{primary['spa_p_value']:.4f}`. Le maximum observé "
                    f"`{sharpe_diffs.max():+.4f}` n'est pas une preuve de surperformance."
                ))
                """
            ),
            _md(
                """
                ## 4. Qu'aurait choisi un praticien sans regarder le test ?

                Les configurations « déployables » sont choisies uniquement sur entraînement et
                validation, par sélection walk-forward. Leur performance finale est descriptive :
                aucun test pairé individuel n'a été pré-spécifié pour démontrer qu'elles sont
                inférieures à la référence.
                """
            ),
            _code(
                """
                def candidate_label(spec):
                    params = "_".join(f"{k}={v}" for k, v in sorted(spec["ml_params"].items()))
                    return (f"{spec['model_type']}__{params}__shrink={spec['shrinkage_weight']}"
                            f"__pen={spec['turnover_penalty']}")

                def geometric_annual_return(series):
                    return float((1 + series).prod() ** (252 / len(series)) - 1)

                selected_return_series = {"regime_conditional": benchmark_returns}
                selected_rows = [{
                    "Stratégie": "regime_conditional",
                    "Sharpe net observé": benchmark_sharpe,
                    "Δ vs régime": 0.0,
                    "Rendement annualisé géométrique": geometric_annual_return(benchmark_returns),
                }]
                for display_name, spec in q2["deployable_challengers"].items():
                    label = candidate_label(spec)
                    selected_returns = wide[label]
                    selected_return_series[display_name] = selected_returns
                    selected_sharpe = net_sharpe(selected_returns)
                    selected_rows.append({
                        "Stratégie": display_name,
                        "Sharpe net observé": selected_sharpe,
                        "Δ vs régime": selected_sharpe - benchmark_sharpe,
                        "Rendement annualisé géométrique": geometric_annual_return(selected_returns),
                    })
                selected = pd.DataFrame(selected_rows).set_index("Stratégie")
                display(selected.style.format({
                    "Sharpe net observé": "{:.4f}", "Δ vs régime": "{:+.4f}",
                    "Rendement annualisé géométrique": "{:.2%}",
                }))

                fig, ax = plt.subplots(figsize=(9.5, 4.8))
                plot_values = selected["Sharpe net observé"]
                bars = ax.bar(["Régimes", "RF sélectionné", "XGBoost sélectionné"], plot_values,
                              color=[COLORS["regime_conditional"], COLORS["rf_signal_tuned"], COLORS["xgb_signal_tuned"]])
                ax.set_ylim(0, max(plot_values) * 1.18)
                ax.set_ylabel("Sharpe net observé")
                ax.set_title("Configurations sélectionnées sans accès au segment de test")
                for bar, value in zip(bars, plot_values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                            f"{value:.4f}", ha="center", fontweight="bold")
                plt.show()

                growth = pd.DataFrame({
                    name: (1 + returns).cumprod()
                    for name, returns in selected_return_series.items()
                }).rename(columns={
                    "regime_conditional": "Régimes",
                    "rf_signal_tuned": "RF sélectionné",
                    "xgb_signal_tuned": "XGBoost sélectionné",
                })
                ax = growth.plot(
                    figsize=(11, 5.2), linewidth=2,
                    color=[COLORS["regime_conditional"], COLORS["rf_signal_tuned"], COLORS["xgb_signal_tuned"]],
                )
                ax.set_ylabel("Valeur cumulée d'un dollar (nette de coûts)")
                ax.set_xlabel("Date du segment de test gelé")
                ax.set_title("Trajectoires des configurations sélectionnées sans accès au test")
                ax.legend(title=None)
                plt.tight_layout()
                plt.show()
                """
            ),
            _code(
                """
                telemetry = q2["fallback_telemetry"]
                selected_fallbacks = {}
                for name, spec in q2["deployable_challengers"].items():
                    selected_fallbacks[name] = telemetry["candidates"][candidate_label(spec)]["fallback_count"]

                telemetry_table = pd.DataFrame({
                    "Périmètre": ["Référence à régimes", "Famille des 240 candidats", "RF sélectionné", "XGBoost sélectionné"],
                    "Fallbacks": [telemetry["benchmark"]["fallback_count"], telemetry["total_candidate_fallbacks"],
                                  selected_fallbacks["rf_signal_tuned"], selected_fallbacks["xgb_signal_tuned"]],
                    "Fits": [telemetry["benchmark"]["n_fits"], 240 * 249, 249, 249],
                })
                telemetry_table["Taux"] = telemetry_table["Fallbacks"] / telemetry_table["Fits"]
                display(telemetry_table.style.format({"Taux": "{:.4%}"}))
                """
            ),
            _md(
                """
                ## 5. Conclusion générale

                ### Succès d'ingénierie
                Une chaîne Bronze/Silver/Gold versionnée, des contrats de données, une validation
                sans fuite, une télémétrie des fallbacks et 241 séries comparables ont été produits.

                ### Succès méthodologique
                Le nouvel univers retire les défauts d'expressivité et de décalage de marché qui
                limitaient les deux univers précédents. La pré-inscription et la correction RC/SPA
                empêchent de transformer le meilleur résultat brut d'une recherche en conclusion.

                ### Résultat économique
                Aucun avantage Sharpe du modèle à régimes ou de la famille RF/XGBoost n'est établi
                sur `global_2004`. Le meilleur différentiel brut parmi 240 candidats ne résiste pas
                à la correction, et les deux challengers honnêtement sélectionnés affichent un
                Sharpe net observé inférieur à la référence.

                ### Limites
                Q2 change simultanément la coupe d'actifs et la politique de variables macro ; son
                résultat ne peut donc pas être attribué au seul élargissement de l'univers. Le choix
                externe de construire `global_2004` après diagnostic des univers précédents n'est
                pas corrigé par RC/SPA. Enfin, il s'agit d'un prototype de recherche en USD, sans
                exécution d'ordres ni recommandation d'investissement.
                """
            ),
        ],
        "global_2004 — expressivité, Q1, Q2 et tests multiples",
    )


def main() -> None:
    """Write deterministic notebook sources; execution is a separate verification step."""
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "phase6_currency_correction.ipynb": build_currency_notebook(),
        "phase7_model_decision_explainability.ipynb": build_explainability_notebook(),
        "phase8_validation_and_statistical_evidence.ipynb": build_validation_notebook(),
        "phase9_risk_cost_and_robustness.ipynb": build_risk_notebook(),
        "phase10_global_2004_evidence.ipynb": build_global_2004_notebook(),
    }
    for name, notebook in notebooks.items():
        nbf.write(notebook, NOTEBOOK_DIR / name)
        print(f"Wrote {NOTEBOOK_DIR / name}")


if __name__ == "__main__":
    main()
