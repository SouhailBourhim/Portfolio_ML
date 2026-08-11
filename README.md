# Système ML d'Optimisation de Portefeuille

**PFA (Projet de Fin d'Année)** — INPT (Institut National des Postes et Télécommunications)
**Client :** EURAFRIC Information, Bouskoura, Maroc
**Équipe :** Souhail Bourhim, Zakarya EL WALI, Yasmine BOUAJINE
**Encadrant :** Abdelmouttalib

## Objectif et périmètre

Construire un **prototype de recherche reproductible** d'optimisation de portefeuille,
combinant des actions de la Bourse de Casablanca (BVC) et des ETF internationaux. Le
projet évalue, dans un cadre de backtesting temporel, si des méthodes statistiques et
de Machine Learning atténuent quatre faiblesses structurelles de la théorie moderne du
portefeuille (MPT) de Markowitz :

| # | Problème | Approche évaluée |
|---|----------|-------------|
| P1 | Estimation bruitée de la covariance | Régularisation et covariance dynamique (Ledoit-Wolf → EWMA → DCC-GARCH) |
| P2 | Non-stationnarité des rendements | Hidden Markov Models (régimes de marché) |
| P3 | Rupture de la diversification en crise | Contraintes de portefeuille + covariance dynamique + HMM |
| P4 | Surapprentissage du backtesting | Sélection strictement antérieure (purge + embargo) + backtesting walk-forward + comparaisons pairées |

Le livrable est une aide à l'analyse et à la recherche : il **ne fournit ni conseil
d'investissement, ni recommandation client, ni exécution automatique d'ordres**. Il ne
doit pas être présenté comme un système de production ou comme une preuve que le ML
surperforme les méthodes classiques.

> ### ⚠️ Deux réserves qui accompagnent CHAQUE chiffre de portefeuille
>
> **1. Exposition de change USD/MAD non couverte.** Les actifs BVC sont libellés en
> dirhams, les ETF en dollars. Les rendements étant sans unité, l'arithmétique de
> portefeuille reste valide, mais **tout résultat publié incorpore cette exposition
> de change** — un risque économique matériel, pas une réserve de forme. Aucune
> couverture n'est modélisée : le sujet est hors périmètre et le reste.
>
> **2. Étiquette de modèle et modèle effectif.** Plusieurs estimateurs se dégradent
> au lieu d'échouer (DCC-GARCH → Ledoit-Wolf, signaux ML → moyenne empirique,
> régimes → branche défensive). Le décompte publié est l'énoncé « Intégrité des
> modèles » du bloc **Faits publiés** ci-dessous : il est généré depuis
> `data/gold/fit_report_summary.json`, jamais saisi, et confronté à un second
> artefact (`dashboard_regime.parquet`) par `TestFallbackCountsAgree`. La portée
> est exactement cet ensemble ; ce n'est pas l'affirmation qu'aucun modèle ne replie
> jamais, et l'instrumentation existe pour rendre cette question vérifiable sur tout
> instantané futur. Mesuré, pas supposé — voir
> [`docs/MODEL_INTEGRITY.md`](docs/MODEL_INTEGRITY.md) et l'artefact
> `data/gold/fit_report_summary.json`.

## Univers d'actifs

- **Actions BVC :** IAM.CS (Maroc Telecom), ATW.CS (Attijariwafa Bank), CIH.CS (CIH Bank), BCP.CS (Banque Centrale Populaire)
- **ETF internationaux :** SPY, QQQ, EEM, GLD, TLT
- **Indicateurs macro :** FRED (VIX, US10Y, DXY, CREDIT_SPREAD) + Bank Al-Maghrib (EUR/MAD, USD/MAD, taux directeur)

Deux univers de backtest :

| Univers | Composition | Fenêtre | Crises couvertes |
|---|---|---|---|
| `etf_2017` | 5 ETF | **2004-11 → aujourd'hui** (~5 650 jours) | 2008, COVID 2020, choc de taux 2022 |
| `full_2021` | 9 actifs (BVC + ETF) | 2021-07 → aujourd'hui (~1 320 jours) | choc de taux 2022 |

La fenêtre ETF a été étendue de 2017 à 2004 le 2026-07-25 : l'ancien départ était une décision de
projet, pas une limite de données. Les ~12 années supplémentaires réduisent la largeur des
intervalles de confiance bootstrap de **38 %** et intègrent la crise de 2008
([note](docs/ETF_DEEP_HISTORY_EXPERIMENT.md)). L'univers à 9 actifs reste tronqué à 2021-07 par la
disponibilité des données BVC.

**Rendements totaux, pas seulement les prix.** Les ETF arrivent ajustés des dividendes
(`yfinance auto_adjust`) et, depuis le 2026-07-25, les actions BVC le sont également via
[`src/dividends.py`](src/dividends.py) (montants et dates de détachement récupérés auprès de la
Bourse de Casablanca). Avant cette correction, les actifs marocains étaient sous-estimés de
3,0–4,3 %/an, ce qui **gonflait** l'avantage mesuré des optimiseurs — détail et impact chiffré dans
[`docs/DIVIDEND_BIAS.md`](docs/DIVIDEND_BIAS.md).

## Architecture des données

Le pipeline suit une architecture en médaillon à trois couches :

```
Bronze (données brutes, immuables)
   → Silver (nettoyage, alignement calendaire, log-rendements)
      → Gold (features prêtes pour le ML, validées par Pandera)
```

Une visite guidée complète du code de la Phase 1 (en anglais) est disponible dans
[`docs/PHASE1_WALKTHROUGH.md`](docs/PHASE1_WALKTHROUGH.md).

## Documentation (livrables encadrant)

Un livrable détaillé (français, Word) accompagne chaque phase — architecture, choix de
conception justifiés, résultats réels, tests, limitations et traçabilité P1–P4 :

- [`docs/Livrable_Phase1_Infrastructure_Donnees.docx`](docs/Livrable_Phase1_Infrastructure_Donnees.docx)
- [`docs/Livrable_Phase2_Backtesting_Markowitz.docx`](docs/Livrable_Phase2_Backtesting_Markowitz.docx)
- [`docs/Livrable_Phase3_Feature_Engineering_ML.docx`](docs/Livrable_Phase3_Feature_Engineering_ML.docx)
- [`docs/Livrable_Phase4_Regime_Covariance.docx`](docs/Livrable_Phase4_Regime_Covariance.docx)
- [`docs/Livrable_Phase4B_Adaptive_ML_Signals.docx`](docs/Livrable_Phase4B_Adaptive_ML_Signals.docx)
- [`docs/Livrable_Phase4C_Optimisation_Sensible_aux_Couts.docx`](docs/Livrable_Phase4C_Optimisation_Sensible_aux_Couts.docx)
- [`docs/Livrable_Phase5_Evaluation_OOS.docx`](docs/Livrable_Phase5_Evaluation_OOS.docx)
- [`docs/Livrable_Phase6-7_Suite_Portfolio_ML.docx`](docs/Livrable_Phase6-7_Suite_Portfolio_ML.docx)
- [`docs/Livrable_Phase8_Etudes_Robustesse.docx`](docs/Livrable_Phase8_Etudes_Robustesse.docx) — cinq études post-Phase 5 (données profondes, fondamentaux, walk-forward imbriqué, plafond conditionné au régime, comportement en crise), rapportées comme analyses de robustesse exploratoires

Notebooks de validation, exécutés et lisibles avec leurs résultats :
[`phase1_eda.ipynb`](notebooks/phase1_eda.ipynb) ·
[`phase2_backtest.ipynb`](notebooks/phase2_backtest.ipynb) ·
[`phase3_features.ipynb`](notebooks/phase3_features.ipynb) ·
[`phase4_regime_covariance.ipynb`](notebooks/phase4_regime_covariance.ipynb) ·
[`phase4b_adaptive_ml_signals.ipynb`](notebooks/phase4b_adaptive_ml_signals.ipynb) ·
[`phase4c_cost_aware.ipynb`](notebooks/phase4c_cost_aware.ipynb) ·
[`phase5_oos_evaluation.ipynb`](notebooks/phase5_oos_evaluation.ipynb).

### ⭐ Note de recherche — comportement en crise (analyse exploratoire, 2026-07)

**P3** (rupture de la diversification en crise) était le problème le moins directement étayé :
toutes les phases rapportaient un Sharpe et un drawdown sur période complète, aucune ne mesurait
le comportement **pendant** les crises. Cinq fenêtres, délimitées par les dates de sommet-à-creux
**publiées du S&P 500** et fixées avant tout examen des résultats.

**Résultat A — sur ces cinq fenêtres, l'optimisation sous contrainte a mieux protégé
le capital que le 1/N.**

| Crise | optimiseurs | équipondéré |
|---|---:|---:|
| Crise financière 2008 | -21,2 % · DD -26,9 % | **-30,2 % · DD -36,2 %** |
| Dette européenne 2011 | **+1,6 %** (positif) | -5,4 % |
| COVID 2020 | DD -16,2 % · récup. 37 j | DD -19,1 % · récup. 71 j |

Le **délai de récupération** est l'écart le plus régulier : environ **deux fois moins de temps sous
l'eau**. ⚠️ Attribution honnête : ce gain revient à la **contrainte et au modèle de covariance**
(P1/P3), pas à la couche de régime — sur 3 des 5 fenêtres les trois optimiseurs sont identiques.

**Résultat B — le HMM est plus souvent dans son état « baissier » pendant ces cinq
fenêtres.**

Régime baissier **91,7 % pendant** ces fenêtres contre **29,2 % hors fenêtre** — rapport
**3,13×**, avec **5/5** fenêtres au-dessus du taux de base. C'est une association
descriptive, pas une preuve confirmatoire : il n'y a que cinq événements, les fenêtres
sont étudiées a posteriori et « baissier » est défini comme l'état dont le rendement moyen
est le plus faible. Le test des signes calculé dans l'artefact est donc conservé comme
diagnostic exploratoire, et non comme un résultat généralisable.

À garder dans deux phrases distinctes : le détecteur produit une lecture causale de l'état
du marché ; l'effet économique d'agir sur cette lecture n'est pas établi. Détails :
[`docs/CRISIS_WINDOWS_EXPERIMENT.md`](docs/CRISIS_WINDOWS_EXPERIMENT.md) ·
`experiments/crisis_windows.py`.

### Note de recherche — expansion des données (2026-07)

Le signal ML était-il *sous-alimenté* en données ? Test sur un univers marocain **profond de 12
actions sur ~20 ans (2005–2024, 56 000 lignes de panel ≈ 5× l'univers actuel)**, assemblé à partir
d'historiques investing.com. Résultat : davantage de données augmente le coefficient
d'information (×2–4, ~0,07), sans établir d'avantage robuste de portefeuille. La limite
observée concerne donc la qualité et la nature des données autant que leur quantité. Détails : [`docs/DEEP_MOROCCO_EXPERIMENT.md`](docs/DEEP_MOROCCO_EXPERIMENT.md) ·
[`notebooks/deep_morocco_data_expansion.ipynb`](notebooks/deep_morocco_data_expansion.ipynb) ·
`experiments/deep_morocco_starvation.py`.

### Note de recherche — plafond conditionné au régime : hypothèse réfutée (2026-07)

Suite directe de la note sur le plafond : si la contrainte régularise mieux que tout modèle de
covariance, il faut conditionner **le plafond** au régime plutôt que la covariance — le resserrer en
régime baissier (davantage de *shrinkage* quand les corrélations explosent : P1 + P3), le relâcher en
régime haussier. Aucune modification du moteur principal n'a été nécessaire :
`RegimeConditionalStrategy` accepte déjà des sous-stratégies, donc c'est
`MaxSharpe(max_weight=plafond_haussier)` + `MinVarianceLW(max_weight=plafond_baissier)`.

Le protocole inclut le **contrôle capable de tuer l'hypothèse** : des plafonds fixes (isolant « c'est
le niveau qui aide ») et une variante **INVERSÉE** (large en baissier, serré en haussier — la
mauvaise direction). Résultats pré-enregistrés avant exécution.

> ⚠️ **SUPERSEDED — chiffres `full_2021` pré-correction.** Les valeurs `full_2021`
> ci-dessous ont été calculées sur l'univers MIXTE MAD/USD, avant la conversion en
> numéraire unique. Elles sont conservées comme trace historique et ne doivent pas
> être citées comme résultats courants. Les valeurs `etf_2017` sont inchangées :
> cet univers est mono-devise USD.

| | référence (25/25) | meilleur candidat | contrôle INVERSÉ | meilleur fixe |
|---|---:|---:|---:|---:|
| `full_2021` *(SUPERSEDED)* | **1,2363** | `both_40_15` 1,2663 (**+0,030**) | 1,1430 (−0,093) | 1,2333 (−0,003) |
| `etf_2017` | **0,9371** | `aggressive_40_25` 0,8525 (**−0,085**) | 0,8899 (−0,047) | 0,9525 (+0,015) |

**Verdict (C) sur les deux univers : aucune variante ne dépasse la référence de façon matérielle.**
Le mécanisme supposé échoue — resserrer en baissier n'apporte rien (+0,0016 / −0,117) ; et surtout
**le contrôle change de signe selon l'univers** (l'INVERSÉ *bat* le meilleur candidat « correct » sur
`etf_2017`), ce qui exclut un véritable effet de régime. Un résultat négatif propre : une hypothèse
bien motivée, pré-enregistrée, contrôlée — et réfutée.

Confirmation incidente de la dégénérescence du plafond : sur `etf_2017` au plafond 0,20,
`min_variance_lw` et `max_sharpe` renvoient **exactement 0,7694**, le Sharpe de l'équipondéré, car
`5 × 0,20 = 1,0` impose le 1/N quel que soit l'objectif.

Détails : [`docs/REGIME_CONDITIONAL_CAP_EXPERIMENT.md`](docs/REGIME_CONDITIONAL_CAP_EXPERIMENT.md).

### Note de recherche — walk-forward imbriqué : acheter de la puissance statistique (2026-07)

La limite explicitement annoncée par la Phase 5 : sur `full_2021`, la fenêtre de test gelée ne fait
que ~1,75 an (455 lignes) et **tous** les intervalles de confiance dépassent 2,2 de Sharpe. Ce n'est
pas un problème de modèle — c'est un problème de **taille d'échantillon**, qu'aucun modèle meilleur
ne corrige. Le walk-forward imbriqué (`experiments/nested_walkforward.py`) ré-sélectionne la
configuration F7 à **6 frontières successives** et concatène chaque segment hors échantillon :
**793 lignes hors échantillon contre 455**, la sélection ne voyant jamais sa propre fenêtre
d'évaluation.

| Stratégie | Découpage unique | largeur | **Imbriqué** | largeur | resserrement |
|---|---:|---:|---:|---:|---:|
| `regime_conditional` | 1,213 | 2,167 | **1,672** [0,89 ; 2,51] | **1,612** | 25,6 % |
| `xgb_signal_tuned` | **1,308** | 2,260 | 1,436 [0,70 ; 2,26] | 1,558 | 31,1 % |
| `equal_weight` | 1,003 | 2,333 | 1,284 [0,48 ; 2,18] | 1,705 | 26,9 % |
| `rf_signal_tuned` | 1,040 | 2,125 | 1,256 [0,57 ; 2,04] | 1,472 | 30,7 % |

- **Largeur moyenne des intervalles : 2,221 → 1,587, soit −28,6 %**, mieux que les 24,3 % prédits par
  la seule racine carrée de l'échantillon. Sharpe déflaté **0,835 sur 198 configurations** (contre
  0,67 sur 36) : recherche plus large *et* gagnant plus robuste.
- **Toutes les bornes inférieures sont désormais positives** (0,478 à 0,893) ; au découpage unique,
  celle de `equal_weight` descendait à **−0,090**. Sur cet univers, chaque stratégie examinée est
  maintenant crédiblement positive hors échantillon.
- ⚠️ **La hausse des niveaux est un effet de période, pas une amélioration** : le Sharpe ponctuel de
  *toutes* les stratégies monte, la fenêtre imbriquée commençant en 2023-07 contre 2024-10. Seules
  les comparaisons *internes à un même passage* ont un sens.
- Le classement **repasse à `regime_conditional`** — soit un **troisième ordre différent** en trois
  évaluations de `full_2021`. La lecture honnête n'est pas « le régime gagne finalement » mais que
  **l'ordre ponctuel est instable au protocole d'évaluation**, ce que la Phase 5 concluait déjà. Les
  intervalles se chevauchent presque intégralement : aucune significativité, dans aucun sens.

Détails : [`docs/NESTED_WALKFORWARD_EXPERIMENT.md`](docs/NESTED_WALKFORWARD_EXPERIMENT.md).

### Note de recherche — la contrainte de plafond fait plus que les modèles (2026-07)

Sur l'univers `etf_2017`, le **plafond de 25 % par actif** — choisi en Phase 2 comme une
*contrainte réaliste de gestion*, pas comme un outil de modélisation — s'avère être le régulariseur
le plus puissant du projet. En balayant le seul plafond sur 248 rééquilibrages et 20,7 ans, tout le
reste étant fixé :

| `max_weight` | Meilleur Sharpe net classique | Allocations distinctes (`min_variance_lw`) |
|---|---:|---:|
| **0,25 (le nôtre)** | **0,953** | **1** sur 248 |
| 0,30 | 0,939 | 171 |
| 0,35 | 0,933 | 248 |
| 0,40 | 0,912 | 248 |
| 1,00 (sans plafond) | 0,865 | 248 |

Deux conséquences :
- **Un écart de 10,1 % de Sharpe dû à la seule contrainte**, décroissant de façon monotone à mesure
  que le plafond se relâche — **plus que l'écart entre deux modèles quelconques** sur cet univers
  (Ledoit-Wolf, EWMA, DCC-GARCH et la commutation de régime HMM réunis le déplacent moins). C'est le
  résultat de **Jagannathan & Ma (2003)** reproduit sur nos données : une contrainte de poids
  active équivaut mathématiquement à un rétrécissement (*shrinkage*) de la matrice de covariance.
  La contrainte réalisait donc le contrôle d'erreur d'estimation (P1) que le ML devait apporter.
- **À 0,25, le plafond détermine presque toute l'allocation** : avec 5 actifs, `5 × 0,25 = 1,25`,
  donc tout portefeuille admissible place au moins 4 actifs *au plafond*. L'univers à 9 actifs n'est
  pas concerné (`9 × 0,25 = 2,25`).

À présenter comme un résultat, pas comme une limite : *le contrôle du risque le plus efficace sur
cet univers s'est révélé être la limite de position qu'un mandat réel imposerait de toute façon.*
Détails : [`docs/ETF_DEEP_HISTORY_EXPERIMENT.md`](docs/ETF_DEEP_HISTORY_EXPERIMENT.md) ·
`experiments/etf_cap_verdict.py`.

### Note de recherche — fondamentaux (2026-07)

Suite directe de la note précédente : puisque le plafond serait la **qualité** des données, on
ajoute des **fondamentaux point-in-time** (P/E, P/B, P/S, D/E scrapés depuis stockanalysis.com,
décalés de 90 jours ouvrés pour respecter le délai de publication AMMC) aux 4 actions BVC de
l'univers `full_2021`. Résultat honnête, **troisième** confirmation du plafond « précision de
prédiction ≠ performance du portefeuille » :
- **Le modèle apprend davantage** : IC ×2 en validation croisée purgée (0,028 → 0,058 pour RF),
  `FUND_pb` devient la 2ᵉ feature la plus importante (15,8 % de l'importance totale).
- **Le portefeuille n'en profite pas** : Sharpe net *baisse* de 1,21 à 0,88 sur la fenêtre test
  gelée. Les intervalles de confiance marginaux se chevauchent ; ils quantifient
  l'incertitude mais ne testent pas la différence entre stratégies. Les estimations
  ponctuelles vont toutes dans le mauvais sens.
Trois tests indépendants (Phase 5 sur prix, Deep-Morocco sur 20 ans de prix, fondamentaux) convergent
vers le même constat : à l'échelle de cet univers, le signal F7 de prédiction de rendement
n'établit pas d'avantage incrémental sur la ligne de base régime + covariance dynamique. Détails :
[`docs/FUNDAMENTALS_EXPERIMENT.md`](docs/FUNDAMENTALS_EXPERIMENT.md) ·
`experiments/fundamentals_ic_lift.py` · `experiments/fundamentals_portfolio.py` ·
`src/fundamentals.py`.

## État du projet

| Phase | Description | Statut |
|-------|--------------|--------|
| Phase 1 | Infrastructure de données (Bronze/Silver/Gold) | ✅ Terminée |
| Phase 2 | Baseline Markowitz + backtesting sans biais de lookahead | ✅ Terminée |
| Phase 3 | Feature engineering ML | ✅ Terminée |
| Phase 4 | Modèles ML (HMM + covariance dynamique) | ✅ Terminée |
| Phase 4B | Modèles de signal ML adaptatifs (F7 : RandomForest + XGBoost) | ✅ Terminée |
| Phase 4C | Optimisation sensible aux coûts + régularisation de μ | ✅ Terminée |
| Phase 5 | Évaluation out-of-sample (sélection strictement antérieure, test gelé, comparaisons pairées) | ✅ Terminée |
| Phase 6+7 | Suite Portfolio ML — dashboard Streamlit + API REST FastAPI | ✅ Terminée |

## Suite Portfolio ML (dashboard + API)

Les phases 6 et 7 sont livrées comme **une application de recherche** Streamlit à deux
pages, partageant une couche de données unique — deux pages ne peuvent donc jamais afficher
des chiffres divergents pour la même stratégie. Ce démonstrateur ne constitue pas un outil
de gestion en production et ne doit pas servir à prendre ou exécuter une décision d'investissement.

```bash
# 1. Générer les artefacts que le dashboard lit (ou : ./scripts/dvc.sh repro dashboard_data)
./.venv/bin/python src/run_dashboard_data.py

# 2. Lancer le dashboard
streamlit run dashboard/streamlit_app.py

# 3. (Optionnel) Lancer l'API REST — documentation interactive sur /docs
uvicorn api.main:app --app-dir src
```

- **📊 Résultats de recherche** — page destinée aux décideurs : l'écart de Sharpe
  **observé** entre la stratégie à régimes et la meilleure approche classique, sur la
  période walk-forward complète et net de coûts, avec le numéraire de chaque univers
  (`full_2021` en MAD non couvert, `etf_2017` en USD), les intervalles de confiance de
  la fenêtre de test, la chronologie des régimes et les limites. Les valeurs courantes
  figurent dans le bloc « Faits publiés » ci-dessous ; aucun écart n'est présenté comme
  une supériorité statistiquement démontrée, dans un sens comme dans l'autre.
- **🛠️ Explorateur de stratégies** — comparaison de stratégies, métriques nettes de coûts,
  allocations historiques et export CSV, réservée à l'analyse de recherche.
- **API REST** (`src/api/`) — `/strategies`, `/metrics`, `/equity`, `/weights`, `/compare` et les
  contrats publiés `/version`, `/published-allocation`, `/explanations/{universe}`, `/model-card/{system}`.
  Elle sert les mêmes artefacts Gold versionnés, ne réentraîne jamais un modèle en requête et ne
  fournit ni conseil, ni recommandation client, ni exécution. Les contrats publiés incluent la
  provenance du snapshot ; voir [`docs/INFERENCE_CONTRACT.md`](docs/INFERENCE_CONTRACT.md).

**Garde-fou d'intégrité :** aucun chiffre n'est codé en dur. `tests/test_run_dashboard_data.py`
vérifie que chaque chiffre affiché découle bien des artefacts Gold produits par le même passage
(y compris par inspection du code source, pour interdire toute valeur saisie à la main).

## Structure du dépôt

```
├── src/                  # Pipeline de données (ingestion, nettoyage, features, validation)
│   ├── regime.py         # Détection de régime HMM (Phase 4)
│   ├── dcc_garch.py      # Estimateur DCC-GARCH (Phase 4)
│   ├── run_phase4.py     # Comparaison Phase 4 vs. haie Phase 2 (MLflow)
│   ├── ml_signals.py     # Panel de features par actif + prédiction de rendement (Phase 4B / F7)
│   ├── run_phase4b.py    # Comparaison Phase 4B vs. haie Phase 4 (MLflow)
│   ├── run_phase4c.py    # Optimisation sensible aux coûts + régularisation μ (Phase 4C)
│   ├── fundamentals.py   # Fondamentaux point-in-time BVC (expérience 2026-07)
│   ├── run_dashboard_data.py  # Artefacts consommés par le dashboard (Phase 6+7)
│   ├── api/              # Service REST FastAPI (Phase 6)
│   └── orchestration/    # Assets Dagster (planification quotidienne du pipeline)
├── dashboard/            # Application Streamlit à deux pages (Phase 6+7)
│   ├── streamlit_app.py  # Point d'entrée
│   ├── pages/            # 1_Resultats_recherche · 2_Explorateur_strategies
│   └── shared/           # Couche de données + bibliothèque de graphiques communes
├── experiments/          # Notes de recherche (deep-Morocco, fondamentaux)
├── docs/                 # Walkthrough Phase 1 + livrables encadrant (Phases 1-5)
├── notebooks/            # Notebooks de validation évidentielle (P1-P4, par phase)
├── tests/                # Tests unitaires + test d'intégration (fixtures synthétiques, hors ligne)
├── scripts/              # setup_launchd.sh — planification autonome sous macOS
├── data/                 # Bronze/Silver/Gold (géré par DVC, non versionné dans git)
├── workspace.yaml        # Point d'entrée Dagster
├── dvc.yaml              # Pipeline DVC (ingest → clean → features)
├── params.yaml           # Paramètres du pipeline
└── requirements.txt
```

## Installation

```bash
# 1. Créer et activer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell ou cmd)

# 2a. Installer les dépendances (développement — versions souples)
pip install -r requirements.txt

# 2b. OU rejouer l'environnement EXACT des résultats publiés
pip install -r requirements.lock.txt
```

### Environnement testé

Les résultats versionnés ont été produits sous **Python 3.11.14 / macOS (arm64)** avec les
versions figées dans [`requirements.lock.txt`](requirements.lock.txt), notamment
`xgboost==3.2.0`, `scikit-learn==1.9.0`, `numpy==2.4.6`, `pandas==2.3.3`, `hmmlearn==0.3.3`.

`requirements.txt` déclare des intervalles `>=` : il désigne une *famille* d'environnements,
pas celui d'où viennent les chiffres. La distinction n'est pas théorique — c'est sous
`xgboost 3.2.0` qu'a été diagnostiqué l'arrêt natif (`-11`) de la Phase 5 et validée la
politique « un seul worker natif » ; une autre version pourrait modifier les résultats ou
réactiver ce défaut. Le fichier figé fait donc partie des entrées du manifeste de snapshot.

> **Vérification rapide sans configuration :** `pytest` fonctionne immédiatement après
> l'installation — les 417 tests sont hors-ligne (aucune clé API, aucune donnée requise) —
> et les notebooks se consultent avec leurs résultats déjà exécutés. En revanche,
> `python src/pipeline.py` nécessite la clé FRED ci-dessous et un accès internet :
> le dossier `data/` n'est pas versionné dans git et se génère à la première exécution.

Créer un fichier `.env` à la racine avec votre clé API FRED
(gratuite sur https://fred.stlouisfed.org/docs/api/api_key.html) :

```
FRED_API_KEY=votre_clé_ici
```

> ⚠️ **macOS :** ne pas placer le projet sous `~/Desktop`, `~/Documents` ou `~/Downloads` —
> la protection TCC de macOS empêche les processus lancés par launchd (Dagster) d'y accéder.

## Protocole de validation et preuve statistique (Phase 2)

### Sélection strictement antérieure (forward-only)

La sélection des hyperparamètres ML n'utilise plus la validation croisée purgée
en K blocs. Celle-ci entraînait le modèle sur **toutes** les dates hors de la bande
de purge, y compris **postérieures** au bloc de validation : défendable pour des
labels chevauchants, mais ce n'est pas une simulation de décisions séquentielles
d'allocation. Noter un pli de 2018 avec un modèle ajusté en partie sur 2019-2024
répond à une question qu'aucun gérant ne peut poser en direct.

`PurgedWalkForwardSplit` (`src/purged_kfold.py`) impose, sur chaque pli :

```
train_start <= train_end < embargo_start <= val_start <= val_end < test_start
```

- fenêtre **expansible** par défaut (elle reproduit le réentraînement du backtest) ;
- **purge** = horizon du label, **embargo** = tampon distinct de corrélation sérielle ;
- regroupement **par date** : une date n'est jamais coupée entre train et validation ;
- `InsufficientHistory` est **levée** si l'historique ne permet pas la géométrie
  configurée — un protocole silencieusement dégradé produirait quand même des
  hyperparamètres « choisis », sans que rien en aval ne puisse le détecter.

`PurgedKFold` est **conservée et toujours testée** : elle n'a pas changé de sens,
c'est l'appelant qui a changé d'outil. La géométrie réellement appliquée est
publiée dans [`data/gold/phase5_validation_protocol.json`](data/gold/) — dates de
train/embargo/validation, effectifs, et **IC par pli** (pas seulement sa moyenne :
une moyenne de 0,02 sur cinq plis stables et une moyenne de 0,02 tirée par un seul
pli à 0,12 sont le même nombre et deux preuves très différentes).

**La fenêtre de test gelée n'est pas touchée** : la sélection ne voit que le segment
train+validation, donc `val_end < test_start` par construction.

### Comparaison pairée — la preuve exigée pour toute affirmation de supériorité

Un intervalle de confiance marginal décrit l'incertitude **d'une** stratégie ; il ne
teste **pas** si deux stratégies diffèrent. Deux intervalles peuvent se chevaucher
largement alors que la différence pairée est systématiquement positive, parce que les
stratégies partagent les mêmes journées de marché et que leurs erreurs sont corrélées.
L'inverse est vrai aussi : un non-chevauchement n'établit pas une différence.

`metrics.paired_block_bootstrap` teste la différence directement, sur les **mêmes
dates** de la fenêtre gelée, **nette de coûts** : les deux séries sont rééchantillonnées
avec les **mêmes indices de blocs**, ce qui préserve la dépendance sérielle de chaque
stratégie et la corrélation intra-journalière entre elles.

La **p-value est centrée sous l'hypothèse nulle** : les différences rééchantillonnées
sont recentrées sur zéro pour simuler « aucune surperformance », puis on mesure la
part de cette distribution au-delà de la différence **observée**. Ce n'est **pas** la
fraction brute de tirages positifs — celle-ci est rapportée séparément sous
`prob_sharpe_diff_positive` et ne doit jamais être présentée comme une p-value.

Résultats : [`data/gold/paired_comparison_results.json`](data/gold/), une ligne par
(univers, candidat, référence), avec différence de rendement annualisé, différence de
Sharpe net, intervalle pairé, p-value, probabilité de différence positive, écarts de
turnover et de coûts, et une **interprétation économique générée à partir des chiffres**
plutôt que rédigée à la main.

### Correction pour tests multiples — statut : établie

Un White Reality Check ou un Hansen SPA correct rééchantillonne le **maximum** sur les
séries de rendement de **tous** les candidats explorés, sur un index commun. Cette
exigence est désormais satisfaite : l'étape `reality_check` réévalue les **240
configurations atteignables** — 15 jeux d'hyperparamètres × 16 combinaisons de leviers,
soit l'espace que la recherche hiérarchique pouvait sélectionner, et non les 51 essais
que le registre avait enregistrés — sur les dates de test gelées, et conserve la série
de rendement nette de chacune.

**Résultat.** Face à la référence primaire **pré-spécifiée** (`regime_conditional`),
aucun des 240 candidats n'établit de surperformance, sur aucun des deux univers ni
aucune des deux statistiques (RC 0,069–0,881 ; SPA 0,093–0,636). Même en retenant le
meilleur candidat **jugé sur la fenêtre de test elle-même** — Sharpe 1,87 sur
`full_2021` contre 1,31 pour celui sélectionné honnêtement — la ligne de base à régimes
n'est pas dépassée une fois la recherche prise en compte. Cet écart 1,87 / 1,31 mesure
directement ce que la sélection sur l'échantillon de test fabrique.

Les comparaisons face à `equal_weight` sont **exploratoires par pré-spécification** :
la stratégie à régimes, et depuis la correction des dividendes le `max_sharpe`
classique, franchissent déjà ce plancher, donc le dépasser n'établit pas que la couche
ML apporte quelque chose. RC et SPA sont rapportés côte à côte, jamais l'un seul, avec
le nombre de candidats retenus par SPA — ici 227 à 240 sur 240, ce qui montre que
l'écart entre les deux tests vient de la studentisation et non de la règle d'exclusion.
Huit comparaisons externes ont été menées et ne sont pas elles-mêmes corrigées.

Détail : [`docs/MULTIPLE_TESTING.md`](docs/MULTIPLE_TESTING.md) ; artefacts
`data/gold/reality_check_results.json` et `reality_check_series.parquet`.

### Pipeline de données

```bash
# Pipeline complet Bronze → Silver → Gold, avec suivi MLflow
python src/pipeline.py

# Étapes individuelles (utile en débogage)
python src/ingest.py        # Bronze : téléchargement yfinance / FRED / BVCscrap / BAM
python src/clean.py         # Silver : alignement calendaire, log-rendements, validation
python src/features.py      # Gold   : tests de stationnarité, features macro
python src/ml_features.py   # Gold   : features ML causales (Phase 3, les 2 univers)
python src/run_backtest.py  # Backtest walk-forward + haie Phase 4 (Phase 2)
python src/run_phase4.py    # HMM régime + covariance dynamique vs. haie (Phase 4)
python src/run_phase4b.py   # Signaux ML adaptatifs (RF/XGBoost) vs. haie Phase 4 (Phase 4B)
python src/run_phase4c.py   # Optimisation sensible aux coûts + régularisation μ (Phase 4C)
python src/run_phase5.py    # Évaluation OOS : sélection forward-only + test gelé + comparaisons pairées (Phase 5)
```

### Tests

```bash
pytest                      # suite complète (417 tests, ~1 min 45, aucun accès réseau)
pytest -q                   # sortie compacte
pytest tests/test_clean.py  # un seul module
pytest -k "forward_fill"    # tests dont le nom correspond au motif
```

Les tests sont le premier réflexe de débogage : s'ils passent mais que le pipeline
échoue, le problème est externe (réseau, clé API, source de données).

### Suivi des expériences (MLflow)

```bash
mlflow ui                   # interface sur http://127.0.0.1:5000
```

### Planification (Dagster)

```bash
# Session interactive : UI + scheduler dans le terminal courant
dagster dev -w workspace.yaml
# → UI sur http://127.0.0.1:3000 (activer le schedule dans Overview → Schedules)

# Planification autonome sous macOS (survit aux redémarrages)
./scripts/setup_launchd.sh              # installe les LaunchAgents
./scripts/setup_launchd.sh --uninstall  # les désinstalle (conserve .dagster_home/)

# Vérifier la santé du daemon (le processus peut être vivant avec des threads morts !)
curl -s http://127.0.0.1:3000/graphql -H "Content-Type: application/json" \
  -d '{"query":"{ instance { daemonHealth { allDaemonStatuses { daemonType healthy } } } }"}'

# Redémarrer les services après un incident
launchctl kickstart -k gui/$(id -u)/com.portfolioml.dagster-daemon
launchctl kickstart -k gui/$(id -u)/com.portfolioml.dagster-webserver
```

### Versionnage des données (DVC)

Les données et artefacts ne sont pas versionnés par Git. DVC enregistre le graphe de
production et le cache local protège ce poste de travail, ce qui est indispensable car la
source BVC (medias24) est une fenêtre glissante : les lignes anciennes disparaissent
définitivement.

**Remote : Cloudflare R2 (bucket privé), configuré le 2026-08-03.** `.dvc/config` porte
l'URL et le point d'entrée ; les identifiants restent hors du dépôt. Chaque poste les
renseigne une fois, dans `.dvc/config.local` (exclu par `.dvc/.gitignore`) :

```bash
./scripts/dvc.sh remote modify --local origin access_key_id VOTRE_CLE
```
```bash
./scripts/dvc.sh remote modify --local origin secret_access_key VOTRE_SECRET
```

Deux réglages ne sont pas optionnels face à R2 et figurent déjà dans `.dvc/config` :
`region auto` (sans quoi boto3 échoue à résoudre la région) et `endpointurl` — écrit
sans tiret bas, orthographe que DVC refuse.

```bash
./scripts/dvc.sh pull       # récupérer les données depuis le remote (clone neuf)
./scripts/dvc.sh push       # publier le cache après toute exécution modifiant data/
./scripts/dvc.sh status     # les données correspondent-elles à dvc.lock ?
./scripts/dvc.sh checkout   # restaurer les données depuis le cache local
./scripts/dvc.sh repro      # ré-exécuter uniquement les étapes affectées par un changement
git log -p dvc.lock         # historique des versions de données
```

> **Le remote porte l'état courant, pas tout l'historique.** `dvc push --all-commits`
> échoue à collecter les commits antérieurs à 2026-07 (leur `dvc.yaml` suit une structure
> à trois étapes que DVC ne sait plus charger). Les versions anciennes n'existent donc que
> dans le cache local de ce poste : **ne jamais lancer `dvc gc` sur ce dépôt** — la commande
> économiserait 5 Mo et détruirait la seule copie de cet historique.

Après une exécution DVC, vérifier et figer le snapshot publié :

```bash
./scripts/dvc.sh repro snapshot_manifest
./.venv/bin/python src/snapshot.py verify
pytest tests/test_artifact_consistency.py -q
git add dvc.lock
```

`dvc repro` met à jour le cache et `dvc.lock`. `dvc commit` ne sert que lorsqu'un
output DVC a été produit manuellement hors de DVC. Le manifeste
`data/gold/snapshot_manifest.json` atteste le commit Git, les fichiers, leurs SHA-256
et la structure des Parquet — il décrit le snapshot, le remote le transporte.

Le cache de dividendes BVC (`data/bronze/bvc_dividends`) est produit par l'étape
`scrape_dividends` et non plus seulement déclaré comme dépendance de `clean`. La
distinction est concrète : DVC ne stocke le contenu que des *sorties*, si bien qu'un
clone neuf récupérait tous les résultats sans pouvoir relancer le pipeline qui les avait
produits. L'étape est en `persist: true` — le fetch lit le cache d'abord, donc elle ne
retourne sur le réseau que si le cache manque réellement.

### Requêtes analytiques sur la couche Gold (DuckDB)

```python
from src.utils import query_gold

# Exemple : rendements 2022 (SQL directement sur les fichiers Parquet)
df = query_gold("""
    SELECT * FROM 'data/gold/log_returns.parquet'
    WHERE Date >= '2022-01-01' AND Date <= '2022-12-31'
""")
```

### Notebooks

```bash
jupyter notebook notebooks/phase1_eda.ipynb                # EDA évidentielle (P1-P4)
jupyter notebook notebooks/phase2_backtest.ipynb            # baselines + backtesting
jupyter notebook notebooks/phase3_features.ipynb            # validation des features ML
jupyter notebook notebooks/phase4_regime_covariance.ipynb   # régime HMM + covariance dynamique
jupyter notebook notebooks/phase4b_adaptive_ml_signals.ipynb # signaux ML adaptatifs (RF/XGBoost, F7)
jupyter notebook notebooks/phase4c_cost_aware.ipynb          # optimisation sensible aux coûts + régularisation μ
jupyter notebook notebooks/phase5_oos_evaluation.ipynb       # évaluation OOS (notebook antérieur au protocole forward-only)
```

## Références principales

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley.
- DeMiguel, Garlappi & Uppal (2009). *Optimal Versus Naive Diversification*.

<!-- BEGIN RELEASE FACTS — generated by scripts/build_release_facts.py -->

### Faits publiés — ce que ce dépôt établit, et ce qu'il n'établit pas

> Bloc généré depuis `src/release_facts.py`. Les mêmes phrases, mot pour mot,
> figurent dans le rapport, le tableau de bord et l'API ; un test échoue si
> une surface s'en écarte.

1. `full_2021` est libellé en MAD, converti au taux de référence officiel de Bank Al-Maghrib (USDMAD, MAD par USD). Le portefeuille reste **non couvert** : la variation de change réalisée est incluse dans la performance, aucun contrat à terme ni coût de roulement n'est modélisé.
2. `etf_2017` est libellé en USD : cet univers ne contient que des ETF mono-devise, n'a donc jamais présenté de défaut de numéraire, et est **inchangé** par la correction.
3. Les niveaux de Sharpe ne sont pas comparables d'un univers à l'autre : les deux univers sont libellés dans des devises différentes, sur des fenêtres différentes et avec un nombre d'actifs différent.
4. Sur `full_2021`, l'écart ponctuel entre `regime_conditional` et `max_sharpe` est de **-10,47 %** (0,9571 contre 1,0690 de Sharpe net). L'écart est également défavorable sur `etf_2017` : **-1,6 %**.
5. Ces écarts ne démontrent pas une supériorité de l'approche classique : aucun test pairé de cette différence n'est présenté.
6. `regime_conditional` demeure le comparateur primaire pré-spécifié des tests White/SPA. Ce choix a été fixé avant l'observation des résultats et n'est pas réécrit maintenant que le signe de l'écart a changé.
7. Correction pour tests multiples (White 2000, Hansen 2005) sur les 240 configurations atteignables : aucun candidat n'établit de surperformance face au comparateur primaire pré-spécifié.
8. Walk-forward imbriqué (fenêtre OOS 2023-07-28 → 2026-07-24, 781 lignes) : le classement est sensible au protocole d'évaluation et à la fenêtre hors échantillon associée. Ratio de Sharpe dégonflé (DSR) = 0,6707 sur 198 configurations.
9. Intégrité des modèles : 6 ajustements sur 1 184 ont emprunté un repli d'estimateur (4 stratégies évaluées, 296 dates de rééquilibrage). Sur ces rééquilibrages, le résultat a été produit par un estimateur de substitution et non par le modèle que son étiquette désigne : les séries concernées sont des HYBRIDES. La portée de cette mesure est exactement l'ensemble compté ci-dessus : elle n'affirme pas qu'aucun repli n'est possible sur un autre instantané.
10. Aucune stratégie n'est recommandée, déployée, ni présentée comme une valeur ajoutée établie. Ce livrable est un prototype de recherche.

<!-- END RELEASE FACTS -->
