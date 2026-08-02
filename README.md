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
| P4 | Surapprentissage du backtesting | Validation purgée/embargo + backtesting walk-forward |

Le livrable est une aide à l'analyse et à la recherche : il **ne fournit ni conseil
d'investissement, ni recommandation client, ni exécution automatique d'ordres**. Il ne
doit pas être présenté comme un système de production ou comme une preuve que le ML
surperforme les méthodes classiques.

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

| | référence (25/25) | meilleur candidat | contrôle INVERSÉ | meilleur fixe |
|---|---:|---:|---:|---:|
| `full_2021` | **1,2363** | `both_40_15` 1,2663 (**+0,030**) | 1,1430 (−0,093) | 1,2333 (−0,003) |
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
  gelée (IC de confiance à 90 % chevauchant, donc non-significatif — mais toutes les estimations
  ponctuelles vont dans le mauvais sens).
Trois tests indépendants (Phase 5 sur prix, Deep-Morocco sur 20 ans de prix, fondamentaux) convergent
vers la même conclusion : à l'échelle de cet univers, le signal F7 de prédiction de rendement
n'améliore pas significativement la ligne de base régime + covariance dynamique. Détails :
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
| Phase 5 | Évaluation out-of-sample (K-Fold purgé, sélection honnête, IC bootstrap) | ✅ Terminée |
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

- **📊 Résultats de recherche** — page destinée aux décideurs : l'écart de Sharpe **observé**
  (+6,2 % sur `full_2021`, −1,6 % sur `etf_2017`) sur la période walk-forward complète, les
  intervalles de confiance de la fenêtre de test, la chronologie des régimes et les limites.
  Aucun écart n'est présenté comme une supériorité statistiquement démontrée.
- **🛠️ Explorateur de stratégies** — comparaison de stratégies, métriques nettes de coûts,
  allocations historiques et export CSV, réservée à l'analyse de recherche.
- **API REST** (`src/api/`) — `/strategies`, `/metrics`, `/equity`, `/weights`, `/compare`. Sert
  les mêmes artefacts Gold versionnés ; `/compare` renvoie toujours l'intervalle de confiance et
  la mise en garde avec l'écart de performance.

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
│   ├── pages/            # 1_Histoire_de_valeur · 2_Outil_gestionnaire
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
> l'installation — les 411 tests sont hors-ligne (aucune clé API, aucune donnée requise) —
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

## Comment vérifier qu'un résultat est à jour

**Ne pas se fier à la date de modification des fichiers.** DVC restaure les sorties depuis son
cache en conservant leur horodatage d'origine : un artefact parfaitement valide peut afficher
une date ancienne, et un fichier récent peut être périmé. L'horodatage ne dit rien.

Deux commandes font foi :

```bash
# 1. Le graphe est-il à jour vis-à-vis du code et des données ?
./scripts/dvc.sh status                 # « Data and pipelines are up to date. »

# 2. Les artefacts publiés correspondent-ils au manifeste de release ?
./.venv/bin/python src/snapshot.py verify
```

`snapshot.py verify` recalcule le SHA-256 des 23 entrées du snapshot (données brutes, matrices
Gold, résultats de phase, artefacts du dashboard, étude de crise, `params.yaml`, `dvc.yaml`,
`requirements.lock.txt`) et les compare au manifeste versionné. Il **échoue** si :

- un artefact requis est absent ou a changé d'un seul octet ;
- le manifeste a été produit depuis un arbre de travail **non commité** — le `git_commit`
  inscrit ne désignerait alors pas le code qui a produit les chiffres ;
- la révision inscrite n'est pas un ancêtre de la révision courante (autre branche, historique
  réécrit).

La révision inscrite est celle qui a **produit** les artefacts ; le commit qui enregistre le
manifeste en est nécessairement l'enfant, d'où le test d'ascendance plutôt que d'égalité.

Régénérer le manifeste après une reconstruction — **depuis un arbre propre** :

```bash
./scripts/dvc.sh repro --force snapshot_manifest
```

> **Portabilité — limite assumée.** Aucun *remote* DVC n'est configuré : le snapshot est
> vérifiable localement, mais les données ne peuvent pas être reconstruites à partir du seul
> dépôt Git. Un relecteur externe a besoin soit d'un remote DVC approuvé, soit d'une archive
> de release fournie séparément. Les données de marché ne sont pas republiées ici (licences).

## Docker (environnement reproductible)

Alternative à l'installation locale : l'image fige l'OS, la version de Python et toutes les
dépendances — le projet s'exécute à l'identique sur macOS, Windows ou Linux (seul
[Docker Desktop](https://www.docker.com/products/docker-desktop/) est requis).

**Option A — image pré-construite (Docker Hub).** L'image exacte que nous avons validée
(multi-architecture : amd64 + arm64), sans rien construire :

```bash
docker pull souhailbourhim/portfolio-ml:phase1
docker run --rm souhailbourhim/portfolio-ml:phase1        # exécute la suite de tests
```

**Option B — construire depuis les sources :**

```bash
# 1. Vérification sans aucune configuration : la suite de tests (hors-ligne)
docker compose run --rm test

# 2. Pipeline complet (nécessite un fichier .env avec FRED_API_KEY)
docker compose run --rm pipeline

# 3. Notebooks dans le navigateur (token affiché dans les logs)
docker compose up notebook        # → http://localhost:8888
```

Les données ne sont pas incluses dans l'image (elles sont générées à l'exécution et gérées
par DVC) : `data/` et `mlruns/` sont montés depuis l'hôte, donc les sorties persistent et
restent visibles par DVC. La planification Dagster/launchd reste hors périmètre Docker
(spécifique macOS).

## Commandes utiles

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
python src/run_phase5.py    # Évaluation OOS : K-Fold purgé + sélection honnête + IC bootstrap (Phase 5)
```

### Tests

```bash
pytest                      # suite complète (390 tests, ~2 min, aucun accès réseau)
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

> **État actuel : aucun remote DVC partagé n'est configuré dans ce dépôt.** Un clone
> neuf ne peut donc pas récupérer seul les données ni reproduire les chiffres publiés.
> Avant une remise ou une revue externe, il faut publier le snapshot dans un remote DVC
> contrôlé ou fournir une archive de données avec son manifeste et ses checksums.

```bash
./scripts/dvc.sh status     # les données correspondent-elles à dvc.lock ?
./scripts/dvc.sh checkout   # restaurer les données depuis le cache (fichier supprimé/corrompu)
./scripts/dvc.sh repro      # ré-exécuter uniquement les étapes affectées par un changement
git log -p dvc.lock         # historique des versions de données
```

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
et la structure des Parquet ; il ne remplace pas un remote DVC ou une archive contrôlée.

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
jupyter notebook notebooks/phase5_oos_evaluation.ipynb       # évaluation OOS : K-Fold purgé + IC bootstrap
```

## Références principales

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley.
- DeMiguel, Garlappi & Uppal (2009). *Optimal Versus Naive Diversification*.
