# Système ML d'Optimisation de Portefeuille

**PFA (Projet de Fin d'Année)** — INPT (Institut National des Postes et Télécommunications)
**Client :** EURAFRIC Information, Bouskoura, Maroc
**Équipe :** Souhail Bourhim, Zakarya EL WALI, Yasmine BOUAJINE
**Encadrant :** Abdelmouttalib

## Objectif

Construire un système ML de qualité production permettant d'optimiser un portefeuille
combinant des actions de la Bourse de Casablanca (BVC) et des ETF internationaux, en
corrigeant quatre faiblesses structurelles de la théorie moderne du portefeuille (MPT)
de Markowitz :

| # | Problème | Solution ML |
|---|----------|-------------|
| P1 | Estimation bruitée de la covariance | Covariance dynamique (Ledoit-Wolf → EWMA → DCC-GARCH) |
| P2 | Non-stationnarité des rendements | Hidden Markov Models (régimes de marché) |
| P3 | Rupture de la diversification en crise | HMM + covariance dynamique |
| P4 | Surapprentissage du backtesting | Purged K-Fold CV + walk-forward backtesting |

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

Notebooks de validation, exécutés et lisibles avec leurs résultats :
[`phase1_eda.ipynb`](notebooks/phase1_eda.ipynb) ·
[`phase2_backtest.ipynb`](notebooks/phase2_backtest.ipynb) ·
[`phase3_features.ipynb`](notebooks/phase3_features.ipynb) ·
[`phase4_regime_covariance.ipynb`](notebooks/phase4_regime_covariance.ipynb) ·
[`phase4b_adaptive_ml_signals.ipynb`](notebooks/phase4b_adaptive_ml_signals.ipynb) ·
[`phase4c_cost_aware.ipynb`](notebooks/phase4c_cost_aware.ipynb) ·
[`phase5_oos_evaluation.ipynb`](notebooks/phase5_oos_evaluation.ipynb).

### Note de recherche — expansion des données (2026-07)

Le signal ML était-il *sous-alimenté* en données ? Test sur un univers marocain **profond de 12
actions sur ~20 ans (2005–2024, 56 000 lignes de panel ≈ 5× l'univers actuel)**, assemblé à partir
d'historiques investing.com. Résultat honnête : plus de données ont rendu le **modèle plus
intelligent** (coefficient d'information ×2–4, ~0,07) mais **pas** d'avantage de portefeuille
statistiquement significatif — le plafond est la **qualité** des données (fondamentaux), pas la
quantité. Détails : [`docs/DEEP_MOROCCO_EXPERIMENT.md`](docs/DEEP_MOROCCO_EXPERIMENT.md) ·
[`notebooks/deep_morocco_data_expansion.ipynb`](notebooks/deep_morocco_data_expansion.ipynb) ·
`experiments/deep_morocco_starvation.py`.

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

Les phases 6 (outil de production) et 7 (démonstration de valeur) sont livrées comme **une seule
application Streamlit à deux pages**, partageant une couche de données unique — deux pages ne
peuvent donc jamais afficher des chiffres divergents pour la même stratégie.

```bash
# 1. Générer les artefacts que le dashboard lit (ou : dvc repro dashboard_data)
python src/run_dashboard_data.py

# 2. Lancer le dashboard
streamlit run dashboard/streamlit_app.py

# 3. (Optionnel) Lancer l'API REST — documentation interactive sur /docs
uvicorn api.main:app --app-dir src
```

- **📊 Histoire de valeur** — page destinée aux décideurs : ce que le système ML apporte face au
  Markowitz classique (**+6,2 % de Sharpe net sur `full_2021`**), la validation hors échantillon
  avec intervalles de confiance, la chronologie des régimes détectés, et les limites énoncées
  explicitement (dont le cas `etf_2017` où le système **perd −1,6 %**).
- **🛠️ Outil du gestionnaire** — page métier : comparaison de stratégies, métriques nettes de
  coûts, allocations cibles, historique des rééquilibrages, export CSV.
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

# 2. Installer les dépendances
pip install -r requirements.txt
```

> **Vérification rapide sans configuration :** `pytest` fonctionne immédiatement après
> l'installation — les 369 tests sont hors-ligne (aucune clé API, aucune donnée requise) —
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
pytest                      # suite complète (369 tests, ~2 min, aucun accès réseau)
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

Le cache DVC protège les données contre toute perte — indispensable car la source BVC
(medias24) est une fenêtre glissante : les lignes anciennes disparaissent définitivement.

```bash
dvc status                  # les données correspondent-elles à dvc.lock ?
dvc commit                  # snapshot des données actuelles dans le cache (après un run)
dvc checkout                # restaurer les données depuis le cache (fichier supprimé/corrompu)
dvc repro                   # ré-exécuter uniquement les étapes affectées par un changement
git log -p dvc.lock         # historique des versions de données
```

Après chaque exécution du pipeline qui modifie les données : `dvc commit` puis
commiter `dvc.lock` dans git — c'est ce couple qui rend chaque version restaurable.

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
