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

## Architecture des données

Le pipeline suit une architecture en médaillon à trois couches :

```
Bronze (données brutes, immuables)
   → Silver (nettoyage, alignement calendaire, log-rendements)
      → Gold (features prêtes pour le ML, validées par Pandera)
```

Une visite guidée complète du code de la Phase 1 (en anglais) est disponible dans
[`docs/PHASE1_WALKTHROUGH.md`](docs/PHASE1_WALKTHROUGH.md).

## État du projet

| Phase | Description | Statut |
|-------|--------------|--------|
| Phase 1 | Infrastructure de données (Bronze/Silver/Gold) | ✅ Terminée |
| Phase 2 | Baseline Markowitz + backtesting sans biais de lookahead | 🟡 Prochaine |
| Phase 3 | Feature engineering ML | ⏳ À venir |
| Phase 4 | Modèles ML (HMM + covariance dynamique) | ⏳ À venir |
| Phase 5 | Évaluation out-of-sample | ⏳ À venir |
| Phase 6 | Production (API + dashboard) | ⏳ À venir |

## Structure du dépôt

```
├── src/                  # Pipeline de données (ingestion, nettoyage, features, validation)
│   └── orchestration/    # Assets Dagster (planification quotidienne du pipeline)
├── docs/                 # Documentation technique (walkthrough Phase 1)
├── notebooks/            # Analyse exploratoire (EDA évidentielle pour P1-P4)
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
> l'installation — les 52 tests sont hors-ligne (aucune clé API, aucune donnée requise) —
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

## Commandes utiles

### Pipeline de données

```bash
# Pipeline complet Bronze → Silver → Gold, avec suivi MLflow
python src/pipeline.py

# Étapes individuelles (utile en débogage)
python src/ingest.py        # Bronze : téléchargement yfinance / FRED / BVCscrap / BAM
python src/clean.py         # Silver : alignement calendaire, log-rendements, validation
python src/features.py      # Gold   : tests de stationnarité, features macro
```

### Tests

```bash
pytest                      # suite complète (49 tests, ~3 s, aucun accès réseau)
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

### Notebook EDA

```bash
jupyter notebook notebooks/phase1_eda.ipynb
```

## Références principales

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley.
- DeMiguel, Garlappi & Uppal (2009). *Optimal Versus Naive Diversification*.
