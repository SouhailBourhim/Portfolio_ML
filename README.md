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
| P1 | Estimation bruitée de la covariance | DCC-GARCH |
| P2 | Non-stationnarité des rendements | Hidden Markov Models (régimes de marché) |
| P3 | Rupture de la diversification en crise | HMM + DCC-GARCH |
| P4 | Surapprentissage du backtesting | Purged K-Fold CV + walk-forward backtesting |

## Univers d'actifs

- **Actions BVC :** IAM.CS (Maroc Telecom), ATW.CS (Attijariwafa Bank), CIH.CS (CIH Bank), BCP.CS (Banque Centrale Populaire)
- **ETF internationaux :** SPY, QQQ, EEM, GLD, TLT
- **Indicateurs macro :** FRED (VIX, US10Y, DXY, HY Spread) + Bank Al-Maghrib (EUR/MAD, USD/MAD, taux directeur)

## Architecture des données

Le pipeline suit une architecture en médaillon à trois couches :

```
Bronze (données brutes, immuables)
   → Silver (nettoyage, alignement calendaire, log-rendements)
      → Gold (features prêtes pour le ML, validées par Pandera)
```

## État du projet

| Phase | Description | Statut |
|-------|--------------|--------|
| Phase 1 | Infrastructure de données (Bronze/Silver/Gold) | ✅ Terminée |
| Phase 2 | Baseline Markowitz + backtesting sans biais de lookahead | ⏳ À venir |
| Phase 3 | Feature engineering ML | ⏳ À venir |
| Phase 4 | Modèles ML (HMM + DCC-GARCH) | ⏳ À venir |
| Phase 5 | Évaluation out-of-sample | ⏳ À venir |
| Phase 6 | Production (API + dashboard) | ⏳ À venir |

## Structure du dépôt

```
├── src/                  # Pipeline de données (ingestion, nettoyage, features, validation)
├── notebooks/            # Analyse exploratoire (EDA évidentielle pour P1-P4)
├── tests/                # Tests unitaires (fixtures synthétiques)
├── data/                 # Bronze/Silver/Gold (géré par DVC, non versionné dans git)
├── dvc.yaml              # Pipeline DVC (ingest → clean → features)
├── params.yaml           # Paramètres du pipeline
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine avec votre clé API FRED :

```
FRED_API_KEY=votre_clé_ici
```

## Exécution du pipeline

```bash
python src/pipeline.py
```

Cela exécute l'ingestion (Bronze), le nettoyage et la validation (Silver), puis la
génération des features (Gold), avec suivi des expériences via MLflow.

## Tests

```bash
pytest
```

## Références principales

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Tsay, R. S. (2010). *Analysis of Financial Time Series*. Wiley.
