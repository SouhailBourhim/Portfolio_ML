# Montage du rapport dans le modèle INPT

`main.tex` est la **source canonique du rapport final autonome**. Les fichiers
de ce dossier peuvent aussi être montés dans le modèle INPT PFE (page de garde et
mise en page institutionnelles), sans modifier le fond vérifié ici.

```bash
cd docs/rapport_final
tectonic -X compile main.tex        # ou : xelatex main.tex (deux passes)
```

Le build de release est vérifié à chaque modification par compilation XeTeX,
contrôle des références et inspection visuelle du PDF. Moteur : XeTeX (requis par
`fontspec` / `polyglossia`).

## 1. Fichiers à copier dans le projet Overleaf

```
rapport_style.sty              → à la racine du projet
assets/figures/*               → assets/figures/  (créer le dossier)
assets/logos/*                 → là où le modèle attend les logos
frontmatter/*.tex              → frontmatter/
chapters/*.tex                 → chapters/  (ou le dossier du modèle)
```

## 2. Préambule — trois lignes à ajouter

```latex
\usepackage{rapport_style}   % encadrés Définition / Remarque / Résultat
\usepackage{longtable}       % liste des acronymes
\usepackage{booktabs}        % (déjà chargé par rapport_style, sans effet si doublé)
```

`rapport_style.sty` charge lui-même `xcolor`, `tcolorbox[most]`, `graphicx`,
`booktabs` et `multirow`. Il n'écrase aucun réglage du modèle.

## 3. Ordre d'inclusion

```latex
% ── pages liminaires ───────────────────────────────────────────────
\input{frontmatter/dedicace}
\input{frontmatter/remerciements}
\input{frontmatter/resume}          % Résumé (fr) + Abstract (en)
\input{frontmatter/acronymes}
\input{frontmatter/glossaire}
\tableofcontents
\listoffigures
\listoftables

% ── corps ──────────────────────────────────────────────────────────
\input{chapters/Chapter1}       % Contexte : entreprise, stage, sujet
\input{chapters/Chapter2}       % Problématique : les quatre échecs (P1–P4)
\input{chapters/Chapter3}       % Architecture : chaîne de données, garde-fous
\input{chapters/Chapter4}       % Modélisation et protocole d'évaluation
\input{chapters/Chapter5}       % Résultats, validation, produit livré
\input{chapters/Chapter6}       % Révisions et gouvernance de la preuve
\input{chapters/Chapter7}       % Démonstrateur et préparation industrielle
\input{chapters/Conclusion}
```

Le glossaire peut aussi être renvoyé en annexe si le modèle le prévoit : il est
autonome et ne dépend d'aucun compteur.

## 4. Références croisées utilisées

Les chapitres se citent mutuellement ; ces étiquettes doivent rester
inchangées :

| Étiquette | Chapitre |
|---|---|
| `chap:contexte` | 1 — Contexte |
| `chap:problematique` | 2 — Problématique |
| `chap:architecture` | 3 — Architecture |
| `chap:modelisation` | 4 — Modélisation et évaluation |
| `chap:resultats` | 5 — Résultats |
| `chap:gouvernance-preuve` | 6 — Révisions et gouvernance de la preuve |
| `chap:demonstrateur` | 7 — Démonstrateur et préparation industrielle |
| `chap:conclusion` | Conclusion |

## 5. Résumés retenus dans l'édition finale

Cette édition inclut uniquement le résumé français et l'abstract anglais. Le
résumé arabe de l'édition antérieure est exclu de ce dossier et du PDF final.

## 6. Figures — comment les régénérer

Aucune figure n'est dessinée à la main : toutes sont produites par un script à
partir des artefacts versionnés du projet.

```bash
python scripts/build_figures_chap1.py    # chronologie (lit l'historique git)
python scripts/build_figures_chap2.py    # frontière, régimes, corrélations, P1–P4
python scripts/build_figures_chap3.py    # pipeline DVC (lit dvc.yaml)
python scripts/build_figures_chap4.py    # walk-forward, validation purgée, régimes
python scripts/build_figures_chap5.py    # équity, intervalles, crises, plafond
python scripts/build_final_report_figures.py  # figures transversales des chap. 6–7
```

Les captures du dashboard sont régénérées par `scripts/capture_dashboard.py` et
la documentation OpenAPI par `scripts/capture_api_docs.py`. Les captures Dagster
et MLflow sont versionnées dans `assets/figures/`.

## 7. À personnaliser avant remise

- `frontmatter/dedicace.tex` — gabarit, à réécrire.
- `frontmatter/remerciements.tex` — à relire ; les noms sont ceux du dossier de
  projet.
- Page de garde : les champs sont déjà renseignés dans `main.tex` (filière
  **Smart ICT**, promotion **2025/2026**, encadrant **M. Abdelmouttalib MAQIL**,
  période **18 juin – 1er août 2026**, projet **réalisé à distance**). Pas de
  bloc jury : il n'y en a pas à ce stade. Si une soutenance est programmée,
  ajouter les examinateurs sous « Encadré par ».
- Les logos du bandeau sont `inpt_trim.png` (gauche) et `logo_anrt.jpg`
  (droite) ; l'illustration du campus (`inpt_campus_trim.png`) est en pied de
  page, comme dans le modèle officiel.
