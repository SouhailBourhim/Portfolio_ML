# Montage du rapport dans le modèle INPT

Ce dossier ne contient **pas** de `main.tex` : le squelette, la page de garde et
la mise en page viennent du modèle INPT PFE. Les fichiers ci-dessous sont conçus
pour y être déposés tels quels.

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
% \input{frontmatter/resume_ar}     % ملخص — XeLaTeX/LuaLaTeX uniquement, voir §5
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
| `chap:conclusion` | Conclusion |

## 5. Le résumé en arabe

`frontmatter/resume_ar.tex` **ne compile pas avec pdfLaTeX**. Deux options :

- **Le projet est déjà en XeLaTeX ou LuaLaTeX** → ajouter au préambule les
  quatre lignes `polyglossia` indiquées en tête du fichier, installer une police
  arabe (Amiri ou Scheherazade New), puis décommenter la ligne `\input`.
- **Le projet reste en pdfLaTeX** → ne pas l'inclure. Le résumé français et
  l'abstract anglais suffisent ; un texte arabe mal rendu dessert le document.

## 6. Figures — comment les régénérer

Aucune figure n'est dessinée à la main : toutes sont produites par un script à
partir des artefacts versionnés du projet.

```bash
python scripts/build_figures_chap1.py    # chronologie (lit l'historique git)
python scripts/build_figures_chap2.py    # frontière, régimes, corrélations, P1–P4
python scripts/build_figures_chap3.py    # pipeline DVC (lit dvc.yaml)
python scripts/build_figures_chap4.py    # walk-forward, validation purgée, régimes
python scripts/build_figures_chap5.py    # équity, intervalles, crises, plafond
```

Les captures d'écran (`dashboard_page1.png`, `dashboard_page2.png`,
`dagster_assets.png`, `mlflow_ui.png`) sont prises manuellement ; elles sont
versionnées dans `assets/figures/`.

## 7. À personnaliser avant remise

- `frontmatter/dedicace.tex` — gabarit, à réécrire.
- `frontmatter/remerciements.tex` — à relire ; les noms sont ceux du dossier de
  projet.
- Page de garde du modèle : filière **Smart ICT**, année **2025–2026**,
  encadrant **M. Abdelmouttalib MAQIL** (EURAFRIC Information), période
  **18 juin – 1er août 2026**.
