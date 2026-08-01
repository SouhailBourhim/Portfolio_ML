# Montage du rapport dans le modèle INPT

Le rapport final est monté dans le **modèle INPT PFE** (page de garde et mise en
page officielles). Les fichiers de ce dossier sont conçus pour y être déposés
tels quels.

`main.tex` est un **build autonome de vérification** : il n'est pas le rapport
final, mais il prouve que tout compile et produit un PDF lisible sans Overleaf.

```bash
cd docs/rapport
tectonic -X compile main.tex        # ou : xelatex main.tex (deux passes)
```

État du dernier build vérifié : **73 pages, 0 référence non résolue, 0 caractère
manquant, 2 débordements de marge résiduels (4,0 pt et 0,2 pt — invisibles à
l'impression).** Moteur : XeTeX (requis par `fontspec` / `polyglossia`).

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

`frontmatter/resume_ar.tex` **ne compile pas avec pdfLaTeX** : il exige XeLaTeX
ou LuaLaTeX. Si le projet Overleaf reste en pdfLaTeX, ne pas l'inclure — le
résumé français et l'abstract anglais suffisent.

Si le projet est en XeLaTeX/LuaLaTeX, quatre points ont été **vérifiés à la
compilation** et doivent être respectés, faute de quoi le document échoue ou
s'imprime avec des rectangles vides :

1. **Déclarer l'arabe en toute fin de préambule**, après `hyperref` et tous les
   autres paquets. `polyglossia` charge `bidi` dès qu'une langue de droite à
   gauche est déclarée, et `bidi` exige d'être chargé en dernier. Sinon :
   `Unable to properly define \@@leqno`.

2. **Utiliser l'environnement `Arabic` avec une majuscule.** En minuscules,
   `\begin{arabic}` entre en collision avec la commande LaTeX
   `\arabic{compteur}` et la compilation s'arrête sur `Missing number`. La
   majuscule est la forme documentée par polyglossia pour ce cas.

3. **Choisir une police qui contient aussi les chiffres et la ponctuation
   latins.** Geeza Pro n'a ni les chiffres ni le caractère « : » : le texte
   s'imprime avec des rectangles à la place de `1952`, `6,2` et `390`. Amiri
   convient, chargée **par nom de fichier** — la seule forme qui fonctionne à la
   fois sur Overleaf/TeX Live et avec tectonic :

   ```latex
   \setotherlanguage{arabic}
   \newfontfamily\arabicfont[Script=Arabic,Extension=.ttf,
                             UprightFont=Amiri-Regular]{Amiri}
   ```

4. **Encadrer les titres arabes de `\textarabic{}`** (`\chapter*`,
   `\addcontentsline`). Sinon ils sont composés dans la police latine du
   document, qui ne possède aucun glyphe arabe.

`main.tex` applique les quatre points et sert de référence ; l'inclusion y est
pilotée par `\ARABICRESUME` (1 = inclure, 0 = omettre).

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
