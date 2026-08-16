# ⚠️ `docs/rapport` — version HISTORIQUE, non maintenue

**Ce répertoire n'est plus le rapport de soumission.** Il est conservé comme
archive et ne doit être ni cité, ni distribué, ni corrigé.

## Le rapport maintenu

| | |
|---|---|
| **Sources** | [`docs/rapport_final/`](../rapport_final/) |
| **PDF distribué** | [`output/pdf/Rapport_PFA_Final_2026.pdf`](../../output/pdf/Rapport_PFA_Final_2026.pdf) |

C'est **la seule** version de soumission. Le PDF distribué est une copie de
`docs/rapport_final/main.pdf` ; reconstruire l'arbre source ne le met pas à jour,
il faut recopier le fichier (`tests/test_wording_guards.py` vérifie le rendu
distribué pour cette raison).

## Pourquoi les deux existent

`docs/rapport` est la version courte antérieure : **chapitres 1 à 5 et
conclusion**, sans les chapitres 6 (numéraire et correction de change), 7
(démonstrateur et industrialisation) ni 8 (univers `global_2004`). Les deux
arbres ont divergé au fil des révisions ; ils ont chacun leur propre conclusion
et leurs propres faits publiés.

Cette divergence a déjà causé une erreur : le 2026-08-16, la conclusion de
`rapport_final` a été copiée ici par mégarde, écrasant un fait de publication
que `tests/test_artifact_consistency.py` contrôle. Le test l'a détectée, et le
répertoire a été restauré. Le marquer explicitement comme historique évite que
la confusion se reproduise.

## Règles

- **Ne pas corriger** un chiffre ou une formulation ici. Les corrections vont
  dans `docs/rapport_final/`.
- **Ne pas recopier** de contenu entre les deux arbres : leurs conclusions et
  leurs chapitres ne sont pas interchangeables.
- **Ne pas y ajouter** de nouveau chapitre. Le chapitre 8 (`global_2004`)
  n'existe que dans `rapport_final`, où les chapitres 6 et 7 le précèdent ;
  l'insérer ici produirait un document incohérent.
- `tests/test_artifact_consistency.py` continue de contrôler les faits publiés
  de cet arbre — il reste vérifié en tant qu'archive, pas maintenu en tant que
  livrable.
