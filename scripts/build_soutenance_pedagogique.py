"""build_soutenance_pedagogique.py — the vulgarised defence deck, in French.

WHY A SECOND DECK. `Soutenance_PFA_Portfolio_ML_INPT_EURAFRIC.pptx` is written
for a reader who already knows what a covariance matrix is and who wants the
negative headline early. The PFA jury is a mixed academic panel with no finance
background: it needs the vocabulary before the result, and the argument needs to
lead with what the work establishes rather than with what it refutes.

This deck therefore re-sequences the same evidence: define, then problem, then
method, then findings. It never states a performance number that is not read
from a committed Gold artifact — the slide text is assembled from
`src/release_facts.py` and `data/gold/*.json`, so a stale deck fails loudly
instead of showing an outdated Sharpe on a projector.

HONESTY. Slide "Ce que nous établissons" states the primary comparison result
plainly. It is kept deliberately: a jury asks, and a deck that hides its own
headline is worth less than one that frames it. The framing is the contribution
— the chain is what makes the negative statement sayable at all.

Usage:
    python scripts/build_soutenance_pedagogique.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from release_facts import load_facts  # noqa: E402

GOLD = ROOT / "data" / "gold"
FIG_FR = ROOT / "output" / "presentation" / "_figures_fr"
FIG_REPORT = ROOT / "docs" / "rapport_final" / "assets" / "figures"
# `walkforward.png` here is the PNG rendering of the chapter-4 figure that
# scripts/build_figures_chap4.py writes as PDF; it is committed because the deck
# needs a raster and the chapter builder only emits vector.
FIG_RAPPORT = ROOT / "docs" / "rapport" / "assets" / "figures"
OUT = ROOT / "output" / "presentation" / "Soutenance_PFA_Vulgarisee_FR.pptx"

# House grid, extracted from the existing deck (see build_slides_global_2004.py).
BLUE, INK, GREY = RGBColor(0x3D, 0x8D, 0xFF), RGBColor(0x11, 0x18, 0x27), RGBColor(0x5D, 0x66, 0x75)
RED, TEAL, AMBER = RGBColor(0xB7, 0x43, 0x3E), RGBColor(0x0F, 0x76, 0x6E), RGBColor(0xA8, 0x68, 0x12)
NAVY, GREEN = RGBColor(0x1B, 0x3A, 0x6B), RGBColor(0x2F, 0x7D, 0x3B)
PANEL, CALLOUT = RGBColor(0xF3, 0xF4, 0xF6), RGBColor(0xFF, 0xF0, 0xD5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = 960.0, 540.0          # slide, in points

# Speaker notes, one per slide, in build order. ~20 minutes of spoken French.
NOTES = [
    "Bonjour. Nous présentons un projet de fin d’année mené à l’INPT avec EURAFRIC "
    "Information : un système qui répartit un capital entre actifs, et surtout un "
    "dispositif capable de dire si cette répartition est réellement meilleure. "
    "[~40 s]",

    "Quatre temps. Je commence par le vocabulaire, parce que la suite n’a de sens "
    "qu’avec quatre ou cinq mots précis. Puis le problème, notre réponse, les "
    "résultats. [~30 s]",

    "Un portefeuille, c’est un capital réparti entre plusieurs actifs. La seule "
    "chose que notre système décide, ce sont ces pourcentages — les poids. Il ne "
    "prédit pas un cours ; il répond à « combien mettre où ». Et il reprend cette "
    "décision chaque mois, ce qui coûte des frais à chaque fois. [~60 s]",

    "Pour comparer deux portefeuilles, le rendement seul ne suffit pas. Ces deux "
    "courbes finissent au même niveau, mais l’une tremble trois fois plus. Le ratio "
    "de Sharpe mesure exactement cela : le rendement obtenu par unité de risque "
    "subi. C’est la note qui servira à tout comparer. [~60 s]",

    "L’idée de Markowitz, prix Nobel : en combinant des actifs qui ne bougent pas "
    "ensemble, on réduit le risque sans perdre de rendement. La courbe bleue est "
    "l’ensemble des meilleurs compromis. Le point important : le modèle a besoin de "
    "deux ingrédients qu’on ne connaît jamais vraiment — il les estime. [~60 s]",

    "Quatre mots qui décident si un résultat vaut quelque chose. Un backtest rejoue "
    "l’histoire. Une fuite de données, c’est utiliser sans le vouloir une "
    "information du futur. Le surapprentissage, c’est apprendre le bruit. Et le "
    "data snooping : à force d’essayer, on finit toujours par trouver un gagnant "
    "apparent. Tout notre protocole est construit contre ces quatre risques. [~70 s]",

    "Voilà la question. Elle est posée sous contraintes réelles de gestion : pas de "
    "vente à découvert, aucun actif au-delà de 25 %, frais déduits, décision "
    "mensuelle. Et nous l’étudions sur trois terrains évalués séparément — nous ne "
    "mélangeons jamais des dirhams et des dollars. [~60 s]",

    "Pourquoi c’est difficile. Quatre échecs connus de l’approche classique : elle "
    "prend des estimations bruitées pour des certitudes, elle suppose que demain "
    "ressemble à hier, elle s’effondre en crise, et elle se laisse flatter par la "
    "répétition des essais. Chacun appelle une réponse technique précise — c’est "
    "notre plan de construction. [~70 s]",

    "Une image pour le troisième échec. En temps normal, actions et valeurs refuges "
    "se comportent différemment. En crise, les actions se resserrent : la "
    "corrélation monte jusqu’à 0,94 en mars 2020. Le portefeuille se comporte alors "
    "comme un seul actif — la diversification disparaît quand on en a le plus "
    "besoin. [~60 s]",

    "Le piège central, et c’est notre message principal. Nous avons évalué 240 "
    "stratégies, liste figée avant tout calcul. Le meilleur candidat dépasse la "
    "référence de 0,093 de Sharpe. Sur le papier, c’est publiable. Les tests de "
    "sélection multiple répondent p = 0,90 : c’est le maximum d’une recherche, pas "
    "une découverte. Sans ce test, nous aurions annoncé un faux résultat. [~80 s]",

    "Notre réponse est une chaîne complète, à sens unique, de la donnée brute à la "
    "décision publiée. Trois règles la gouvernent : aucun chiffre saisi à la main, "
    "aucune décision qui regarde l’avenir, aucune affirmation sans test. [~50 s]",

    "Les données d’abord. Trois couches : le brut qu’on ne réécrit jamais, une "
    "couche de mise en cohérence, et une couche finale — la seule que les modèles "
    "ont le droit de lire. Cela nous a permis de corriger deux biais réels : les "
    "dividendes de la Bourse de Casablanca, et la conversion en dirhams au taux "
    "officiel de Bank Al-Maghrib. Les deux ont changé nos résultats ; nous les avons "
    "appliqués quand même. [~75 s]",

    "Les modèles, par paliers. On part des références classiques, on rend la "
    "structure de risque dynamique, puis on ajoute un modèle de régimes — un HMM "
    "qui reconnaît tout seul si le marché est calme ou sous tension — et enfin des "
    "challengers Random Forest et XGBoost. Chaque étage doit battre le précédent "
    "pour être retenu. [~70 s]",

    "L’épreuve. À chaque date de décision, le modèle est réentraîné avec les seules "
    "données déjà disponibles, puis subit le mois suivant sans le connaître. On "
    "écarte les jours à cheval sur la frontière, on garde une dernière période "
    "totalement intacte, et chaque mouvement est facturé avant d’être compté. [~65 s]",

    "Trois verrous entre un bon chiffre et une affirmation : la comparaison est "
    "appariée jour par jour ; le fait d’avoir essayé 240 configurations est corrigé "
    "statistiquement ; et la référence à battre a été fixée à l’avance, jamais "
    "réécrite après coup. C’est ce qui rend nos résultats vérifiables. [~60 s]",

    "Les résultats, hors échantillon et nets de frais. Les stratégies se tiennent, y "
    "compris à travers 2008, 2020 et 2022 — aucune ne décroche. La vraie question "
    "n’est donc pas « est-ce que ça marche », mais « l’écart entre ces courbes "
    "est-il réel ». C’est exactement ce que la chaîne sait trancher. [~60 s]",

    "Premier résultat concret. Le modèle de régimes n’a jamais reçu la liste des "
    "crises : il apprend seul. Sur les cinq crises de la période, fixées à l’avance "
    "d’après des dates externes, il les a toutes reconnues. 92 % des mois de crise "
    "classés « sous tension », contre 29 % hors crise. C’est un diagnostic "
    "directement lisible par un gérant. [~70 s]",

    "Deuxième résultat, et le plus actionnable. À moteur identique, relâcher la "
    "règle « aucun actif au-delà de 25 % » coûte 10 % de Sharpe. C’est davantage que "
    "l’effet de n’importe quel modèle testé. Une règle de gouvernance simple absorbe "
    "l’erreur d’estimation mieux que la sophistication. [~65 s]",

    "Troisième résultat, celui dont nous sommes le plus fiers. Avec un backtest "
    "seul, nous aurions écrit « notre XGBoost améliore le Sharpe de 0,093 ». Notre "
    "chaîne répond : c’est le maximum d’une recherche, les tests ne le retiennent "
    "pas. Un prototype qui ne sait pas invalider ses propres résultats n’a pas de "
    "valeur pour une direction des risques. [~70 s]",

    "La lecture d’ensemble, en toute transparence. À gauche, ce que nous "
    "établissons. À droite, ce que nous n’établissons pas : sur nos univers et nos "
    "fenêtres, aucune couche ML ne démontre de surperformance significative face à "
    "son comparateur. C’est une réponse, pas un échec — la question posée était "
    "« peut-on le savoir », et nous savons. Les deux colonnes viennent du même "
    "dispositif : c’est ce qui rend la première crédible. [~80 s]",

    "Le livrable n’est pas un notebook. Un tableau de bord, une API en lecture seule "
    "dont les chiffres sont lus depuis les artefacts, des données versionnées, une "
    "intégration continue, un conteneur, et près de 900 tests automatisés qui "
    "gardent les invariants. [~55 s]",

    "Ce qui reste. Élargir l’univers marocain, faire valider le dispositif par une "
    "équipe indépendante, modéliser la couverture du risque de change, et passer à "
    "l’échelle sur davantage d’actifs — c’est là qu’un optimiseur a le plus de marge "
    "pour exprimer une vue. [~55 s]",

    "En conclusion : un système complet d’allocation, évalué sans tricher, capable "
    "de reconnaître les crises, de montrer ce qui améliore réellement un "
    "portefeuille — et de refuser ce qui n’est que du hasard. Merci de votre "
    "attention. [~40 s]",

    "ANNEXE — à garder pour les questions. Les chiffres exacts par univers, et la "
    "mise en garde : les Sharpe ne se comparent pas d’un univers à l’autre "
    "(devises, fenêtres et nombres d’actifs différents).",

    "ANNEXE — intégrité et périmètre. 7 ajustements sur 1 184 ont utilisé un "
    "estimateur de repli : ils sont comptés et signalés comme hybrides plutôt que "
    "publiés sous une étiquette qu’ils ne méritent pas. Et le cadrage : prototype de "
    "recherche académique, aucune recommandation d’investissement.",
]


# ── number formatting ────────────────────────────────────────────────────────

def fr(value: float, dp: int = 4, signed: bool = False) -> str:
    """One number, French convention: decimal comma and a true minus sign."""
    text = f"{value:+.{dp}f}" if signed else f"{value:.{dp}f}"
    return text.replace(".", ",").replace("-", "−")


def pct(value: float, dp: int = 1, signed: bool = True) -> str:
    return fr(value, dp, signed) + " %"


# ── primitives ───────────────────────────────────────────────────────────────

def text(slide, x, y, w, h, s, size, *, bold=False, color=INK, align=None, spacing=None):
    box = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(s.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = para.add_run()
        run.text = line
        run.font.name, run.font.size, run.font.bold = "Arial", Pt(size), bold
        run.font.color.rgb = color
        if align is not None:
            para.alignment = align
        if spacing:
            para.line_spacing = spacing
    return box


def panel(slide, x, y, w, h, fill=PANEL):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Pt(x), Pt(y), Pt(w), Pt(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    try:
        shape.adjustments[0] = 0.03
    except Exception:
        pass
    return shape


def rule(slide, x, y, w, color=BLUE, thickness=3.0):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Pt(x), Pt(y), Pt(w), Pt(thickness))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def kpi(slide, x, y, w, value, caption, color, *, size=30):
    text(slide, x, y, w, size + 16, value, size, bold=True, color=color)
    text(slide, x, y + size + 16, w, 40, caption, 11.5, color=GREY, spacing=1.2)


def card(slide, x, y, w, h, titre, corps, color, *, fill=PANEL, ts=13, cs=11.5):
    panel(slide, x, y, w, h, fill)
    rule(slide, x + 18, y + 18, 34, color)
    text(slide, x + 18, y + 32, w - 36, 22, titre, ts, bold=True, color=color)
    text(slide, x + 18, y + 58, w - 36, h - 74, corps, cs, color=GREY, spacing=1.32)


def picture(slide, path: Path, x, y, max_w, max_h):
    """Place a picture scaled to fit (max_w, max_h) and centred in that box."""
    if not path.exists():
        raise SystemExit(f"figure missing: {path}\nRun scripts/build_figures_soutenance_fr.py first.")
    pic = slide.shapes.add_picture(str(path), Pt(x), Pt(y))
    ratio = min(max_w / Emu(pic.width).pt, max_h / Emu(pic.height).pt)
    pic.width, pic.height = Pt(Emu(pic.width).pt * ratio), Pt(Emu(pic.height).pt * ratio)
    pic.left = Pt(x + (max_w - Emu(pic.width).pt) / 2)
    pic.top = Pt(y + (max_h - Emu(pic.height).pt) / 2)
    return pic


def new_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)
    return slide


class Deck:
    """Slide factory that numbers pages as they are added."""

    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.3333), Inches(7.5)
        self.n = 0

    def slide(self, eyebrow: str, titre: str, *, sous: str | None = None):
        self.n += 1
        s = new_slide(self.prs)
        text(s, 48, 22, 560, 16, eyebrow.upper(), 9, bold=True, color=BLUE)
        text(s, 48, 44, 840, 44, titre, 26, bold=True, color=INK)
        text(s, 900, 27, 34, 18, f"{self.n:02d}", 9, bold=True, color=GREY,
             align=PP_ALIGN.RIGHT)
        if sous:
            text(s, 48, 86, 840, 20, sous, 12.5, color=GREY)
        return s

    def bare(self):
        self.n += 1
        return new_slide(self.prs)

    def caption(self, s, txt, y=500):
        text(s, 48, y, 864, 30, txt, 11, color=GREY, spacing=1.25)

    def attach_notes(self, notes: list[str]) -> None:
        """Write the speaker notes, one per slide, in build order.

        Kept as a flat list rather than threaded through every slide call: the
        length assertion below then fails loudly the moment a slide is added or
        removed without its note, which is the failure a rehearsal would
        otherwise discover the evening before.
        """
        if len(notes) != self.n:
            raise SystemExit(f"{self.n} slides built but {len(notes)} notes written")
        for slide, txt in zip(self.prs.slides, notes):
            slide.notes_slide.notes_text_frame.text = txt

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(path))


# ── evidence ─────────────────────────────────────────────────────────────────

def evidence() -> dict:
    facts = load_facts()
    crisis = json.loads((GOLD / "crisis_windows.json").read_text(encoding="utf-8"))
    cap = json.loads((GOLD / "etf_cap_verdict.json").read_text(encoding="utf-8"))
    q1 = json.loads((GOLD / "global_2004_q1_results.json").read_text(encoding="utf-8"))
    q2 = json.loads((GOLD / "global_2004_q2_results.json").read_text(encoding="utf-8"))
    ready = json.loads((GOLD / "global_2004_readiness.json").read_text(encoding="utf-8"))

    best_by_cap = {c: max(r.items(), key=lambda kv: kv[1]["sharpe_net"])
                   for c, r in cap["results"].items()}
    badge = re.search(r"tests-(\d+)", (ROOT / "README.md").read_text(encoding="utf-8"))
    if badge is None:
        raise SystemExit("README.md no longer carries the test-count badge the deck reads.")

    return {
        "facts": facts,
        "crisis": crisis["regime_detection"]["etf_2017"],
        "n_crises": len(crisis["crises"]),
        "cap": {c: (k, v["sharpe_net"]) for c, (k, v) in best_by_cap.items()},
        "q1": q1,
        "q2": q2,
        "ready": ready,
        "n_tests": int(badge.group(1)),
    }


# ── the deck ─────────────────────────────────────────────────────────────────

def build() -> Path:
    e = evidence()
    f = e["facts"]
    full, etf = f["universes"]["full_2021"], f["universes"]["etf_2017"]
    fb = f["fallbacks"]
    d = Deck()

    # 01 — title ─────────────────────────────────────────────────────────────
    s = d.bare()
    panel(s, 0, 0, W, H, WHITE)
    rule(s, 48, 96, 96, BLUE, 4)
    text(s, 48, 52, 700, 18, "PROJET DE FIN D’ANNÉE · INPT × EURAFRIC INFORMATION",
         10.5, bold=True, color=BLUE)
    text(s, 48, 128, 830, 120,
         "Optimisation de portefeuille\npilotée par Machine Learning", 40, bold=True,
         color=INK, spacing=1.12)
    text(s, 48, 262, 760, 56,
         "Peut-on améliorer la répartition d’un capital grâce à l’apprentissage "
         "automatique —\net savoir, de façon vérifiable, si l’amélioration est réelle ?",
         15, color=GREY, spacing=1.35)
    panel(s, 48, 352, 864, 1, RGBColor(0xC9, 0xCE, 0xD6))
    text(s, 48, 380, 430, 70,
         "Souhail BOURHIM · Zakarya EL WALI · Yasmine BOUAJINE", 13, bold=True, color=INK)
    text(s, 48, 404, 430, 40, "Étudiants ingénieurs · INPT", 11.5, color=GREY)
    text(s, 560, 380, 352, 70, "Encadré par M. Abdelmouttalib MAQIL", 13, bold=True, color=INK)
    text(s, 560, 404, 352, 40, "EURAFRIC Information", 11.5, color=GREY)
    text(s, 48, 470, 500, 20, "Soutenance — 2026", 11, color=GREY)

    # 02 — roadmap ───────────────────────────────────────────────────────────
    s = d.slide("Feuille de route", "Quatre temps, une seule question")
    parties = [
        ("1", "Le vocabulaire", "Portefeuille, risque, Sharpe,\nbacktest : tout ce qu’il faut\npour suivre la suite.", NAVY),
        ("2", "Le problème", "Pourquoi répartir un capital\nest difficile — et pourquoi\nun bon backtest peut mentir.", RED),
        ("3", "Notre réponse", "Une chaîne complète : données,\nmodèles, épreuve temporelle,\ntests statistiques.", TEAL),
        ("4", "Les résultats", "Ce que le système détecte,\nce qui améliore vraiment,\net ce qui est livré.", AMBER),
    ]
    for i, (num, titre, corps, col) in enumerate(parties):
        x = 48 + i * 220
        panel(s, x, 136, 200, 300)
        text(s, x + 20, 162, 60, 46, num, 34, bold=True, color=col)
        text(s, x + 20, 220, 160, 24, titre, 15, bold=True, color=INK)
        text(s, x + 20, 258, 168, 150, corps, 11.5, color=GREY, spacing=1.45)
    d.caption(s, "Le fil conducteur : une décision d’investissement n’a de valeur que si "
                 "l’on peut prouver qu’elle n’est pas le fruit du hasard.", 468)

    # 03 — vocabulary 1 ──────────────────────────────────────────────────────
    s = d.slide("Le vocabulaire · 1/4", "Un portefeuille, une allocation, un rééquilibrage",
                sous="Le système ne choisit pas « quand acheter » : il choisit des pourcentages, "
                     "tous les mois.")
    picture(s, FIG_FR / "allocation_expliquee.png", 48, 118, 864, 320)
    panel(s, 48, 452, 864, 58, CALLOUT)
    text(s, 70, 466, 820, 40,
         "À retenir : la sortie du modèle est un vecteur de poids. Tout le reste — "
         "performance, risque, frais — en découle mécaniquement.",
         12.5, bold=True, color=INK, spacing=1.25)

    # 04 — vocabulary 2 : Sharpe ─────────────────────────────────────────────
    s = d.slide("Le vocabulaire · 2/4", "Le ratio de Sharpe : la note qui compte",
                sous="Gagner beaucoup en dormant mal ne vaut pas gagner autant tranquillement.")
    picture(s, FIG_FR / "sharpe_explique.png", 48, 118, 600, 340)
    card(s, 668, 130, 244, 150, "Rendement",
         "Ce que le portefeuille rapporte,\nen pourcentage, sur une année.", TEAL)
    card(s, 668, 292, 244, 166, "Risque (volatilité)",
         "L’amplitude des secousses :\nplus la courbe tremble,\nplus le risque est élevé.", RED)
    d.caption(s, "Dans toute la présentation, les Sharpe sont nets de frais de transaction et "
                 "calculés hors échantillon — sur des données que le modèle n’a jamais vues.", 478)

    # 05 — vocabulary 3 : diversification & Markowitz ────────────────────────
    s = d.slide("Le vocabulaire · 3/4", "Markowitz : ne pas mettre tous ses œufs au même endroit",
                sous="Prix Nobel 1990. L’idée : combiner des actifs qui ne bougent pas ensemble "
                     "réduit le risque sans sacrifier le rendement.")
    picture(s, FIG_REPORT / "frontiere_efficiente.png", 48, 118, 560, 350)
    card(s, 628, 126, 284, 156, "La frontière efficiente",
         "Chaque point gris est un portefeuille possible. La courbe bleue rassemble "
         "les meilleurs : à risque donné, aucun ne rapporte plus.", NAVY)
    card(s, 628, 296, 284, 172, "Ce dont le modèle a besoin",
         "Le rendement attendu de chaque actif, et la façon dont ils bougent "
         "les uns par rapport aux autres (la « covariance »).\n"
         "Ces deux entrées sont estimées — donc incertaines.", AMBER)

    # 06 — vocabulary 4 : backtest & overfitting ─────────────────────────────
    s = d.slide("Le vocabulaire · 4/4", "Backtest, fuite de données, surapprentissage",
                sous="Les trois mots qui décident si un résultat de finance quantitative "
                     "vaut quelque chose.")
    defs = [
        ("Backtest", "Rejouer l’histoire : on simule la stratégie sur le passé et on regarde "
                     "ce qu’elle aurait gagné.", NAVY),
        ("Fuite de données", "Utiliser, même par accident, une information postérieure à la "
                                "décision. Le backtest devient magnifique — et faux.", RED),
        ("Surapprentissage", "Le modèle apprend le bruit du passé au lieu de la règle. "
                             "Il excelle sur l’histoire, échoue sur l’avenir.", AMBER),
        ("Data snooping", "À force d’essayer des stratégies, on finit par en trouver une qui "
                          "brille par pur hasard.", TEAL),
    ]
    for i, (titre, corps, col) in enumerate(defs):
        x, y = 48 + (i % 2) * 444, 128 + (i // 2) * 176
        card(s, x, y, 420, 158, titre, corps, col, cs=12)
    d.caption(s, "Ces quatre notions structurent tout le protocole d’évaluation du projet.", 476)

    # 07 — the problem ───────────────────────────────────────────────────────
    s = d.slide("Le problème · contexte", "La question posée par le projet")
    panel(s, 48, 124, 864, 108, CALLOUT)
    text(s, 72, 146, 816, 70,
         "« Répartir un capital entre des actions de la Bourse de Casablanca et des ETF "
         "internationaux.\nLe Machine Learning fait-il mieux que la méthode classique — "
         "une fois payés les frais et vérifiée la statistique ? »",
         14.5, bold=True, color=INK, spacing=1.35)
    contraintes = [
        ("Long-only", "aucune vente à découvert :\nune contrainte de gestion réelle"),
        ("Plafond 25 %", "aucun actif ne peut dépasser\nun quart du portefeuille"),
        ("Frais réels", "chaque rebalancement coûte,\net le coût est déduit"),
        ("Mensuel", "la décision est reprise\nà chaque fin de mois"),
    ]
    text(s, 48, 256, 500, 20, "LES CONTRAINTES IMPOSÉES AU SYSTÈME", 10, bold=True, color=BLUE)
    for i, (titre, corps) in enumerate(contraintes):
        x = 48 + i * 220
        panel(s, x, 284, 200, 124)
        text(s, x + 18, 304, 170, 24, titre, 13.5, bold=True, color=NAVY)
        text(s, x + 18, 334, 170, 64, corps, 11, color=GREY, spacing=1.3)
    text(s, 48, 428, 864, 24, "TROIS TERRAINS D’ÉTUDE, ÉVALUÉS SÉPARÉMENT",
         10, bold=True, color=BLUE)
    terrains = [
        ("full_2021", "4 actions BVC + 5 ETF · en dirhams · 2021–2026"),
        ("etf_2017", "5 ETF internationaux · en dollars · 2004–2026"),
        ("global_2004", "10 ETF multi-classes · en dollars · 2004–2026"),
    ]
    for i, (nom, corps) in enumerate(terrains):
        x = 48 + i * 292
        text(s, x, 456, 270, 20, nom, 13, bold=True, color=TEAL)
        text(s, x, 478, 270, 34, corps, 11, color=GREY)

    # 08 — four failures ─────────────────────────────────────────────────────
    s = d.slide("Le problème · pourquoi c’est difficile",
                "Quatre façons dont l’optimisation classique échoue",
                sous="Chaque échec appelle une réponse technique précise — c’est le plan "
                     "de construction du système.")
    picture(s, FIG_REPORT / "quatre_problemes.png", 48, 124, 864, 300)
    panel(s, 48, 438, 864, 70, PANEL)
    text(s, 70, 452, 820, 46,
         "Traduction : l’optimiseur classique prend des estimations incertaines pour des "
         "certitudes, suppose que demain ressemble à hier,\net s’effondre exactement au "
         "moment où la diversification devrait protéger.",
         12, color=GREY, spacing=1.3)

    # 09 — correlations in crisis ────────────────────────────────────────────
    s = d.slide("Le problème · le moment critique",
                "En crise, la diversification disparaît quand on en a besoin",
                sous="Les actions se mettent à tomber ensemble ; seuls l’or et les obligations "
                     "s’en détachent.")
    picture(s, FIG_REPORT / "correlations_crise.png", 48, 118, 640, 350)
    kpi(s, 712, 150, 200, "0,79 → 0,94",
        "corrélation entre actions\nen mars 2020 : le portefeuille\nse comporte comme un seul actif",
        RED, size=26)
    card(s, 712, 296, 200, 172, "Conséquence",
         "Un modèle qui suppose\nune structure de risque fixe\nse trompe précisément\n"
         "les jours qui coûtent cher.", NAVY, cs=11)

    # 10 — the trap ──────────────────────────────────────────────────────────
    s = d.slide("Le problème · le piège central",
                "Le meilleur backtest d’une longue recherche est presque toujours une illusion")
    picture(s, FIG_FR / "snooping_240.png", 48, 112, 620, 340)
    q2t = e["q2"]["family_tests"]["primary_sharpe"]
    kpi(s, 700, 130, 212, str(q2t["n_candidates"]),
        "stratégies entièrement\névaluées, liste figée\navant tout calcul", NAVY)
    kpi(s, 700, 262, 212, fr(q2t["best_differential"], 3, signed=True),
        "l’écart du « champion » :\nséduisant sur le papier", RED)
    kpi(s, 700, 394, 212, "p = " + fr(q2t["reality_check_p_value"], 2),
        "le test de sélection multiple\nne retient pas ce résultat", TEAL, size=26)
    d.caption(s, "C’est le biais qui explique une grande partie des stratégies publiées "
                 "qui ne fonctionnent jamais en réel. Le projet le traite comme un adversaire, "
                 "pas comme un détail.", 470)

    # 11 — our answer ────────────────────────────────────────────────────────
    s = d.slide("Notre réponse", "Une chaîne complète, de la donnée brute à la décision publiée")
    picture(s, FIG_FR / "chaine_simple.png", 48, 130, 864, 260)
    for i, (titre, corps, col) in enumerate([
        ("Rien n’est saisi à la main", "Chaque chiffre affiché est régénéré\ndepuis un artefact versé.", TEAL),
        ("Rien ne regarde l’avenir", "Une décision n’utilise que\nl’information antérieure.", NAVY),
        ("Rien n’est publié sans test", "Une amélioration doit survivre\nà la correction statistique.", AMBER),
    ]):
        card(s, 48 + i * 292, 400, 280, 110, titre, corps, col, ts=12.5, cs=11)

    # 12 — data ──────────────────────────────────────────────────────────────
    s = d.slide("Notre réponse · les données",
                "Une donnée n’entre dans le système que si elle est traçable")
    etapes = [
        ("Bronze", "La donnée brute, conservée telle quelle. On ne réécrit jamais la source.", NAVY),
        ("Silver", "Calendriers alignés, conversion en dirhams au taux officiel Bank Al-Maghrib, "
                   "contrôles de cohérence.", TEAL),
        ("Gold", "Rendements, indicateurs et preuves : la seule couche que les modèles "
                 "et le rapport ont le droit de lire.", GREEN),
    ]
    for i, (titre, corps, col) in enumerate(etapes):
        card(s, 48 + i * 292, 124, 280, 168, titre, corps, col)
    text(s, 48, 316, 864, 20, "CE QUE CELA ÉVITE CONCRÈTEMENT", 10, bold=True, color=BLUE)
    pieges = [
        ("Le biais de dividende", "Les dividendes des actions BVC sont réintégrés à leur date "
                                  "de détachement : ignorer cela sous-estime la performance réelle."),
        ("Le piège de la devise", "Un ETF coté en dollars n’est pas comparable à une action "
                                     "cotée en dirhams. Tout est ramené à une seule monnaie, au taux officiel."),
    ]
    for i, (titre, corps) in enumerate(pieges):
        x = 48 + i * 444
        panel(s, x, 344, 420, 132)
        text(s, x + 20, 364, 380, 22, titre, 13, bold=True, color=RED)
        text(s, x + 20, 392, 380, 74, corps, 11.5, color=GREY, spacing=1.32)
    d.caption(s, "Ces deux corrections ont changé les résultats publiés. Elles ont été "
                 "appliquées malgré cela — c’est le prix d’un chiffre auquel on peut se fier.", 490)

    # 13 — models ────────────────────────────────────────────────────────────
    s = d.slide("Notre réponse · les modèles",
                "La complexité est ajoutée par paliers, jamais d’un bloc",
                sous="Chaque étage doit prouver qu’il améliore l’étage précédent, "
                     "sinon il reste expérimental.")
    picture(s, FIG_FR / "escalier_modeles.png", 48, 118, 864, 330)
    panel(s, 48, 456, 864, 56, CALLOUT)
    text(s, 70, 470, 820, 40,
         "Le cœur du système : un modèle de régimes (HMM) qui reconnaît, sans "
         "supervision, si le marché est calme ou sous tension — et change de posture.",
         12.5, bold=True, color=INK, spacing=1.25)

    # 14 — the walk-forward protocol ─────────────────────────────────────────
    s = d.slide("Notre réponse · l’épreuve",
                "Rejouer l’histoire sans jamais connaître la suite",
                sous="Le modèle est réentraîné à chaque décision, avec les seules données "
                     "disponibles ce jour-là.")
    picture(s, FIG_RAPPORT / "walkforward.png", 48, 112, 596, 348)
    card(s, 668, 124, 244, 122, "Purge et embargo",
         "On écarte les jours à cheval sur la frontière pour qu’aucune information "
         "ne fuite.", NAVY, cs=11)
    card(s, 668, 258, 244, 112, "Test final gelé",
         "La dernière période reste intouchée jusqu’à la comparaison finale.", TEAL, cs=11)
    card(s, 668, 382, 244, 112, "Frais déduits",
         "Chaque mouvement de portefeuille est facturé avant d’être compté.", AMBER, cs=11)

    # 15 — statistical guards ────────────────────────────────────────────────
    s = d.slide("Notre réponse · les garde-fous",
                "Trois verrous entre un bon chiffre et une affirmation")
    verrous = [
        ("1 · Comparaison appariée",
         "Les deux stratégies sont comparées jour par jour, sur exactement les mêmes "
         "dates. Un intervalle de confiance accompagne chaque écart.", NAVY),
        ("2 · Correction du data snooping",
         "White Reality Check et Hansen SPA corrigent le fait d’avoir essayé "
         f"{e['q2']['candidate_ledger']['executed_count']} configurations. "
         "La liste a été figée avant le premier calcul.", RED),
        ("3 · Comparateur pré-spécifié",
         "La référence à battre est choisie à l’avance et jamais réécrite après "
         "avoir vu les résultats.", TEAL),
    ]
    for i, (titre, corps, col) in enumerate(verrous):
        card(s, 48 + i * 292, 128, 280, 210, titre, corps, col, ts=13, cs=11.5)
    panel(s, 48, 356, 864, 96, CALLOUT)
    text(s, 70, 374, 816, 24, "Pourquoi cela compte pour un jury", 11, bold=True, color=AMBER)
    text(s, 70, 400, 816, 46,
         "Sans ces trois verrous, n’importe quelle équipe peut produire une courbe qui "
         "monte. Avec eux, un résultat annoncé devient vérifiable — et réfutable.",
         12.5, bold=True, color=INK, spacing=1.3)
    d.caption(s, "L’ensemble est rejouable : les données sont versionnées, le pipeline "
                 f"est décrit étape par étape, et {e['n_tests']} tests automatisés "
                 "gardent les invariants.", 470)

    # 16 — results : out-of-sample ───────────────────────────────────────────
    s = d.slide("Résultats", "Hors échantillon et net de frais, le système tient la distance",
                sous="À gauche l’univers mixte marocain, à droite vingt ans d’ETF "
                     "internationaux couvrant 2008, 2020 et 2022.")
    picture(s, FIG_REPORT / "courbes_equity.png", 48, 118, 864, 320)
    panel(s, 48, 448, 864, 62, PANEL)
    text(s, 70, 462, 820, 40,
         "Les stratégies se tiennent : aucune ne décroche, y compris à travers les "
         "crises. La question n’est donc pas « est-ce que ça marche »,\n"
         "mais « l’écart entre ces courbes est-il réel ou dans le bruit » — "
         "et c’est précisément ce que la chaîne sait trancher.",
         12, color=GREY, spacing=1.3)

    # 17 — finding 1 : regime detection ──────────────────────────────────────
    c = e["crisis"]
    s = d.slide("Résultat · 1", "Le modèle repère les crises — sans qu’on les lui apprenne",
                sous="Les cinq fenêtres de crise sont des dates externes, fixées avant "
                     "d’examiner les résultats.")
    picture(s, FIG_FR / "detection_crises_fr.png", 48, 112, 616, 340)
    kpi(s, 696, 132, 216, f"{e['n_crises']} / {e['n_crises']}",
        "crises détectées\npar un modèle non supervisé", TEAL)
    kpi(s, 696, 258, 216, pct(c["bear_rate_in_crisis"] * 100, 1, signed=False),
        "de rééquilibrages classés\n« tension » pendant les crises\ncontre "
        + pct(c["bear_rate_outside"] * 100, 1, signed=False) + " hors crise", RED, size=26)
    kpi(s, 696, 398, 216, "p = " + fr(c["significance"]["sign_test_p_conservative"], 3),
        "test des signes, lecture prudente", NAVY, size=24)
    d.caption(s, "Un diagnostic exploitable : le système sait dire dans quel état de marché "
                 "il pense se trouver, et cette information est lisible par un gérant.", 476)

    # 18 — finding 2 : the cap ───────────────────────────────────────────────
    cap25, cap_no = e["cap"]["0.25"], e["cap"]["1.00"]
    gain = (cap25[1] / cap_no[1] - 1) * 100
    s = d.slide("Résultat · 2", "La contrainte de gestion vaut plus que la sophistication",
                sous="Même moteur, mêmes données, mêmes frais : seule la règle "
                     "« aucun actif au-delà de 25 % » change.")
    picture(s, FIG_REPORT / "plafond.png", 48, 118, 600, 330)
    kpi(s, 680, 140, 232, fr(cap25[1]), "Sharpe net avec le plafond de 25 %", TEAL)
    kpi(s, 680, 262, 232, fr(cap_no[1]), "Sharpe net sans plafond", RED)
    kpi(s, 680, 384, 232, pct(gain, 1), "d’écart : plus que tout modèle testé", NAVY, size=26)
    d.caption(s, "Lecture métier : une règle de gouvernance simple absorbe l’erreur "
                 "d’estimation mieux qu’un modèle complexe. C’est un résultat "
                 "directement actionnable.", 470)

    # 19 — finding 3 : what the chain refuses ────────────────────────────────
    s = d.slide("Résultat · 3", "La chaîne sait refuser un résultat flatteur")
    panel(s, 48, 124, 420, 200, PANEL)
    text(s, 70, 144, 380, 22, "CE QU’UN BACKTEST AURAIT PERMIS D’ÉCRIRE", 10, bold=True, color=RED)
    text(s, 70, 176, 380, 54,
         "« Notre modèle XGBoost améliore le Sharpe de "
         + fr(q2t["best_differential"], 3, signed=True) + " »", 15, bold=True, color=INK,
         spacing=1.3)
    text(s, 70, 244, 380, 64,
         "Vrai au sens arithmétique : c’est bien le meilleur des "
         f"{q2t['n_candidates']} candidats évalués.", 11.5, color=GREY, spacing=1.32)
    panel(s, 492, 124, 420, 200, CALLOUT)
    text(s, 514, 144, 380, 22, "CE QUE LA CHAÎNE RÉPOND", 10, bold=True, color=AMBER)
    text(s, 514, 176, 380, 54,
         "« C’est le maximum d’une recherche.\nLes tests ne le retiennent pas. »",
         15, bold=True, color=INK, spacing=1.3)
    text(s, 514, 244, 380, 64,
         "White Reality Check p = " + fr(q2t["reality_check_p_value"], 3)
         + " · Hansen SPA p = " + fr(q2t["spa_p_value"], 3)
         + f" · {q2t['n_candidates_beating_benchmark']} candidats devançaient "
           "la référence en apparence.", 11.5, color=GREY, spacing=1.32)
    panel(s, 48, 344, 864, 100, PANEL)
    text(s, 70, 362, 816, 22, "POURQUOI C’EST LA CONTRIBUTION LA PLUS UTILE", 10,
         bold=True, color=BLUE)
    text(s, 70, 390, 816, 46,
         "Un prototype qui produit des résultats a peu de valeur s’il ne sait pas les "
         "invalider. Celui-ci a été conçu pour que\nla déception soit détectable "
         "— c’est exactement ce qu’une direction des risques exige d’un modèle.",
         12.5, bold=True, color=INK, spacing=1.3)
    d.caption(s, "Même logique appliquée à nos propres corrections : dividendes, "
                 "historique et conversion en dirhams ont été corrigés même lorsque "
                 "cela dégradait le résultat affiché.", 462)

    # 20 — what is established, and what is not ──────────────────────────────
    s = d.slide("Lecture d’ensemble", "Ce que nous établissons — et ce que nous n’établissons pas")
    panel(s, 48, 124, 480, 326, PANEL)
    text(s, 72, 146, 430, 22, "ÉTABLI", 10, bold=True, color=TEAL)
    etabli = [
        "Détection non supervisée des cinq crises de la période.",
        "Le plafond de 25 % améliore le Sharpe net de "
        + pct(gain, 1) + " sur les ETF.",
        "Une chaîne reproductible de bout en bout, versionnée et testée.",
        "Un dispositif capable de refuser un faux positif de sélection.",
        "Une explicabilité exacte : chaque poids se remonte jusqu’à sa cause.",
    ]
    y = 178
    for item in etabli:
        text(s, 72, y, 18, 20, "✓", 13, bold=True, color=TEAL)
        text(s, 96, y, 410, 44, item, 11.5, color=INK, spacing=1.3)
        y += 54
    panel(s, 552, 124, 360, 326, CALLOUT)
    text(s, 576, 146, 320, 22, "NON ÉTABLI", 10, bold=True, color=AMBER)
    text(s, 576, 178, 320, 130,
         "Sur nos univers et nos fenêtres, aucune couche ML n’établit de surperformance "
         "statistiquement significative face à son comparateur pré-spécifié.",
         12.5, bold=True, color=INK, spacing=1.32)
    text(s, 576, 296, 320, 140,
         "C’est une réponse, pas un échec : la question posée était « peut-on le "
         "savoir ».\n\nUne équipe qui aurait annoncé l’inverse n’aurait pas pu le démontrer.",
         11.5, color=GREY, spacing=1.35)
    d.caption(s, "Les deux colonnes viennent du même dispositif. C’est ce qui rend "
                 "la première crédible.", 466)

    # 21 — the delivered system ──────────────────────────────────────────────
    s = d.slide("Le livrable", "Un prototype réellement utilisable, pas un notebook")
    picture(s, FIG_FR / "dashboard_page1_haut.png", 48, 118, 420, 248)
    picture(s, FIG_FR / "api_swagger_haut.png", 492, 118, 420, 248)
    text(s, 48, 376, 420, 20, "Tableau de bord Streamlit", 12, bold=True, color=INK)
    text(s, 48, 398, 420, 20, "Deux vues : résultats de recherche et explorateur de stratégies",
         10.5, color=GREY)
    text(s, 492, 376, 420, 20, "API REST FastAPI, en lecture seule", 12, bold=True, color=INK)
    text(s, 492, 398, 420, 20, "Contrat de devise explicite, documentation générée", 10.5, color=GREY)
    livrables = [
        (str(e["n_tests"]), "tests automatisés"),
        (f"{fb['total_fits']:,}".replace(",", " "), "ajustements de modèles tracés"),
        ("DVC + CI", "données versionnées,\nportes de release"),
        ("Docker", "environnement reproductible"),
    ]
    for i, (v, c_) in enumerate(livrables):
        kpi(s, 48 + i * 222, 428, 200, v, c_, NAVY, size=24)

    # 22 — perspectives ──────────────────────────────────────────────────────
    s = d.slide("Perspectives", "Ce qui transformerait le prototype en dispositif validé")
    suites = [
        ("01", "Élargir l’univers marocain",
         "Industrialiser le panel profond 2005–2024 : réconciliation des sources, "
         "dividendes, opérations sur titres, licence de redistribution.", NAVY),
        ("02", "Validation indépendante",
         "Séparer l’équipe qui développe de celle qui valide, comme l’exige une "
         "gouvernance de modèles bancaire.", TEAL),
        ("03", "Couverture du risque de change",
         "Modéliser un contrat de couverture USD/MAD et son coût de roulement, "
         "aujourd’hui absents.", AMBER),
        ("04", "Passage à l’échelle",
         "Plus d’actifs, plus de classes : c’est là qu’un optimiseur a le plus "
         "de marge pour exprimer une vue.", RED),
    ]
    for i, (num, titre, corps, col) in enumerate(suites):
        x, y = 48 + (i % 2) * 444, 128 + (i // 2) * 180
        panel(s, x, y, 420, 162)
        text(s, x + 20, y + 20, 40, 28, num, 18, bold=True, color=col)
        text(s, x + 64, y + 22, 336, 24, titre, 13.5, bold=True, color=INK)
        text(s, x + 64, y + 52, 336, 92, corps, 11.5, color=GREY, spacing=1.32)

    # 23 — conclusion ────────────────────────────────────────────────────────
    s = d.bare()
    panel(s, 0, 0, W, H, WHITE)
    rule(s, 48, 120, 96, BLUE, 4)
    text(s, 48, 76, 600, 18, "CONCLUSION", 10.5, bold=True, color=BLUE)
    text(s, 48, 152, 800, 110,
         "Nous avons construit la chaîne\nqui permet de savoir.", 36, bold=True, color=INK,
         spacing=1.15)
    text(s, 48, 280, 780, 56,
         "Un système complet d’allocation, évalué sans tricher, capable de reconnaître "
         "les crises,\nde montrer ce qui améliore réellement un portefeuille — et de "
         "refuser ce qui n’est que du hasard.",
         14, color=GREY, spacing=1.4)
    piliers = [
        ("Rigueur", "évaluation temporelle stricte,\ncorrection de la sélection multiple", NAVY),
        ("Traçabilité", "données versionnées,\nsurfaces régénérées", TEAL),
        ("Transparence", "limites explicites,\nexplicabilité exacte", AMBER),
    ]
    for i, (titre, corps, col) in enumerate(piliers):
        x = 48 + i * 292
        rule(s, x, 380, 40, col)
        text(s, x, 398, 264, 24, titre, 14, bold=True, color=INK)
        text(s, x, 424, 264, 50, corps, 11.5, color=GREY, spacing=1.32)
    text(s, 48, 496, 600, 20, "Merci de votre attention — questions bienvenues.",
         12.5, bold=True, color=BLUE)

    # ── annexes ─────────────────────────────────────────────────────────────

    s = d.slide("Annexe · A", "Les chiffres exacts, par univers")
    rows = [
        ("Univers", "Comparateur classique", "Système à régimes", "Écart observé"),
        ("full_2021 · MAD", f"{full['best_classical']['name']} — " + fr(full["best_classical"]["sharpe_net"]),
         fr(full["regime"]["sharpe_net"]), pct(full["point_difference_pct"], 2)),
        ("etf_2017 · USD", f"{etf['best_classical']['name']} — " + fr(etf["best_classical"]["sharpe_net"]),
         fr(etf["regime"]["sharpe_net"]), pct(etf["point_difference_pct"], 2)),
    ]
    q1 = e["q1"]
    y = 130
    for i, row in enumerate(rows):
        if i == 0:
            for j, cell in enumerate(row):
                text(s, 48 + j * 224, y, 214, 20, cell.upper(), 9.5, bold=True, color=BLUE)
            y += 26
            continue
        panel(s, 48, y - 8, 864, 44, PANEL if i % 2 else WHITE)
        for j, cell in enumerate(row):
            text(s, 60 + j * 224, y + 2, 214, 24, cell, 12,
                 bold=(j == 0), color=INK if j < 3 else RED)
        y += 52
    text(s, 48, 268, 864, 20, "EXTENSION DE RECHERCHE · global_2004 (10 ETF, 2004–2026)",
         9.5, bold=True, color=BLUE)
    text(s, 48, 294, 864, 100,
         "Univers pré-enregistré et horodaté avant toute ingestion de données, "
         "construit pour que l’optimiseur puisse réellement exprimer une vue "
         f"({e['ready']['allocation_freedom']['min_variance_lw']['distinct_allocations']} "
         f"allocations distinctes sur {e['ready']['allocation_freedom']['n_rebalances']} "
         "rééquilibrages, contre 1 sur etf_2017).\n"
         "Résultat Q1 : " + fr(q1["candidate"]["net_sharpe"])
         + " pour le système à régimes contre " + fr(q1["comparator"]["net_sharpe"])
         + " pour le comparateur classique.",
         12, color=GREY, spacing=1.35)
    text(s, 48, 408, 864, 80,
         f["statements"]["not_comparable"].replace("`", ""), 11.5, color=GREY, spacing=1.35)

    s = d.slide("Annexe · B", "Intégrité des modèles et traçabilité")
    kpi(s, 48, 136, 260, f"{fb['total_fallbacks']} / {fb['total_fits']:,}".replace(",", " "),
        "ajustements ayant utilisé un estimateur de repli — tracés, comptés et publiés",
        NAVY, size=28)
    kpi(s, 340, 136, 260, str(fb["n_strategies"]),
        "stratégies évaluées sur " + str(fb["n_dates"]) + " dates de rééquilibrage",
        TEAL, size=28)
    kpi(s, 632, 136, 280, str(e["n_tests"]),
        "tests automatisés, dont des tests de cohérence entre artefacts",
        AMBER, size=28)
    panel(s, 48, 268, 864, 118, CALLOUT)
    text(s, 70, 288, 816, 22, "CE QUE CELA SIGNIFIE", 10, bold=True, color=AMBER)
    text(s, 70, 316, 816, 60,
         "Lorsqu’un estimateur ne converge pas, le système bascule sur une solution "
         "de repli — et le dit. Les séries concernées sont\nsignalées comme hybrides "
         "plutôt que présentées sous l’étiquette du modèle qu’elles ne sont pas.",
         12.5, color=INK, spacing=1.3)
    text(s, 48, 406, 864, 90,
         "Aucune stratégie n’est recommandée, déployée, ni présentée comme une valeur "
         "ajoutée établie. Ce livrable est un prototype de recherche académique : "
         "ni conseil financier, ni recommandation client, ni exécution d’ordres.",
         12, color=GREY, spacing=1.35)

    d.attach_notes(NOTES)
    d.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path.relative_to(ROOT)}")
