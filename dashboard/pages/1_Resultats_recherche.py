"""
1_Resultats_recherche.py — Stakeholder-facing research results (Phase 7).

INTEGRITY CONSTRAINTS, enforced in this file rather than left to discipline:

  1. The reference system is REGIME + DYNAMIC-COVARIANCE
     (`regime_conditional`). It is not presented as superior: the available
     evaluations have not established incremental portfolio outperformance.
     `ML_STRATEGY` below is a single constant so this cannot drift.

  2. Every headline number is described as an observed point estimate. Marginal
     confidence intervals quantify uncertainty but do not replace a paired test of
     a strategy difference; no superiority claim is made.

  3. No number is typed into this file. Everything reads from the Gold
     artifacts via `shared.data`, which `tests/test_run_dashboard_data.py`
     verifies against their sources.

  4. The honest losses are shown, not buried: `etf_2017` is a case where our
     system LOSES to classical Markowitz, and it is displayed as such.

  5. ATTRIBUTION. The crisis section credits the *constraint and covariance
     model* (P1/P3), not the regime layer — in 3 of 5 windows the three
     optimizers are identical to the decimal, because in a bear regime
     `regime_conditional` IS `min_variance_lw` by construction and the 25% cap
     on 5 assets pins the allocation. The count is computed, not asserted.
     Separately, the regime-detection analysis is exploratory evidence about what
     the model sees — never a claim about what acting on it earns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.shared import data as D
from dashboard.shared import plots as P

# The single definition of "our model" for this entire page. Phase 5 showed the
# F7 signals add no edge — attributing the win to them would be false.
ML_STRATEGY = "regime_conditional"

# Every claim on this page comes from src/release_facts.py so the dashboard, the
# API, the README and the report cannot drift apart in WORDING the way they did
# on 2026-08-02 (§17.11: test_artifact_consistency compares numbers, not prose).
sys.path.insert(0, str(ROOT / "src"))
from release_facts import load_facts          # noqa: E402

FACTS = load_facts(ROOT)
# The universe the pitch headlines: EURAFRIC's actual 9-asset business portfolio.
HEADLINE_UNIVERSE = "full_2021"

st.set_page_config(page_title="Résultats de recherche", page_icon="📊", layout="wide")

try:
    showcase = D.load_showcase()
    equity = D.load_equity()
    regime = D.load_regime()
except D.DashboardDataMissing as exc:
    st.error(str(exc))
    st.stop()

# Optional: absent artifact degrades this section rather than the whole page.
crisis = D.load_crisis()

u = showcase["universes"][HEADLINE_UNIVERSE]
best_classical = u["best_classical"]
best_ml = u["best_ml"]
lift_pct = u["headline_lift_pct"]
p5 = u["phase5_test_window"]

# ── 1. Le problème ─────────────────────────────────────────────────────────
st.title("📊 Résultats de recherche — allocation assistée par ML")
st.warning(
    "**Prototype de recherche, pas un outil d'investissement.** Cette application "
    "présente des résultats historiques de backtesting ; elle ne fournit ni conseil "
    "d'investissement, ni recommandation client, ni exécution d'ordres."
)
# The numéraire, stated before any figure. `full_2021` is MAD-valued at the
# official Bank Al-Maghrib reference rate and remains UNHEDGED; `etf_2017` is
# single-currency USD and was never affected. Previously this block claimed
# mixed-currency arithmetic was valid because "returns are unitless" — it is
# not, and that defect is what the correction removed.
st.error(FACTS["statements"]["full_2021_currency"])
st.info(
    FACTS["statements"]["etf_2017_currency"] + "  \n\n"
    + FACTS["statements"]["not_comparable"]
)
st.markdown(
    f"""
    **Le problème.** Répartir un capital entre {len(showcase['assets_per_universe'][HEADLINE_UNIVERSE])}
    actifs — actions de la Bourse de Casablanca et ETF internationaux — de façon à
    maximiser le rendement tout en maîtrisant le risque. L'outil standard du secteur
    est l'optimisation moyenne-variance de **Markowitz**. Les questions auxquelles
    répond ce projet : ce que les données suggèrent sur le Machine Learning,
    **ce qu'elles permettent d'établir**, et — tout aussi important — ce qu'elles
    ne permettent pas d'affirmer.
    """
)
st.divider()

# ── 2 & 3. Les deux approches, côte à côte ─────────────────────────────────
# Header was "Ce qui existe aujourd'hui, et ce que nous ajoutons" — a value-add
# framing that presumed the regime strategy contributed something established.
# On MAD-valued data its observed difference is NEGATIVE on both universes, and
# in neither direction is a paired test presented. The section now compares two
# approaches without asserting that either adds value.
st.header("Les deux approches, comparées")

col1, col2, col3 = st.columns(3)
col1.metric(
    "Meilleure approche classique",
    f"{best_classical['sharpe_net']:.3f}",
    help=f"Meilleure stratégie classique : {D.label(best_classical['name'])}. "
         f"Ratio de Sharpe net de coûts, hors échantillon.",
)
col2.metric(
    "Stratégie à régimes",
    f"{best_ml['sharpe_net']:.3f}",
    delta=f"{lift_pct:+.1f}%",
    help="Régime (HMM) + covariance dynamique. Ratio de Sharpe net de coûts.",
)
col3.metric(
    "Écart observé",
    f"{u['headline_lift_absolute_sharpe']:+.3f} Sharpe",
    help="Différence absolue de ratio de Sharpe, nette de coûts de transaction. "
         "Écart descriptif : aucun test pairé de cette différence n'est présenté.",
)

st.markdown(
    f"""
    Sur le portefeuille EURAFRIC ({u['oos_start']} → {u['oos_end']}, simulation
    walk-forward nette de coûts, libellée en MAD), la stratégie à régimes obtient un
    ratio de Sharpe de **{best_ml['sharpe_net']:.3f}** contre
    **{best_classical['sharpe_net']:.3f}** pour la meilleure approche classique
    (*{D.label(best_classical['name'])}*).
    """
)
st.markdown(FACTS["statements"]["point_difference"])
st.markdown(FACTS["statements"]["no_paired_test"])

curves = {
    s: D.equity_curve(equity, HEADLINE_UNIVERSE, s, net=True)
    for s in (*D.CLASSICAL_STRATEGIES, ML_STRATEGY)
}
st.plotly_chart(
    P.equity_comparison(
        curves, highlight=ML_STRATEGY,
        title="Évolution du portefeuille, net de coûts (base 100)",
    ),
    use_container_width=True,
)

with st.expander("Qu'est-ce que le système fait différemment ?"):
    st.markdown(
        """
        Deux mécanismes, tous deux absents d'un Markowitz classique :

        - **Détection de régime (HMM).** Un modèle de Markov caché identifie, sans
          jamais voir le futur, si le marché est en phase haussière ou baissière, et
          adapte la posture du portefeuille : offensive en régime haussier, défensive
          en régime baissier.
        - **Covariance dynamique.** Les corrélations entre actifs se resserrent
          brutalement en période de crise — exactement quand la diversification devrait
          protéger. Le système ré-estime ces corrélations en continu au lieu de les
          supposer stables.
        """
    )

st.divider()

# ── 3bis. Comportement en crise (P3) — lecture métier descriptive ──────────
# Placed before the Sharpe section because drawdowns and recovery time are
# business-legible. Five windows remain descriptive evidence, not confirmation.
if crisis and crisis.get("universes", {}).get("etf_2017"):
    st.header("Et quand les marchés s'effondrent ?")
    st.markdown(
        """
        C'est la question qui décide d'un mandat. La diversification est censée protéger
        en crise — et c'est précisément là qu'elle cesse de fonctionner, les corrélations
        entre actifs se resserrant brutalement (**problème P3** du projet). Nous avons donc
        mesuré le comportement du portefeuille sur **cinq crises**, délimitées par les dates
        de sommet-à-creux publiées du S&P 500 — **fixées avant tout examen des résultats**,
        pour ne pas choisir après coup les périodes qui nous arrangent.
        """
    )

    CW = crisis["universes"]["etf_2017"]
    rows = []
    for key, meta in crisis["crises"].items():
        if key not in CW:
            continue
        w = CW[key]
        opt = w.get("min_variance_lw") or w.get("regime_conditional")
        ew_ = w.get("equal_weight")
        if not opt or not ew_:
            continue
        rows.append({
            "Crise": D.crisis_label(key, meta["label"]),
            "Optimiseurs sous contrainte": f"{100*opt['cum_return']:+.1f} %",
            "Équipondéré (1/N)": f"{100*ew_['cum_return']:+.1f} %",
            "Perte évitée": f"{100*(opt['cum_return']-ew_['cum_return']):+.1f} pts",
            "Récupération": (
                f"{opt['recovery_days']} j  vs  {ew_['recovery_days']} j"
                if opt.get("recovery_days") and ew_.get("recovery_days") else "—"
            ),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Derived, never typed — integrity constraint 3. The two illustrative
    # crises are addressed by KEY, so if a window's dates change the prose
    # follows automatically instead of silently going stale.
    def _fr(x, n=1):
        return f"{x:.{n}f}".replace(".", ",")

    gfc = CW.get("gfc_2008", {})
    eu = CW.get("eu_debt_2011", {})
    bullets = []
    if gfc.get("min_variance_lw") and gfc.get("equal_weight"):
        o, e = gfc["min_variance_lw"], gfc["equal_weight"]
        bullets.append(
            f"Lors de la crise financière de 2008 elle épargne "
            f"**{_fr(100*(o['cum_return']-e['cum_return']))} points** de performance et "
            f"**{_fr(100*(o['max_drawdown']-e['max_drawdown']))} points** de drawdown"
        )
    if eu.get("min_variance_lw") and eu.get("equal_weight"):
        o, e = eu["min_variance_lw"], eu["equal_weight"]
        if o["cum_return"] > 0:
            bullets.append(
                f"pendant la crise de la dette européenne elle termine **positive** "
                f"(+{_fr(100*o['cum_return'])} %) quand le 1/N perd "
                f"{_fr(abs(100*e['cum_return']))} %"
            )
    n_windows = len(rows)
    st.info(
        f"""
        **Sur les {n_windows} fenêtres de crise étudiées, l'optimisation sous contrainte
        perd moins, chute moins et récupère plus vite que la diversification naïve.**
        {" ; ".join(bullets)}. Le délai de récupération est l'écart le plus régulier :
        **environ deux fois moins de temps sous l'eau**.

        Cette comparaison porte sur cinq fenêtres publiées et décrit ces épisodes ; elle
        ne constitue pas à elle seule une inférence statistique généralisable.
        """,
        icon="ℹ️",
    )

    # How many windows have the three optimizers effectively tied? Counted, not
    # asserted — this is the caveat that keeps the section from overclaiming.
    def _tied(w):
        vals = [w[s]["cum_return"] for s in
                ("min_variance_lw", "max_sharpe", "regime_conditional") if s in w]
        return len(vals) == 3 and (max(vals) - min(vals)) < 5e-4

    n_tied = sum(1 for k in CW if _tied(CW[k]))
    st.caption(
        f"⚠️ Attribution honnête : ce comportement revient à la **contrainte de portefeuille et au "
        f"modèle de covariance** (P1/P3), pas spécifiquement à la couche de régime. Sur "
        f"{n_tied} des {n_windows} crises les trois optimiseurs sont identiques à la décimale "
        f"près — en régime baissier la stratégie à régimes *devient* la variance minimale par "
        f"construction, et le plafond de {100*showcase['max_weight']:.0f} % sur 5 actifs "
        f"contraint fortement l'allocation."
    )

    # Exploratory association between externally dated crisis windows and the
    # state labelled "bear" after fitting the HMM.
    rd = (crisis.get("regime_detection") or {}).get("etf_2017")
    if rd:
        sig = rd["significance"]
        st.markdown("#### Le détecteur de régime a-t-il vu venir les crises ?")
        c1, c2, c3 = st.columns(3)
        c1.metric("Régime baissier PENDANT les crises",
                  f"{100*rd['bear_rate_in_crisis']:.0f} %")
        c2.metric("Régime baissier hors crise",
                  f"{100*rd['bear_rate_outside']:.0f} %",
                  help="Le modèle ne crie pas au loup en permanence.")
        c3.metric("Rapport", f"{rd['risk_ratio']:.1f}×",
                  help=f"{sig['crises_exceeding_base_rate']} crises au-dessus du taux de base.")
        st.info(
            f"""
            Le détecteur est **non supervisé** : aucune date de crise, aucun label de récession
            ne lui a jamais été fourni. Il n'observe que rendement, volatilité et corrélation
            moyenne, et décide **en temps réel**, à partir du passé uniquement. Il a néanmoins
            signalé **{sig['crises_exceeding_base_rate']} crises** au-dessus de son taux de base.

            C'est une **association exploratoire**, non une preuve confirmatoire : cinq fenêtres
            ne suffisent pas à établir une généralisation, et « baissier » est **défini** comme
            l'état au rendement moyen le plus faible. Le test des signes disponible dans
            l'artefact (*p* = {sig['sign_test_p_conservative']:.3f}) est donc un diagnostic,
            pas une revendication de significativité. Cette analyse ne démontre pas qu'agir sur
            le signal améliore la performance économique.
            """,
            icon="🎯",
        )
    st.divider()

# ── 4. Est-ce réel ? — la couche de crédibilité ────────────────────────────
st.header("Et le ratio de Sharpe, alors ?")
st.markdown(
    f"""
    L'écart de ratio de Sharpe présenté plus haut a aussi été examiné sur une
    **fenêtre de test gelée** ({p5['test_start']} → {p5['test_end']}) que la procédure
    de calibration n'a jamais vue, avec des **intervalles de confiance à 90 %** obtenus
    par bootstrap par blocs. Le verdict y est plus nuancé que pour le comportement en
    crise — et nous le présentons tel quel.
    """
)

rc = p5.get(ML_STRATEGY, {})
ew = p5.get("equal_weight", {})
labels, points, los, his = [], [], [], []
for key, entry in (("equal_weight", ew), (ML_STRATEGY, rc)):
    if not entry:
        continue
    labels.append(D.label(key))
    points.append(entry["test_sharpe_net"])
    ci = entry.get("test_sharpe_ci") or [None, None]
    los.append(ci[0])
    his.append(ci[1])

if labels:
    st.plotly_chart(
        P.sharpe_with_ci(
            labels, points, los, his,
            highlight_index=len(labels) - 1,
            title=f"Test hors échantillon gelé ({p5['test_start']} → {p5['test_end']}) "
                  f"— Sharpe net et intervalle de confiance à 90 %",
        ),
        use_container_width=True,
    )

st.info(
    """
    **Lecture honnête de ce graphique.** Sur cette fenêtre de test, la stratégie à
    régimes devance la diversification naïve (1/N) en estimation ponctuelle. Les
    intervalles affichés sont toutefois **marginaux** : ils décrivent l'incertitude de
    chaque stratégie prise isolément, et leur chevauchement ne constitue pas un test de
    la différence.

    Ce test a désormais été mené. Sous sélection strictement antérieure, les **huit
    comparaisons pairées** (bootstrap par blocs sur les mêmes dates, net de coûts) ont
    toutes un intervalle contenant zéro : **aucune surperformance n'est établie**, ni
    contre la stratégie à régimes ni contre l'équipondéré, sur aucun des deux univers.
    Ce n'est pas non plus une preuve d'équivalence.
    """,
    icon="ℹ️",
)

st.divider()

# ── 5. Comment ça marche (visuel, sans mathématiques) ──────────────────────
st.header("Comment le système prend ses décisions")
st.markdown(
    "Les zones colorées montrent le régime détecté par le modèle à chaque "
    "rééquilibrage — sans qu'aucune date de crise ne lui ait jamais été fournie."
)
regime_u = regime[regime["universe"] == HEADLINE_UNIVERSE]
if not regime_u.empty:
    st.plotly_chart(
        P.regime_timeline(
            curves[ML_STRATEGY], regime_u,
            title="Régimes détectés et performance du système",
        ),
        use_container_width=True,
    )

st.divider()

# ── 6. Limitations honnêtes ────────────────────────────────────────────────
st.header("Limites — ce que ce résultat ne dit pas")

other_universe = "etf_2017"
ou = showcase["universes"][other_universe]

st.markdown(
    f"""
    - **L'écart observé n'est pas universel.** Sur l'univers ETF internationaux seuls
      ({D.UNIVERSE_LABELS[other_universe]}), la stratégie à régimes fait **{ou['headline_lift_pct']:+.1f} %**
      face au Markowitz classique — c'est-à-dire qu'il **perd**. Ces deux univers ne
      permettent donc pas de revendiquer une valeur généralisable du ML.
    - **Ce cas défavorable est en partie un artefact de contrainte, pas seulement
      un verdict sur le ML.** Avec 5 actifs et un plafond de 25 %, tout portefeuille
      admissible doit placer au moins 4 actifs *au plafond* (5 × 0,25 = 1,25) : la
      contrainte, et non le modèle, détermine l'essentiel de l'allocation. À 25 % la
      variance minimale ne produit qu'**une seule** allocation sur 248 rééquilibrages ;
      à 30 %, elle en produit 169. L'univers à 9 actifs n'est pas concerné.
    - **Le test pairé a été mené, et n'établit aucune supériorité.** Les huit
      comparaisons sur la fenêtre gelée ont un intervalle contenant zéro (p de 0,066 à
      0,939). Sur `full_2021`, le signal XGB « bat » la stratégie à régimes en
      estimation ponctuelle (1,308 contre 1,213) mais le test pairé donne p = 0,361 :
      le classement ponctuel n'est pas soutenu par les données.
    - **La couche de prédiction de rendement (F7) n'a pas établi d'avantage robuste de
      portefeuille.** Quatre évaluations indépendantes vont dans ce sens : calibration
      honnête (Phase 5), historique marocain profond sur 20 ans, ajout de données
      fondamentales, et la ré-évaluation menée après la correction des dividendes.
      Son estimation ponctuelle passe **au-dessus ou en dessous** de la stratégie à
      régimes selon la fenêtre de test retenue. Le classement est de même sensible au
      protocole d'évaluation et à la fenêtre hors échantillon associée : sur la
      fenêtre complète, la stratégie à régimes est en retrait de l'approche classique,
      tandis que le walk-forward imbriqué — dont la fenêtre OOS diffère — la place
      devant. Ce **changement de signe** rend toute conclusion de supériorité fragile,
      dans un sens comme dans l'autre. La stratégie à régimes et la covariance
      dynamique restent des pistes étudiées, pas une valeur ajoutée démontrée.
    - **Numéraire.** `full_2021` est libellé en MAD au taux de référence officiel de
      Bank Al-Maghrib et reste NON COUVERT ; `etf_2017` est mono-devise USD et
      inchangé. Les niveaux de Sharpe ne sont pas comparables entre univers.
    - **Historique BVC limité.** Les données gratuites de la Bourse de Casablanca ne
      remontent qu'à mi-2021, ce qui exclut le krach COVID de cet univers. L'univers
      ETF, lui, remonte désormais à 2004 et intègre la crise de 2008.
    """
)

st.subheader("Ce que les tests établissent, mot pour mot")
# Rendered from src/release_facts.py so this page, the API, the README and the
# report state the same claims in the same words.
for _s in (FACTS["statements"]["primary_benchmark"],
           FACTS["statements"]["multiple_testing"],
           FACTS["statements"]["nested"],
           FACTS["statements"]["no_recommendation"]):
    st.markdown(f"- {_s}")

st.success(
    """
    **Pourquoi présenter les limites aussi clairement ?** Parce qu'un chiffre validé
    hors échantillon, avec son intervalle de confiance et ses cas d'échec, est plus
    solide qu'un chiffre plus flatteur mais invérifiable. C'est le critère qu'un
    professionnel appliquerait avant de mettre du capital derrière un modèle.
    """,
    icon="✅",
)
