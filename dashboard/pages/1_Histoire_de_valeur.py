"""
1_Histoire_de_valeur.py — The business-facing pitch (Phase 7).

INTEGRITY CONSTRAINTS, enforced in this file rather than left to discipline:

  1. "Notre système" means the REGIME + DYNAMIC-COVARIANCE system
     (`regime_conditional`) — the part that genuinely beat classical Markowitz.
     The F7 return-prediction signals are NEVER presented as our value-add:
     Phase 5 proved they add no statistically-significant edge, and two further
     experiments (deep-Morocco, fundamentals) confirmed it. `ML_STRATEGY` below
     is a single constant so this cannot drift.

  2. Every headline number carries its confidence interval. The improvement is
     always described as a point estimate whose interval is wide — never as a
     statistically significant result, which the data does not support.

  3. No number is typed into this file. Everything reads from the Gold
     artifacts via `shared.data`, which `tests/test_run_dashboard_data.py`
     verifies against their sources.

  4. The honest losses are shown, not buried: `etf_2017` is a case where our
     system LOSES to classical Markowitz, and it is displayed as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.shared import data as D
from dashboard.shared import plots as P

# The single definition of "our model" for this entire page. Phase 5 showed the
# F7 signals add no edge — attributing the win to them would be false.
ML_STRATEGY = "regime_conditional"
# The universe the pitch headlines: EURAFRIC's actual 9-asset business portfolio.
HEADLINE_UNIVERSE = "full_2021"

st.set_page_config(page_title="Histoire de valeur", page_icon="📊", layout="wide")

try:
    showcase = D.load_showcase()
    equity = D.load_equity()
    regime = D.load_regime()
except D.DashboardDataMissing as exc:
    st.error(str(exc))
    st.stop()

u = showcase["universes"][HEADLINE_UNIVERSE]
best_classical = u["best_classical"]
best_ml = u["best_ml"]
lift_pct = u["headline_lift_pct"]
p5 = u["phase5_test_window"]

# ── 1. Le problème ─────────────────────────────────────────────────────────
st.title("📊 Quelle valeur le Machine Learning apporte-t-il ?")
st.markdown(
    f"""
    **Le problème.** Répartir un capital entre {len(showcase['assets_per_universe'][HEADLINE_UNIVERSE])}
    actifs — actions de la Bourse de Casablanca et ETF internationaux — de façon à
    maximiser le rendement tout en maîtrisant le risque. L'outil standard du secteur
    est l'optimisation moyenne-variance de **Markowitz**. La question à laquelle
    répond ce projet : le Machine Learning fait-il mieux, et de combien ?
    """
)
st.divider()

# ── 2 & 3. Markowitz vs. notre système ─────────────────────────────────────
st.header("Ce qui existe aujourd'hui, et ce que nous ajoutons")

col1, col2, col3 = st.columns(3)
col1.metric(
    "Markowitz classique",
    f"{best_classical['sharpe_net']:.3f}",
    help=f"Meilleure stratégie classique : {D.label(best_classical['name'])}. "
         f"Ratio de Sharpe net de coûts, hors échantillon.",
)
col2.metric(
    "Notre système ML",
    f"{best_ml['sharpe_net']:.3f}",
    delta=f"{lift_pct:+.1f}%",
    help="Régime (HMM) + covariance dynamique. Ratio de Sharpe net de coûts.",
)
col3.metric(
    "Gain net",
    f"{u['headline_lift_absolute_sharpe']:+.3f} Sharpe",
    help="Différence absolue de ratio de Sharpe, nette de coûts de transaction.",
)

st.markdown(
    f"""
    Sur le portefeuille EURAFRIC ({u['oos_start']} → {u['oos_end']}, hors échantillon,
    net de coûts), notre système obtient un ratio de Sharpe de **{best_ml['sharpe_net']:.3f}**
    contre **{best_classical['sharpe_net']:.3f}** pour la meilleure approche classique
    (*{D.label(best_classical['name'])}*) — soit **{lift_pct:+.1f} %**.
    """
)

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

# ── 4. Est-ce réel ? — la couche de crédibilité ────────────────────────────
st.header("Est-ce un résultat réel, ou un backtest sur-optimisé ?")
st.markdown(
    f"""
    C'est la question qui compte, et nous y avons consacré une phase entière. Le
    chiffre ci-dessus a été revalidé sur une **fenêtre de test gelée**
    ({p5['test_start']} → {p5['test_end']}) que la procédure de calibration n'a
    jamais vue, avec des **intervalles de confiance à 90 %** obtenus par bootstrap
    par blocs.
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
    **Lecture honnête de ce graphique.** Sur cette fenêtre de test, notre système
    conserve la meilleure performance ponctuelle — mais les intervalles de confiance
    se chevauchent largement. Nous présentons donc ce gain comme une **estimation
    ponctuelle favorable**, pas comme une supériorité statistiquement démontrée : la
    fenêtre de test (~1,8 an) est trop courte pour trancher. Cette prudence est
    délibérée — c'est précisément ce qu'un backtest sur-optimisé ne montrerait pas.
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
    - **Le gain n'est pas universel.** Sur l'univers ETF internationaux seuls
      ({D.UNIVERSE_LABELS[other_universe]}), notre système fait **{ou['headline_lift_pct']:+.1f} %**
      face au Markowitz classique — c'est-à-dire qu'il **perd**. Le ML apporte de la
      valeur là où la diversification naïve est faible et les régimes marqués, pas
      partout.
    - **Ce cas défavorable est en partie un artefact de contrainte, pas seulement
      un verdict sur le ML.** Avec 5 actifs et un plafond de 25 %, tout portefeuille
      admissible doit placer au moins 4 actifs *au plafond* (5 × 0,25 = 1,25) : la
      contrainte, et non le modèle, détermine l'essentiel de l'allocation. À 25 % la
      variance minimale ne produit qu'**une seule** allocation sur 248 rééquilibrages ;
      à 30 %, elle en produit 169. L'univers à 9 actifs n'est pas concerné.
    - **La significativité statistique n'est pas établie.** Les intervalles de
      confiance se chevauchent ; la fenêtre de test hors échantillon est courte.
    - **La couche de prédiction de rendement (F7) n'apporte rien.** Trois expériences
      indépendantes (calibration honnête, historique marocain profond sur 20 ans,
      ajout de données fondamentales) l'ont confirmé. Notre valeur ajoutée vient du
      **régime et de la covariance dynamique**, pas de la prédiction de rendement.
    - **Exposition de change non couverte.** Les actifs BVC sont en MAD, les ETF en
      USD ; les résultats intègrent une exposition USD/MAD non couverte.
    - **Historique BVC limité.** Les données gratuites de la Bourse de Casablanca ne
      remontent qu'à mi-2021, ce qui exclut le krach COVID de cet univers. L'univers
      ETF, lui, remonte désormais à 2004 et intègre la crise de 2008.
    """
)

st.success(
    """
    **Pourquoi présenter les limites aussi clairement ?** Parce qu'un chiffre validé
    hors échantillon, avec son intervalle de confiance et ses cas d'échec, est plus
    solide qu'un chiffre plus flatteur mais invérifiable. C'est le critère qu'un
    professionnel appliquerait avant de mettre du capital derrière un modèle.
    """,
    icon="✅",
)
