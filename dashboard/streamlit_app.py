"""
streamlit_app.py — Entry point for the Portfolio ML Suite dashboard.

Two research views (Streamlit renders `pages/` in the sidebar):

  1. Résultats de recherche — evidence and caveats for stakeholders. No
     controls; observed differences are never promoted to recommendations.
  2. Explorateur de stratégies — interactive research comparison. It reads the
     same versioned artifacts as the read-only FastAPI service.

Run with:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Portfolio ML — EURAFRIC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Prototype de recherche — optimisation de portefeuille assistée par ML")
st.caption("EURAFRIC Information · INPT · Projet de Fin d'Année")

st.markdown(
    """
    Ce prototype analyse des allocations entre actions de la Bourse de Casablanca et
    ETF internationaux, sous contraintes réalistes de gestion (long-only, plafond de
    25 % par actif, coûts de transaction déduits). Il compare les résultats historiques
    du Machine Learning à la méthode classique de Markowitz, sans fournir de conseil,
    de recommandation client ni d'exécution d'ordres.

    ### Deux vues

    **📊 Résultats de recherche** — *pour les décideurs.* Les résultats observés,
    leur validation hors échantillon et leurs limites d'inférence.

    **🛠️ Explorateur de stratégies** — *pour l'analyse.* Sélection d'univers et de
    stratégie, allocations historiques et export CSV.

    👈 Choisissez une vue dans la barre latérale.
    """
)

with st.sidebar:
    st.markdown("### À propos")
    st.caption(
        "Tous les chiffres affichés proviennent d'artefacts Gold versionnés, "
        "régénérés par `python src/run_dashboard_data.py`. Aucune valeur n'est "
        "codée en dur : un test automatisé (`tests/test_run_dashboard_data.py`) "
        "vérifie que chaque chiffre annoncé correspond à sa source."
    )
