"""
streamlit_app.py — Entry point for the Portfolio ML Suite dashboard.

Two pages, two audiences (Streamlit renders `pages/` in the sidebar):

  1. Histoire de valeur — business-facing pitch. No controls. Shows what the
     ML system adds over classical Markowitz, with the out-of-sample rigor
     visible as a credibility layer rather than a footnote.
  2. Outil du gestionnaire — the operator tool. Interactive; consumes the
     FastAPI service when it's running.

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

st.title("Système d'optimisation de portefeuille piloté par le Machine Learning")
st.caption("EURAFRIC Information · INPT · Projet de Fin d'Année")

st.markdown(
    """
    Ce prototype alloue un capital entre actions de la Bourse de Casablanca et ETF
    internationaux, sous contraintes réalistes de gestion (long-only, plafond de 25 %
    par actif, coûts de transaction déduits), et compare rigoureusement l'apport du
    Machine Learning à la méthode classique de Markowitz.

    ### Deux vues

    **📊 Histoire de valeur** — *pour les décideurs.* Ce que le système ML apporte
    par rapport à la méthode classique, avec la validation hors échantillon qui rend
    ce chiffre crédible.

    **🛠️ Outil du gestionnaire** — *pour l'utilisateur métier.* Sélection d'univers et
    de stratégie, allocations cibles, export CSV.

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
