"""
2_Outil_gestionnaire.py — The operator tool (Phase 6).

Different audience, different rules from Page 1. A portfolio manager comparing
options legitimately needs to see EVERY strategy, including the F7 signals that
the pitch page deliberately excludes — the honest presentation here is "here is
what each one did, with its uncertainty", not "here is our recommendation".

Reads through `shared.data` (the same loader Page 1 uses), so the two pages
cannot disagree about a number. The FastAPI service (`src/api/main.py`) exposes
the identical artifacts for external consumers; this page reads the files
directly rather than round-tripping through HTTP, since they're the same
process's filesystem — the API exists for OTHER consumers, not for this page
to talk to itself.
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

st.set_page_config(page_title="Outil du gestionnaire", page_icon="🛠️", layout="wide")

try:
    showcase = D.load_showcase()
    equity = D.load_equity()
    weights_all = D.load_weights()
except D.DashboardDataMissing as exc:
    st.error(str(exc))
    st.stop()

st.title("🛠️ Outil du gestionnaire")
st.caption(
    "Mode avancé — comparaison de stratégies, allocations cibles et export. "
    "Tous les résultats sont nets de coûts et hors échantillon."
)

# ── Controls ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paramètres")
    universe = st.selectbox(
        "Univers",
        options=list(showcase["universes"]),
        format_func=lambda u: D.UNIVERSE_LABELS.get(u, u),
    )
    u = showcase["universes"][universe]
    available = list(u["strategies"])
    selected = st.multiselect(
        "Stratégies à comparer",
        options=available,
        default=available,
        format_func=D.label,
    )
    show_net = st.radio(
        "Performance affichée", options=[True, False],
        format_func=lambda net: "Nette de coûts" if net else "Brute",
        horizontal=True,
    )
    st.divider()
    st.caption(
        f"**Contexte du backtest**\n\n"
        f"- Rééquilibrage : {showcase['rebalance_freq']} (mensuel)\n"
        f"- Plafond par actif : {showcase['max_weight'] * 100:.0f} %\n"
        f"- Coûts : {showcase['cost_bps']['etf']} bps (ETF) / "
        f"{showcase['cost_bps']['bvc']} bps (BVC)\n"
        f"- Période : {u['oos_start']} → {u['oos_end']}"
    )

if not selected:
    st.warning("Sélectionnez au moins une stratégie dans la barre latérale.")
    st.stop()

# ── Date range ─────────────────────────────────────────────────────────────
universe_equity = equity[equity["universe"] == universe]
all_dates = pd.DatetimeIndex(sorted(universe_equity["Date"].unique()))
start, end = st.select_slider(
    "Période affichée",
    options=list(all_dates),
    value=(all_dates[0], all_dates[-1]),
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"),
)

# ── Equity comparison ──────────────────────────────────────────────────────
st.subheader("Performance comparée")
curves = {}
for s in selected:
    curve = D.equity_curve(equity, universe, s, net=show_net)
    curve = curve.loc[start:end]
    if not curve.empty:
        # Re-base to 100 at the window's start so the comparison is fair when
        # the user narrows the period.
        curves[s] = curve / curve.iloc[0] * 100.0

highlight = "regime_conditional" if "regime_conditional" in curves else None
st.plotly_chart(
    P.equity_comparison(
        curves, highlight=highlight,
        title=f"Valeur du portefeuille, {'nette' if show_net else 'brute'} de coûts (base 100)",
    ),
    use_container_width=True,
)

# ── Metrics table ──────────────────────────────────────────────────────────
st.subheader("Métriques (période complète, hors échantillon)")
rows = []
for s in selected:
    m = u["strategies"][s]
    rows.append({
        "Stratégie": D.label(s),
        "Sharpe net": m["sharpe_net"],
        "Sharpe brut": m["sharpe_gross"],
        "Rendement annualisé": f"{m['annualized_return_net'] * 100:.2f} %",
        "Perte maximale": f"{m['max_drawdown'] * 100:.2f} %",
        "Calmar": m["calmar"],
        "Rotation moyenne": f"{m['avg_turnover'] * 100:.1f} %",
    })
metrics_df = pd.DataFrame(rows).sort_values("Sharpe net", ascending=False)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.caption(
    "⚠️ Les écarts de Sharpe entre stratégies sur cette fenêtre ne sont pas "
    "statistiquement significatifs : les intervalles de confiance hors "
    "échantillon se chevauchent (voir *Histoire de valeur*). À utiliser comme "
    "aide à la décision, pas comme classement définitif."
)

# ── Allocations ────────────────────────────────────────────────────────────
st.subheader("Allocation cible la plus récente")
alloc_strategy = st.selectbox(
    "Stratégie", options=selected, format_func=D.label, key="alloc_strategy"
)
latest = D.latest_weights(weights_all, universe, alloc_strategy)

if latest.empty:
    st.info("Aucune allocation disponible pour cette combinaison.")
else:
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.plotly_chart(
            P.weights_bar(latest, title=f"{D.label(alloc_strategy)}"),
            use_container_width=True,
        )
    with col_table:
        table = pd.DataFrame({
            "Actif": latest.index,
            "Poids": [f"{v * 100:.2f} %" for v in latest.to_numpy()],
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Télécharger (CSV)",
            data=latest.rename("weight").to_csv().encode("utf-8"),
            file_name=f"allocation_{universe}_{alloc_strategy}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Rebalance history ──────────────────────────────────────────────────────
with st.expander("Historique complet des rééquilibrages"):
    hist = weights_all[
        (weights_all["universe"] == universe)
        & (weights_all["strategy"] == alloc_strategy)
    ]
    if hist.empty:
        st.info("Aucun historique disponible.")
    else:
        pivot = hist.pivot_table(index="Date", columns="asset", values="weight")
        pivot = pivot.loc[start:end]
        st.dataframe(
            (pivot * 100).round(2), use_container_width=True,
        )
        st.download_button(
            "⬇️ Télécharger l'historique (CSV)",
            data=pivot.to_csv().encode("utf-8"),
            file_name=f"historique_{universe}_{alloc_strategy}.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "**API REST** — les mêmes données sont exposées pour un usage externe : "
    "`uvicorn api.main:app --app-dir src`, puis `/strategies`, `/metrics`, "
    "`/equity`, `/weights`, `/compare` (documentation interactive sur `/docs`)."
)
