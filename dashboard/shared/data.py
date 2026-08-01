"""
data.py — Single source of truth for what the dashboard reads.

Both pages import from here. Nothing else in `dashboard/` may touch a parquet
or JSON directly — one loader means the pitch page and the manager tool can
never disagree about a number, which is the drift failure this whole layer
exists to prevent (see `tests/test_run_dashboard_data.py`'s rationale).

Everything is cached with `st.cache_data` so page switches and widget
interactions don't re-read from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold"

# Human-facing French labels. Kept next to the loader so a strategy renamed in
# `run_dashboard_data.HEADLINE_STRATEGIES` fails loudly here rather than
# rendering a raw snake_case key to a stakeholder.
STRATEGY_LABELS = {
    "equal_weight": "Équipondéré (1/N)",
    "min_variance_lw": "Markowitz — variance minimale (Ledoit-Wolf)",
    "max_sharpe": "Markowitz — Sharpe maximal",
    "regime_conditional": "Notre système ML (régime + covariance dynamique)",
    "rf_signal_tuned": "Signal ML RandomForest (F7, calibré)",
    "xgb_signal_tuned": "Signal ML XGBoost (F7, calibré)",
}

CLASSICAL_STRATEGIES = ("equal_weight", "min_variance_lw", "max_sharpe")
ML_STRATEGIES = ("regime_conditional",)

UNIVERSE_LABELS = {
    "etf_2017": "ETF internationaux (5 actifs, 2004→)",
    "full_2021": "Portefeuille EURAFRIC (9 actifs BVC + ETF, 2021→)",
}


class DashboardDataMissing(RuntimeError):
    """Raised when the Gold artifacts haven't been generated yet."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise DashboardDataMissing(
            f"Artefact manquant : {path.name}\n\n"
            f"Lancez d'abord :\n\n    python src/run_dashboard_data.py\n\n"
            f"(ou `dvc repro dashboard_data`)"
        )
    return path


@st.cache_data(show_spinner=False)
def load_showcase() -> dict:
    """Metrics table + headline comparison, per universe."""
    return json.loads(_require(GOLD / "dashboard_showcase.json").read_text())


@st.cache_data(show_spinner=False)
def load_equity() -> pd.DataFrame:
    """Long-form daily returns: (Date, universe, strategy, gross_return, net_return)."""
    return pd.read_parquet(_require(GOLD / "dashboard_equity.parquet"))


@st.cache_data(show_spinner=False)
def load_weights() -> pd.DataFrame:
    """Long-form target weights at each rebalance: (Date, universe, strategy, asset, weight)."""
    return pd.read_parquet(_require(GOLD / "dashboard_weights.parquet"))


@st.cache_data(show_spinner=False)
def load_regime() -> pd.DataFrame:
    """HMM regime timeline for regime_conditional: (Date, universe, bull_prob, regime)."""
    return pd.read_parquet(_require(GOLD / "dashboard_regime.parquet"))


@st.cache_data(show_spinner=False)
def load_crisis() -> dict:
    """Per-crisis behaviour + unsupervised regime-detection rates.

    Optional: returns {} when the artifact hasn't been generated, so the pitch
    page degrades to its other sections rather than failing. Everything else
    here is required, because a missing headline is worse than a missing
    supporting section.
    """
    path = GOLD / "crisis_windows.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_phase5() -> dict:
    """The committed out-of-sample evaluation — used for the credibility layer."""
    return json.loads(_require(GOLD / "phase5_results.json").read_text())


def equity_curve(equity: pd.DataFrame, universe: str, strategy: str,
                 net: bool = True) -> pd.Series:
    """Cumulative wealth index (base 100) for one (universe, strategy)."""
    col = "net_return" if net else "gross_return"
    rows = equity[(equity["universe"] == universe) & (equity["strategy"] == strategy)]
    rows = rows.sort_values("Date")
    return pd.Series(
        (1.0 + rows[col]).cumprod().to_numpy() * 100.0,
        index=pd.DatetimeIndex(rows["Date"]),
        name=STRATEGY_LABELS.get(strategy, strategy),
    )


def latest_weights(weights: pd.DataFrame, universe: str, strategy: str) -> pd.Series:
    """Most recent target allocation for one (universe, strategy)."""
    rows = weights[(weights["universe"] == universe) & (weights["strategy"] == strategy)]
    if rows.empty:
        return pd.Series(dtype="float64")
    last_date = rows["Date"].max()
    latest = rows[rows["Date"] == last_date]
    return latest.set_index("asset")["weight"].sort_values(ascending=False)


def label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)
