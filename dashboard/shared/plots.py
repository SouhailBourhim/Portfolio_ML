"""
plots.py — Shared Plotly figures for both dashboard pages.

One plotting library for the pitch page and the manager tool, so the same
concept always looks the same. All French labels; all colour choices carry
meaning (classical = muted greys/blues, our ML system = a single accent
colour, so the "which line is ours" question never needs a legend hunt).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# One accent for "our system", muted tones for everything it's compared to.
COLOR_ML = "#0F766E"          # teal — our regime-conditional system
COLOR_CLASSICAL = "#64748B"   # slate — classical Markowitz
COLOR_BASELINE = "#94A3B8"    # light slate — 1/N
COLOR_ALT = "#B45309"         # amber — secondary/other strategies
COLOR_BULL = "rgba(15, 118, 110, 0.10)"
COLOR_BEAR = "rgba(180, 83, 9, 0.12)"

_STRATEGY_COLORS = {
    "regime_conditional": COLOR_ML,
    "max_sharpe": COLOR_CLASSICAL,
    "min_variance_lw": COLOR_CLASSICAL,
    "equal_weight": COLOR_BASELINE,
}


def _base_layout(fig: go.Figure, title: str = "", height: int = 420,
                 show_legend: bool = True) -> go.Figure:
    # The horizontal legend sits ABOVE the plot, and the title above IT — hence
    # the generous top margin. A tighter margin makes the two overlap, which
    # looked broken in the first render.
    fig.update_layout(
        title=dict(text=title, y=0.98, yanchor="top", font=dict(size=15)),
        height=height,
        margin=dict(l=40, r=20, t=90 if title else 50, b=40),
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=11)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return fig


def equity_comparison(curves: dict[str, pd.Series], highlight: str | None = None,
                      title: str = "", height: int = 420) -> go.Figure:
    """Overlaid wealth curves (base 100). `highlight` is drawn thicker on top."""
    fig = go.Figure()
    ordered = sorted(curves, key=lambda k: (k == highlight))  # highlight drawn last
    for key in ordered:
        series = curves[key]
        is_highlight = key == highlight
        fig.add_trace(go.Scatter(
            x=series.index, y=series.to_numpy(), name=series.name or key,
            mode="lines",
            line=dict(
                color=_STRATEGY_COLORS.get(key, COLOR_ALT),
                width=3.2 if is_highlight else 1.6,
            ),
            opacity=1.0 if is_highlight else 0.75,
        ))
    fig.update_yaxes(title_text="Valeur du portefeuille (base 100)")
    return _base_layout(fig, title, height)


def sharpe_with_ci(labels: list[str], point: list[float],
                   ci_low: list[float | None], ci_high: list[float | None],
                   highlight_index: int | None = None,
                   title: str = "", height: int = 380) -> go.Figure:
    """Horizontal Sharpe bars with confidence-interval whiskers.

    The CI is the whole point: this is the figure that keeps the pitch honest,
    showing a stakeholder that the improvement is a point estimate inside a
    wide interval, not a guarantee.
    """
    colors = [
        COLOR_ML if i == highlight_index else COLOR_CLASSICAL
        for i in range(len(labels))
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=point, y=labels, orientation="h",
        marker=dict(color=colors),
        name="Sharpe net", showlegend=False,
        hovertemplate="%{y}<br>Sharpe net : %{x:.3f}<extra></extra>",
    ))
    # The CI whisker runs horizontally through the bar's vertical CENTRE, so no
    # x-position inside or outside the bar is free at that height. The value
    # label is therefore drawn as an annotation ABOVE the centre line.
    for i, value in enumerate(point):
        fig.add_annotation(
            x=value / 2.0, y=i + 0.22,
            text=f"<b>{value:.3f}</b>", showarrow=False,
            font=dict(color="white", size=13), yref="y", xref="x",
        )
    for i, (lo, hi) in enumerate(zip(ci_low, ci_high)):
        if lo is None or hi is None:
            continue
        fig.add_shape(
            type="line", x0=lo, x1=hi, y0=labels[i], y1=labels[i],
            line=dict(color="rgba(15,23,42,0.55)", width=2),
        )
        for x in (lo, hi):
            fig.add_shape(
                type="line", x0=x, x1=x, y0=i - 0.16, y1=i + 0.16,
                yref="y", line=dict(color="rgba(15,23,42,0.55)", width=2),
            )
    fig.add_vline(x=0, line=dict(color="rgba(148,163,184,0.6)", width=1, dash="dot"))

    # Pad the x-range so whiskers never clip the plot edge.
    finite = [v for v in list(ci_low) + list(ci_high) + list(point) if v is not None]
    if finite:
        lo_b, hi_b = min(finite), max(finite)
        pad = max(0.12, (hi_b - lo_b) * 0.08)
        fig.update_xaxes(range=[min(0.0, lo_b) - pad, hi_b + pad])
    fig.update_xaxes(title_text="Ratio de Sharpe net (annualisé)")
    return _base_layout(fig, title, height, show_legend=False)


def weights_bar(weights: pd.Series, title: str = "", height: int = 360) -> go.Figure:
    """Latest target allocation, one bar per asset."""
    fig = go.Figure(go.Bar(
        x=weights.to_numpy() * 100.0, y=list(weights.index), orientation="h",
        marker=dict(color=COLOR_ML),
        text=[f"{v * 100:.1f}%" for v in weights.to_numpy()], textposition="outside",
        hovertemplate="%{y} : %{x:.2f}%<extra></extra>",
    ))
    fig.update_xaxes(title_text="Allocation (%)")
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, title, height)


def regime_timeline(equity_curve_series: pd.Series, regime: pd.DataFrame,
                    title: str = "", height: int = 420) -> go.Figure:
    """The ML system's wealth curve with detected bull/bear regimes shaded behind it.

    This is the "comment ça marche" figure — it makes the abstract claim
    ("the model switches posture as regimes shift") visible without any maths.
    """
    fig = go.Figure()
    regime = regime.sort_values("Date").reset_index(drop=True)
    dates = pd.DatetimeIndex(regime["Date"])
    end = equity_curve_series.index.max()

    for i, row in regime.iterrows():
        start = dates[i]
        stop = dates[i + 1] if i + 1 < len(dates) else end
        fig.add_vrect(
            x0=start, x1=stop,
            fillcolor=COLOR_BULL if row["regime"] == "bull" else COLOR_BEAR,
            line_width=0, layer="below",
        )

    fig.add_trace(go.Scatter(
        x=equity_curve_series.index, y=equity_curve_series.to_numpy(),
        mode="lines", name=equity_curve_series.name or "Notre système",
        line=dict(color=COLOR_ML, width=2.6),
    ))
    # Legend proxies for the shaded bands.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="Régime haussier détecté",
        marker=dict(size=12, color=COLOR_BULL.replace("0.10", "0.55"), symbol="square"),
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="Régime baissier détecté",
        marker=dict(size=12, color=COLOR_BEAR.replace("0.12", "0.55"), symbol="square"),
    ))
    fig.update_yaxes(title_text="Valeur du portefeuille (base 100)")
    return _base_layout(fig, title, height)
