"""Normalized [0, 1] views for comparing window shapes."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _minmax_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def overlay_figure(close_series: list[np.ndarray], labels: list[str], *, title: str = "") -> go.Figure:
    """One subplot per series (shared x): easier to read than a single crowded overlay."""
    return close_curves_faceted(close_series, labels, title=title or "Close shape (min–max per series)")


def close_curves_faceted(
    close_series: list[np.ndarray],
    labels: list[str],
    *,
    title: str = "",
    row_height_px: int = 112,
    max_total_height: int = 1100,
) -> go.Figure:
    """Stack each normalized close series in its own row (TradingView-style strip chart)."""
    n = len(close_series)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title=title or "No series")
        return fig

    def _short(s: str, max_len: int = 44) -> str:
        s = str(s)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    titles = tuple(_short(lab) for lab in labels) if len(labels) == n else tuple(f"series {i}" for i in range(n))
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=titles,
    )
    palette = ("#2962ff", "#ffa726", "#ab47bc", "#26a69a", "#ef5350", "#42a5f5", "#ec407a", "#78909c")
    for i, (s, lab) in enumerate(zip(close_series, labels, strict=True), start=1):
        y = _minmax_01(s)
        xs = np.arange(len(y), dtype=np.float64)
        color = palette[(i - 1) % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=y,
                mode="lines",
                line=dict(width=2, color=color),
                name=lab,
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(
            title_text="0–1",
            range=[-0.03, 1.03],
            row=i,
            col=1,
            showgrid=True,
            gridcolor="#2a2e39",
            zeroline=False,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#2a2e39", row=i, col=1)

    total_h = min(row_height_px * n + 80, max_total_height)
    fig.update_layout(
        title=title or "Close curves (min–max normalized, one row per series)",
        height=total_h,
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(size=11, color="#d1d4dc"),
        margin=dict(l=48, r=28, t=72, b=40),
    )
    fig.update_xaxes(title_text="Session index (0 = oldest in window)", row=n, col=1)
    return fig
