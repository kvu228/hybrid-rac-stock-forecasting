"""Normalized [0, 1] overlay for comparing window shapes."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _minmax_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def overlay_figure(close_series: list[np.ndarray], labels: list[str], *, title: str = "") -> go.Figure:
    """Overlay multiple 1D series after per-series min–max to [0, 1]."""
    fig = go.Figure()
    for s, lab in zip(close_series, labels, strict=True):
        y = _minmax_01(s)
        xs = np.arange(len(y))
        fig.add_trace(go.Scatter(x=xs, y=y, mode="lines", name=lab))
    fig.update_layout(
        title=title or "Close shape (min–max per series)",
        xaxis_title="Session index",
        yaxis_title="Normalized [0, 1]",
        height=400,
        template="plotly_white",
    )
    return fig
