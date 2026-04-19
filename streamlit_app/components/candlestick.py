"""Plotly OHLCV + volume helper."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def ohlcv_figure(rows: list[dict[str, Any]], *, title: str = "") -> go.Figure:
    """Build a candlestick + volume figure from API-style row dicts."""
    if not rows:
        fig = go.Figure()
        fig.update_layout(title=title or "No data")
        return fig

    df = pd.DataFrame(rows)
    if "time" not in df.columns:
        raise ValueError("rows must include time")
    df["time"] = pd.to_datetime(df["time"], utc=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLCV",
        ),
        row=1,
        col=1,
    )
    colors = ["#2ecc71" if float(df["close"].iloc[i]) >= float(df["open"].iloc[i]) else "#e74c3c" for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df["time"], y=df["volume"], name="Volume", marker_color=colors),
        row=2,
        col=1,
    )
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=520, template="plotly_white")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def add_horizontal_levels(fig: go.Figure, levels: list[float], *, row: int = 1) -> None:
    for lv in levels:
        fig.add_hline(y=float(lv), line_dash="dash", line_color="rgba(99,110,250,0.7)", row=row, col=1)
