"""Plotly OHLCV + volume helper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _polyline_verticals(xs: list[str], y0: np.ndarray, y1: np.ndarray) -> tuple[list[str | None], list[float | None]]:
    """Build one Scatter line with NaN breaks: (x, y0)-(x, y1) per bar."""
    out_x: list[str | None] = []
    out_y: list[float | None] = []
    for x, a, b in zip(xs, y0.tolist(), y1.tolist(), strict=True):
        out_x.extend([x, x, None])
        out_y.extend([float(a), float(b), None])
    return out_x, out_y


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
    df = df.sort_values("time").reset_index(drop=True)

    # Category axis = one slot per returned row → no empty space for holidays / gaps.
    df["_x"] = df["time"].dt.strftime("%Y-%m-%d")
    if df["_x"].duplicated().any():
        df["_x"] = df["time"].dt.strftime("%Y-%m-%d %H:%M")

    x_cat = df["_x"].tolist()
    opn = df["open"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    cls = df["close"].astype(float).to_numpy()
    bull = cls >= opn

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28])
    up_c, down_c = "#089981", "#f23645"

    # Wicks: pure vertical segments low → high (no horizontal caps at H/L).
    for name, mask, color, lw in (
        ("wick+", bull, up_c, 1.35),
        ("wick-", ~bull, down_c, 1.35),
    ):
        if not mask.any():
            continue
        xv = [x_cat[i] for i in np.flatnonzero(mask)]
        wx, wy = _polyline_verticals(xv, low[mask], high[mask])
        fig.add_trace(
            go.Scatter(
                x=wx,
                y=wy,
                mode="lines",
                line=dict(color=color, width=lw),
                name=name,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Bodies: vertical segment open → close (thicker).
    for name, mask, color, lw in (
        ("body+", bull, up_c, 2.8),
        ("body-", ~bull, down_c, 2.8),
    ):
        if not mask.any():
            continue
        xv = [x_cat[i] for i in np.flatnonzero(mask)]
        bx, by = _polyline_verticals(xv, opn[mask], cls[mask])
        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color=color, width=lw),
                name=name,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    time_lbl = df["time"].dt.strftime("%Y-%m-%d %H:%M UTC").tolist()
    customdata = np.column_stack([opn, high, low, cls, time_lbl])
    fig.add_trace(
        go.Scatter(
            x=x_cat,
            y=cls,
            mode="markers",
            marker=dict(size=6, color="rgba(0,0,0,0)"),
            customdata=customdata,
            hovertemplate=(
                "%{customdata[4]}<br>"
                "O %{customdata[0]:.4f} &nbsp; H %{customdata[1]:.4f}<br>"
                "L %{customdata[2]:.4f} &nbsp; C %{customdata[3]:.4f}"
                "<extra></extra>"
            ),
            name="OHLCV",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    colors = [up_c if bull[i] else down_c for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=x_cat,
            y=df["volume"].astype(float),
            name="Volume",
            marker_color=colors,
            marker_line_width=0,
            opacity=0.85,
            width=0.65,
            showlegend=False,
            hovertemplate="Vol %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    ymin = float(df["low"].min())
    ymax = float(df["high"].max())
    span = max(ymax - ymin, max(abs(ymax), 1.0) * 1e-6)
    y_pad = span * 0.06

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=560,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=52, r=24, t=48, b=40),
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(size=12, color="#d1d4dc"),
        dragmode="pan",
    )
    xaxis_cat: dict[str, Any] = dict(
        type="category",
        categoryorder="array",
        categoryarray=x_cat,
        tickangle=-55,
        automargin=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#758696",
        spikethickness=1,
        gridcolor="#2a2e39",
        showgrid=True,
    )
    if len(x_cat) > 40:
        step = max(1, len(x_cat) // 16)
        tick_idx = list(range(0, len(x_cat), step))
        if tick_idx[-1] != len(x_cat) - 1:
            tick_idx.append(len(x_cat) - 1)
        tv = [x_cat[i] for i in tick_idx]
        xaxis_cat["tickvals"] = tv
        xaxis_cat["ticktext"] = tv
    fig.update_xaxes(**xaxis_cat)
    fig.update_yaxes(
        title_text="Price",
        row=1,
        col=1,
        range=[ymin - y_pad, ymax + y_pad],
        fixedrange=False,
        side="right",
        showgrid=True,
        gridcolor="#2a2e39",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Volume",
        row=2,
        col=1,
        fixedrange=False,
        side="right",
        showgrid=True,
        gridcolor="#2a2e39",
        zeroline=False,
    )
    return fig


def filter_zones_for_overlay(
    zones: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    max_zones: int = 18,
    min_strength: float = 3.0,
    price_pad_ratio: float = 0.07,
    merge_gap_ratio: float = 0.012,
) -> list[dict[str, Any]]:
    """Pick a small set of S/R zones relevant to the visible OHLCV window.

    Keeps zones near the on-screen price range, prefers higher *strength*, and
    drops levels that sit almost on top of each other so the chart stays readable.
    """
    active = [z for z in zones if z.get("is_active", True)]
    strong = [z for z in active if float(z.get("strength", 0)) >= min_strength]
    pool = strong if strong else active
    if not pool:
        return []

    if not rows:
        pool_sorted = sorted(pool, key=lambda z: -float(z.get("strength", 0)))
        return _dedupe_zones_by_price_gap(pool_sorted, gap=1.0, max_zones=max_zones)

    lows = [float(r["low"]) for r in rows]
    highs = [float(r["high"]) for r in rows]
    lo, hi = min(lows), max(highs)
    span = max(hi - lo, max(abs(lo), abs(hi)) * 1e-6)
    pad = span * price_pad_ratio
    lo_b, hi_b = lo - pad, hi + pad

    in_band = [z for z in pool if lo_b <= float(z["price_level"]) <= hi_b]
    if not in_band:
        in_band = pool

    in_band.sort(key=lambda z: -float(z.get("strength", 0)))
    gap = max(span * merge_gap_ratio, span * 1e-4)
    return _dedupe_zones_by_price_gap(in_band, gap=gap, max_zones=max_zones)


def _dedupe_zones_by_price_gap(
    zones: list[dict[str, Any]],
    *,
    gap: float,
    max_zones: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for z in zones:
        if len(selected) >= max_zones:
            break
        pl = float(z["price_level"])
        if any(abs(pl - float(s["price_level"])) < gap for s in selected):
            continue
        selected.append(z)
    return selected


def add_sr_overlay(fig: go.Figure, zones: list[dict[str, Any]], *, row: int = 1) -> None:
    """Draw S/R zones as dashed horizontal lines (support vs resistance colors)."""
    for z in zones:
        lv = float(z["price_level"])
        zt = str(z.get("zone_type", "")).upper()
        if zt == "SUPPORT":
            color = "rgba(39, 174, 96, 0.85)"
        elif zt == "RESISTANCE":
            color = "rgba(192, 57, 43, 0.85)"
        else:
            color = "rgba(52, 152, 219, 0.75)"
        fig.add_hline(y=lv, line_dash="dash", line_color=color, line_width=1.2, row=row, col=1)


def add_horizontal_levels(fig: go.Figure, levels: list[float], *, row: int = 1) -> None:
    """Draw raw price levels (no zone metadata). Prefer :func:`add_sr_overlay`."""
    for lv in levels:
        fig.add_hline(y=float(lv), line_dash="dash", line_color="rgba(99,110,250,0.7)", row=row, col=1)
