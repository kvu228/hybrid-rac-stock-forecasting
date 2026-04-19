"""Similar patterns: query embedding + KNN + overlay."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import numpy as np
import httpx
import plotly.express as px
import streamlit as st

from streamlit_app.components.candlestick import ohlcv_figure
from streamlit_app.components.similarity import overlay_figure

st.set_page_config(page_title="Similar patterns", layout="wide")
st.title("Similar patterns (pgvector KNN)")

api = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"),
).rstrip("/")

k = st.sidebar.slider("K neighbors", 5, 50, 20)

with httpx.Client(base_url=api, timeout=120.0) as client:
    sym_resp = client.get("/api/symbols")
    if sym_resp.status_code != 200:
        st.error(sym_resp.text)
        st.stop()
    symbols = [r["symbol"] for r in sym_resp.json()]
    if not symbols:
        st.warning("No symbols.")
        st.stop()
    symbol = st.selectbox("Symbol", symbols)

    latest = client.get(f"/api/ohlcv/{symbol}/latest", params={"n": 120})
    if latest.status_code != 200:
        st.error(latest.text)
        st.stop()
    bars = latest.json()
    if len(bars) < 30:
        st.warning("Need at least 30 sessions.")
        st.stop()
    times = [datetime.fromisoformat(str(b["time"]).replace("Z", "+00:00")) for b in bars]
    idx = st.select_slider(
        "Window end (session)",
        options=list(range(len(times))),
        value=len(times) - 1,
        format_func=lambda i: times[i].strftime("%Y-%m-%d"),
    )
    window_end = times[idx]

    qe = client.post(
        "/api/rac/query-embedding",
        json={"symbol": symbol, "window_end": window_end.isoformat()},
    )
    if qe.status_code != 200:
        st.error(qe.text)
        st.stop()
    qpayload = qe.json()
    emb = qpayload["query_embedding"]

    sim = client.post(
        "/api/rac/similar-patterns",
        json={
            "query_embedding": emb,
            "k": k,
            "similarity_threshold": 0.99,
            "filter_symbol": None,
        },
    )
    if sim.status_code != 200:
        st.error(sim.text)
        st.stop()
    neighbors = sim.json().get("neighbors", [])

st.caption(f"Query window_end = {window_end.isoformat()} — {sim.json().get('query_time_ms', 0):.2f} ms")

cols = st.columns(min(4, max(1, len(neighbors))))

closes_query = [float(b["close"]) for b in qpayload.get("ohlcv", [])]
series_list = [np.array(closes_query, dtype=np.float64)]
labels_overlay = ["query"]
for i, n in enumerate(neighbors[:8]):
    sym_n = n["symbol"]
    ws = n["window_start"]
    we = n["window_end"]
    r = httpx.get(
        f"{api}/api/ohlcv/{sym_n}",
        params={"start": str(ws)[:10], "end": str(we)[:10]},
        timeout=60.0,
    )
    if r.status_code == 200:
        ohr = r.json()
        if len(ohr) >= 30:
            ohr = sorted(ohr, key=lambda x: x["time"])[-30:]
        if i < len(cols) and ohr:
            with cols[i % len(cols)]:
                st.plotly_chart(ohlcv_figure(ohr, title=f"#{n['id']} {sym_n}"), use_container_width=True)
        cl = [float(x["close"]) for x in ohr[-30:]] if ohr else []
        if len(cl) == 30:
            series_list.append(np.array(cl, dtype=np.float64))
            labels_overlay.append(f"id {n['id']}")

if len(series_list) > 1:
    st.plotly_chart(
        overlay_figure(series_list, labels_overlay, title="Close curves (min–max normalized)"),
        use_container_width=True,
    )

if neighbors:
    st.subheader("Neighbors")
    st.dataframe(neighbors, use_container_width=True)
    labels = [str(n.get("label", "")) for n in neighbors]
    fig_pie = px.pie(names=labels, title="Label distribution (neighbors)")
    st.plotly_chart(fig_pie, use_container_width=True)
