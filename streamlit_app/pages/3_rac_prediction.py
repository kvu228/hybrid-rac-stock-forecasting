"""RAC full context + optional predict."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import httpx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RAC prediction", layout="wide")
st.title("RAC prediction (hybrid context)")

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
    current_price = float(bars[idx]["close"])

    qe = client.post(
        "/api/rac/query-embedding",
        json={"symbol": symbol, "window_end": window_end.isoformat()},
    )
    if qe.status_code != 200:
        st.error(qe.text)
        st.stop()
    emb = qe.json()["query_embedding"]

    fc = client.post(
        "/api/rac/full-context",
        json={
            "query_embedding": emb,
            "symbol": symbol,
            "current_price": current_price,
            "k": k,
        },
    )
    if fc.status_code != 200:
        st.error(fc.text)
        st.stop()
    ctx = fc.json()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Neighbors", ctx.get("total_neighbors", 0))
c2.metric("Avg cosine dist", f"{ctx.get('avg_cosine_dist') or 0:.4f}")
c3.metric("Dominant label", ctx.get("dominant_label", "—"))
c4.metric("KNN confidence", f"{(ctx.get('knn_confidence') or 0) * 100:.1f}%")

ld = ctx.get("label_distribution") or {}
if ld:
    st.plotly_chart(px.pie(names=list(ld.keys()), values=list(ld.values()), title="Label distribution"), use_container_width=True)

ratio = ctx.get("sr_position_ratio")
fig_g = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=float(ratio) * 100 if ratio is not None else 0,
        title={"text": "S/R position ratio (%)"},
        gauge={"axis": {"range": [0, 100]}},
    )
)
fig_g.update_layout(height=280)
st.plotly_chart(fig_g, use_container_width=True)

st.subheader("Evidence")
st.write("Neighbor IDs:", ctx.get("neighbor_ids", []))
st.caption("Use **Similar patterns** page with the same window_end to inspect neighbors visually.")

pred = st.button("Run predict (persist)")
if pred:
    with httpx.Client(base_url=api, timeout=120.0) as client:
        pr = client.post(
            "/api/rac/predict",
            params={"persist": "true"},
            json={
                "query_embedding": emb,
                "symbol": symbol,
                "current_price": current_price,
                "k": k,
            },
        )
        if pr.status_code == 200:
            st.success(pr.json())
        else:
            st.error(pr.text)
