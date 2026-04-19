"""Benchmark artifacts + EXPLAIN."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import httpx
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Benchmark", layout="wide")
st.title("Benchmark")

api = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"),
).rstrip("/")

with httpx.Client(base_url=api, timeout=120.0) as client:
    stats = client.get("/api/benchmark/stats")
    if stats.status_code == 200:
        sj = stats.json()
        if sj.get("available"):
            st.subheader("pg_stat_statements (top by total time)")
            st.dataframe(sj.get("statements", []), use_container_width=True)
        else:
            st.info(sj.get("hint", "pg_stat_statements unavailable"))

    lst = client.get("/api/benchmark/results")
    names = []
    if lst.status_code == 200:
        names = [x["name"] for x in lst.json()]

    choice = st.selectbox("Result JSON", names) if names else None
    payload = None
    if choice:
        r = client.get(f"/api/benchmark/results/{choice}")
        if r.status_code == 200:
            payload = r.json()

    st.subheader("EXPLAIN (whitelisted)")
    qk = st.selectbox("query_kind", ["hnsw_knn", "seqscan_knn", "hybrid_context"])
    kk = st.number_input("k", 1, 200, 20)
    ex_btn = st.button("Run EXPLAIN")
    plan_text = ""
    if ex_btn:
        er = client.post("/api/benchmark/explain", json={"query_kind": qk, "k": int(kk)})
        if er.status_code == 200:
            plan_text = er.json().get("plan_text", "")
        else:
            st.error(er.text)

if plan_text:
    st.code(plan_text, language="text")

if payload and isinstance(payload, dict):
    summ = payload.get("summary") or {}
    ex = summ.get("exact_execution_ms_p50_p95_p99")
    hw = summ.get("hnsw_execution_ms_p50_p95_p99")
    if isinstance(ex, list) and isinstance(hw, list) and len(ex) >= 3 and len(hw) >= 3:
        fig = go.Figure(
            data=[
                go.Bar(name="Exact (no HNSW) p50/p95/p99", x=["p50", "p95", "p99"], y=ex[:3]),
                go.Bar(name="HNSW p50/p95/p99", x=["p50", "p95", "p99"], y=hw[:3]),
            ]
        )
        fig.update_layout(barmode="group", title="Latency (ms) from saved run", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    rows = payload.get("rows")
    if isinstance(rows, list) and rows and "recall_at_k" in rows[0]:
        recalls = [float(r["recall_at_k"]) for r in rows if str(r.get("recall_at_k", "")).strip() != ""]
        if recalls:
            st.metric("Mean recall (approx rows)", f"{sum(recalls) / len(recalls):.4f}")

if not names:
    st.warning("No JSON files under benchmark/results. Run benchmark scripts from the repo Makefile.")
