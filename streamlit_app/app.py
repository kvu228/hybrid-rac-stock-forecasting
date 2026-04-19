"""Streamlit entry: open sidebar pages for OHLCV, RAC, and benchmarks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import streamlit as st

st.set_page_config(page_title="Hybrid RAC Stock", layout="wide")
st.title("Hybrid RAC — Stock dashboard")
st.caption("Backend: FastAPI + PostgreSQL (TimescaleDB + pgvector).")

api_default = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
st.session_state.setdefault("API_BASE_URL", api_default)

with st.sidebar:
    st.text_input("API base URL", key="API_BASE_URL", help="Where uvicorn api.main:app is listening.")
    st.divider()
    st.markdown("**Pages** — use the sidebar navigation above.")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link("pages/1_ohlcv_chart.py", label="OHLCV explorer", icon="📈")
    st.page_link("pages/2_similar_patterns.py", label="Similar patterns", icon="🔎")
    st.page_link("pages/3_rac_prediction.py", label="RAC prediction", icon="🧠")
    st.page_link("pages/4_benchmark.py", label="Benchmark", icon="⏱️")

st.info("Chọn một trang từ menu **Pages** bên trái (Streamlit multipage).")
