"""OHLCV explorer with S/R overlay."""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

_repo = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import httpx
import streamlit as st

from streamlit_app.components.candlestick import add_horizontal_levels, ohlcv_figure

st.set_page_config(page_title="OHLCV", layout="wide")
st.title("OHLCV explorer")

api = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"),
).rstrip("/")

with httpx.Client(base_url=api, timeout=60.0) as client:
    sym_resp = client.get("/api/symbols")
    if sym_resp.status_code != 200:
        st.error(sym_resp.text)
        st.stop()
    symbols = [r["symbol"] for r in sym_resp.json()]
    if not symbols:
        st.warning("No symbols in database.")
        st.stop()

    symbol = st.selectbox("Symbol", symbols)
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start", value=date(2020, 1, 1))
    with c2:
        end = st.date_input("End", value=date.today())

    params = {"start": str(start), "end": str(end)}
    ohlcv_r = client.get(f"/api/ohlcv/{symbol}", params=params)
    if ohlcv_r.status_code != 200:
        st.error(ohlcv_r.text)
        st.stop()
    rows = ohlcv_r.json()

    sr_r = client.get(f"/api/sr-zones/{symbol}")
    zones = sr_r.json() if sr_r.status_code == 200 else []

fig = ohlcv_figure(rows, title=f"{symbol} OHLCV")
levels = [float(z["price_level"]) for z in zones if z.get("is_active", True)]
if levels:
    add_horizontal_levels(fig, levels)
st.plotly_chart(fig, use_container_width=True)

if zones:
    st.subheader("S/R zones")
    st.dataframe(zones, use_container_width=True)
