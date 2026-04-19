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

from streamlit_app.components.candlestick import add_sr_overlay, filter_zones_for_overlay, ohlcv_figure

st.set_page_config(page_title="OHLCV", layout="wide")
st.title("OHLCV explorer")

api = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"),
).rstrip("/")
st.sidebar.markdown("**S/R on chart**")
max_sr_lines = st.sidebar.slider("Max zones drawn", 5, 40, 18)
min_sr_strength = st.sidebar.number_input("Min strength", min_value=1.0, value=3.0, step=0.5)

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
    default_day = date.today()
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start", value=default_day, help="Mặc định: 1 ngày (cùng ngày với End).")
    with c2:
        end = st.date_input("End", value=default_day)

    if start > end:
        st.error("Start phải trước hoặc cùng ngày với End.")
        st.stop()

    params = {"start": str(start), "end": str(end)}
    ohlcv_r = client.get(f"/api/ohlcv/{symbol}", params=params)
    if ohlcv_r.status_code != 200:
        st.error(ohlcv_r.text)
        st.stop()
    rows = ohlcv_r.json()
    if not rows:
        st.warning("Không có dữ liệu OHLCV trong khoảng ngày đã chọn (ví dụ ngày nghỉ / cuối tuần). Hãy đổi Start/End.")

    sr_r = client.get(f"/api/sr-zones/{symbol}")
    zones = sr_r.json() if sr_r.status_code == 200 else []

chart_zones = filter_zones_for_overlay(
    zones,
    rows,
    max_zones=int(max_sr_lines),
    min_strength=float(min_sr_strength),
)

fig = ohlcv_figure(rows, title=f"{symbol} OHLCV")
if chart_zones:
    add_sr_overlay(fig, chart_zones)
st.plotly_chart(fig, use_container_width=True)

if zones:
    n_active = sum(1 for z in zones if z.get("is_active", True))
    st.caption(
        f"Chart: **{len(chart_zones)}** zone(s) after filtering (min strength {min_sr_strength:g}, "
        f"max {max_sr_lines} lines, near visible price range). "
        f"API returned **{len(zones)}** row(s), **{n_active}** active."
    )
    st.subheader("S/R zones on chart")
    st.dataframe(chart_zones, use_container_width=True)
    with st.expander("All zones from API"):
        st.dataframe(zones, use_container_width=True)
