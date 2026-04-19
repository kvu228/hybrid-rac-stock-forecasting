"""Similar patterns: query embedding + KNN + overlay."""

from __future__ import annotations

import os
import sys
from collections import Counter
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
from streamlit_app.session_window import (
    end_index_for_calendar_day,
    start_index_from_end_index,
    window_end_date_bounds,
)

# Matches etl/feature_engineer.py window labels (T+5 forward return buckets).
_LABEL_NAMES: dict[int, str] = {0: "Down", 1: "Neutral", 2: "Up"}


def _label_text(value: object) -> str:
    if value is None:
        return "—"
    try:
        key = int(value)
    except (TypeError, ValueError):
        return str(value)
    return _LABEL_NAMES.get(key, str(key))


def _label_distribution_counts(neighbors: list[dict[str, object]]) -> dict[str, float]:
    """Mỗi neighbor một đơn vị (phân bố đếm thuần)."""
    c: Counter[str] = Counter()
    for n in neighbors:
        c[_label_text(n.get("label"))] += 1.0
    return dict(c)


def _label_distribution_weighted(neighbors: list[dict[str, object]], *, eps: float = 0.02) -> dict[str, float]:
    """Trọng số ~ nghịch đảo khoảng cách: láng giềng gần (cos dist nhỏ) đóng góp nhiều hơn."""
    acc: dict[str, float] = {}
    for n in neighbors:
        d = float(n.get("cosine_distance", 0.0))
        w = 1.0 / (max(d, 0.0) + eps)
        lab = _label_text(n.get("label"))
        acc[lab] = acc.get(lab, 0.0) + w
    return acc


st.set_page_config(page_title="Similar patterns", layout="wide")
st.title("Similar patterns (pgvector KNN)")

api = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"),
).rstrip("/")

k = st.sidebar.slider("K neighbors", 5, 50, 20)
# DB filter: cosine_distance < (1 − t). High t = very strict (few/no neighbors).
similarity_threshold = st.sidebar.slider(
    "Similarity threshold (t)",
    min_value=0.35,
    max_value=0.98,
    value=0.72,
    step=0.01,
    help="Càng cao càng khớp chặt (chỉ giữ neighbor có cos distance < 1−t). Nếu không có kết quả, hạ t xuống.",
)
hist_sessions = st.sidebar.slider(
    "Lịch sử tải (phiên)",
    min_value=60,
    max_value=5000,
    value=480,
    step=40,
    help="Số nến gần nhất nạp từ DB (càng lớn càng kéo được cửa sổ 30 phiên về quá khứ xa hơn).",
)

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

    latest = client.get(f"/api/ohlcv/{symbol}/latest", params={"n": int(hist_sessions)})
    if latest.status_code != 200:
        st.error(latest.text)
        st.stop()
    bars = latest.json()
    if len(bars) < 30:
        st.warning("Need at least 30 sessions.")
        st.stop()
    times = [datetime.fromisoformat(str(b["time"]).replace("Z", "+00:00")) for b in bars]
    st.sidebar.caption(
        f"Dữ liệu đã tải: **{times[0].strftime('%Y-%m-%d')}** → **{times[-1].strftime('%Y-%m-%d')}** "
        f"({len(times)} phiên, UTC)"
    )

    min_end_d, max_end_d = window_end_date_bounds(times)
    end_pick = st.date_input(
        "Ngày kết thúc cửa sổ (30 phiên)",
        value=max_end_d,
        min_value=min_end_d,
        max_value=max_end_d,
        help="Ngày theo lịch của **phiên cuối** trong 30 phiên (UTC). Không có phiên đúng ngày thì lấy phiên gần nhất trước đó.",
    )
    j_end = end_index_for_calendar_day(times, end_pick)
    start_i = start_index_from_end_index(j_end)
    c_a, c_b = st.columns(2)
    c_a.metric("Phiên đầu window", times[start_i].strftime("%Y-%m-%d"))
    c_b.metric("Phiên cuối window", times[start_i + 29].strftime("%Y-%m-%d"))
    st.caption(
        f"Đủ 30 phiên giao dịch: **{times[start_i].strftime('%Y-%m-%d %H:%M')}** UTC → "
        f"**{times[start_i + 29].strftime('%Y-%m-%d %H:%M')}** UTC"
    )
    window_end = times[start_i + 29]

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
            "similarity_threshold": similarity_threshold,
            "filter_symbol": None,
        },
    )
    if sim.status_code != 200:
        st.error(sim.text)
        st.stop()
    neighbors = sim.json().get("neighbors", [])

ws_api = qpayload.get("window_start")
we_api = qpayload.get("window_end")
st.caption(
    f"Cửa sổ query (API): **{ws_api}** … **{we_api}** — "
    f"KNN {sim.json().get('query_time_ms', 0):.2f} ms — "
    f"t={similarity_threshold:g} ⇒ cos distance < {1.0 - similarity_threshold:.3f}"
)

query_ohlcv = qpayload.get("ohlcv") or []
if query_ohlcv:
    st.subheader("Query window (30 sessions)")
    st.plotly_chart(
        ohlcv_figure(query_ohlcv, title=f"{symbol} query"),
        use_container_width=True,
    )

if not neighbors:
    st.warning(
        "Không có neighbor nào thỏa điều kiện. Thử **hạ Similarity threshold** trên sidebar "
        "(ví dụ 0.65–0.75), hoặc kiểm tra bảng `pattern_embeddings` đã có dữ liệu và index HNSW."
    )

closes_query = [float(b["close"]) for b in query_ohlcv]
series_list = [np.array(closes_query, dtype=np.float64)] if len(closes_query) == 30 else []
labels_overlay = ["query"] if series_list else []

neighbor_chart_options: list[dict[str, object]] = []
for n in neighbors[:8]:
    sym_n = str(n["symbol"])
    ws = n["window_start"]
    we = n["window_end"]
    r = httpx.get(
        f"{api}/api/ohlcv/{sym_n}",
        params={"start": str(ws)[:10], "end": str(we)[:10]},
        timeout=60.0,
    )
    if r.status_code != 200:
        continue
    ohr = r.json()
    if len(ohr) >= 30:
        ohr = sorted(ohr, key=lambda x: x["time"])[-30:]
    if not ohr:
        continue
    dist = float(n.get("cosine_distance", 0.0))
    lt = _label_text(n.get("label"))
    menu = f"#{n['id']} {sym_n} · {lt} · dist {dist:.4f}"
    neighbor_chart_options.append(
        {
            "menu": menu,
            "ohlcv": ohr,
            "title": f"#{n['id']} {sym_n} · {lt} · cos dist {dist:.4f}",
        }
    )
    cl = [float(x["close"]) for x in ohr[-30:]]
    if len(cl) == 30:
        series_list.append(np.array(cl, dtype=np.float64))
        labels_overlay.append(f"{sym_n} #{n['id']} ({lt})")

if neighbors:
    st.subheader("Nearest neighbors")
    if neighbor_chart_options:
        pick = st.selectbox(
            "Chọn neighbor để xem OHLCV (30 phiên)",
            options=range(len(neighbor_chart_options)),
            format_func=lambda i: str(neighbor_chart_options[i]["menu"]),
            index=0,
        )
        chosen = neighbor_chart_options[int(pick)]
        st.plotly_chart(
            ohlcv_figure(chosen["ohlcv"], title=str(chosen["title"])),
            use_container_width=True,
        )
    else:
        st.info("Không tải được OHLCV cho neighbor nào trong top-8 (kiểm tra API hoặc khoảng window).")

if len(series_list) > 1:
    st.subheader("Close curves (min–max từng chuỗi, mỗi hàng một series)")
    st.caption("Mỗi đường nằm subplot riêng để tránh chồng lên nhau; trục Y 0–1 cho từng cửa sổ.")
    st.plotly_chart(
        overlay_figure(series_list, labels_overlay, title=""),
        use_container_width=True,
    )

if neighbors:
    st.subheader("Neighbors")
    neighbor_rows = [{**n, "label": _label_text(n.get("label"))} for n in neighbors]
    st.dataframe(neighbor_rows, use_container_width=True)

    pie_mode = st.radio(
        "Phân bố nhãn (pie)",
        options=("count", "weighted"),
        format_func=lambda m: (
            "Đếm đều — mỗi neighbor một phiếu"
            if m == "count"
            else "Trọng số theo cos distance — gần hơn (dist nhỏ) nặng hơn"
        ),
        horizontal=True,
        help="Weighted: w = 1/(dist+ε), hợp lý nếu bạn tin láng giềng rất gần đáng tin hơn (giống KNN có trọng số).",
    )
    if pie_mode == "count":
        pie_values = _label_distribution_counts(neighbors)
        pie_title = "Label distribution (neighbors, đếm đều)"
    else:
        pie_values = _label_distribution_weighted(neighbors)
        pie_title = "Label distribution (neighbors, trọng số 1/(cos_dist+0.02))"
    fig_pie = px.pie(
        names=list(pie_values.keys()),
        values=list(pie_values.values()),
        title=pie_title,
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    if pie_mode == "weighted":
        st.caption(
            "Tỷ lệ trên biểu đồ là **phần trọng số** theo nhãn (tổng các w đã chuẩn hoá thành 100%). "
            "Khác với đếm thuần: một neighbor Down rất gần có thể “kéo” phần Down lớn hơn một neighbor Up xa."
        )
