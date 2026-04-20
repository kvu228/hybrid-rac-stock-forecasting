"""Similar patterns: query embedding + KNN + overlay."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import httpx
import numpy as np
import streamlit as st

from streamlit_app.common import (
    api_base_url_input,
    count_label_distribution,
    hist_sessions_slider,
    http_client,
    label_text,
    pick_30_session_window,
    pick_symbol,
    render_label_pie,
)
from streamlit_app.components.candlestick import ohlcv_figure
from streamlit_app.components.similarity import close_curves_faceted


st.set_page_config(page_title="Similar patterns", layout="wide")
st.title("Similar patterns (pgvector KNN)")

api = api_base_url_input()
k = st.sidebar.slider("K neighbors", 5, 50, 20)
# DB filter: cosine_distance < (1 − t). High t = very strict (few/no neighbors).
similarity_threshold = st.sidebar.slider(
    "Similarity threshold (t)",
    min_value=0.35,
    max_value=0.98,
    value=0.72,
    step=0.01,
    help=(
        "Càng cao càng khớp chặt (chỉ giữ neighbor có cos distance < 1−t). "
        "Nếu không có kết quả, hạ t xuống."
    ),
)
hist_sessions = hist_sessions_slider(
    help_text="Số nến gần nhất nạp từ DB (càng lớn càng kéo được cửa sổ 30 phiên về quá khứ xa hơn).",
)

with http_client(api) as client:
    symbol = pick_symbol(client)
    window = pick_30_session_window(client, symbol, hist_sessions)
    assert window is not None  # pick_30_session_window st.stop()s on failure

    qe = client.post(
        "/api/rac/query-embedding",
        json={"symbol": symbol, "window_end": window.window_end.isoformat()},
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
    sim_payload = sim.json()
    neighbors = sim_payload.get("neighbors", [])

st.caption(
    f"Cửa sổ query (API): **{qpayload.get('window_start')}** … **{qpayload.get('window_end')}** — "
    f"KNN {sim_payload.get('query_time_ms', 0):.2f} ms — "
    f"t={similarity_threshold:g} ⇒ cos distance < {1.0 - similarity_threshold:.3f}"
)

query_ohlcv = qpayload.get("ohlcv") or []
if query_ohlcv:
    st.subheader("Query window (30 sessions)")
    st.plotly_chart(ohlcv_figure(query_ohlcv, title=f"{symbol} query"), use_container_width=True)

if not neighbors:
    st.warning(
        "Không có neighbor nào thỏa điều kiện. Thử **hạ Similarity threshold** trên sidebar "
        "(ví dụ 0.65–0.75), hoặc kiểm tra bảng `pattern_embeddings` đã có dữ liệu và index HNSW."
    )

closes_query = [float(b["close"]) for b in query_ohlcv]
series_list: list[np.ndarray] = (
    [np.array(closes_query, dtype=np.float64)] if len(closes_query) == 30 else []
)
labels_overlay: list[str] = ["query"] if series_list else []

neighbor_chart_options: list[dict[str, object]] = []
for n in neighbors[:8]:
    sym_n = str(n["symbol"])
    ws = str(n["window_start"])[:10]
    we = str(n["window_end"])[:10]
    r = httpx.get(f"{api}/api/ohlcv/{sym_n}", params={"start": ws, "end": we}, timeout=60.0)
    if r.status_code != 200:
        continue
    ohr = r.json()
    if not ohr:
        continue
    ohr = sorted(ohr, key=lambda x: x["time"])[-30:]
    dist = float(n.get("cosine_distance", 0.0))
    lt = label_text(n.get("label"))
    menu = f"#{n['id']} {sym_n} · {lt} · dist {dist:.4f}"
    neighbor_chart_options.append(
        {
            "menu": menu,
            "ohlcv": ohr,
            "title": f"#{n['id']} {sym_n} · {lt} · cos dist {dist:.4f}",
        }
    )
    closes_n = [float(x["close"]) for x in ohr]
    if len(closes_n) == 30:
        series_list.append(np.array(closes_n, dtype=np.float64))
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
        close_curves_faceted(series_list, labels_overlay),
        use_container_width=True,
    )

if neighbors:
    st.subheader("Neighbors")
    neighbor_rows = [{**n, "label": label_text(n.get("label"))} for n in neighbors]
    st.dataframe(neighbor_rows, use_container_width=True)

    render_label_pie(
        neighbors=neighbors,
        count_label_fn=lambda ns: count_label_distribution(ns or []),
        weighted_caption=(
            "Tỷ lệ trên biểu đồ là **phần trọng số** theo nhãn (tổng các w đã chuẩn hoá thành 100%). "
            "Khác với đếm thuần: một neighbor Down rất gần có thể “kéo” phần Down lớn hơn một neighbor Up xa."
        ),
    )
