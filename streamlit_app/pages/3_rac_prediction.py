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

from streamlit_app.session_window import (
    end_index_for_calendar_day,
    start_index_from_end_index,
    window_end_date_bounds,
)

_LABEL_NAMES: dict[int, str] = {0: "Down", 1: "Neutral", 2: "Up"}


def _label_text(value: object) -> str:
    if value is None:
        return "—"
    try:
        key = int(value)
    except (TypeError, ValueError):
        return str(value)
    return _LABEL_NAMES.get(key, str(key))


def _rac_pie_counts(label_distribution: dict[str, object]) -> dict[str, float]:
    return {_label_text(k): float(v) for k, v in label_distribution.items()}


def _rac_pie_weighted(neighbor_rows: list[dict[str, object]], *, eps: float = 0.02) -> dict[str, float]:
    acc: dict[str, float] = {}
    for r in neighbor_rows:
        d = float(r.get("cosine_distance", 0.0))
        w = 1.0 / (max(d, 0.0) + eps)
        lab = _label_text(r.get("label"))
        acc[lab] = acc.get(lab, 0.0) + w
    return acc


st.set_page_config(page_title="RAC prediction", layout="wide")
st.title("RAC prediction (hybrid context)")

api = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"),
).rstrip("/")
k = st.sidebar.slider("K neighbors", 5, 50, 20)
hist_sessions = st.sidebar.slider(
    "Lịch sử tải (phiên)",
    min_value=60,
    max_value=5000,
    value=480,
    step=40,
    help="Số nến gần nhất để chọn cửa sổ 30 phiên (kéo sâu về quá khứ).",
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
        f"Đủ 30 phiên: **{times[start_i].strftime('%Y-%m-%d %H:%M')}** UTC → "
        f"**{times[start_i + 29].strftime('%Y-%m-%d %H:%M')}** UTC"
    )
    window_end = times[start_i + 29]
    current_price = float(bars[start_i + 29]["close"])

    qe = client.post(
        "/api/rac/query-embedding",
        json={"symbol": symbol, "window_end": window_end.isoformat()},
    )
    if qe.status_code != 200:
        st.error(qe.text)
        st.stop()
    qemb = qe.json()
    emb = qemb["query_embedding"]

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

ws_api = qemb.get("window_start")
we_api = qemb.get("window_end")
st.caption(f"Cửa sổ embedding (API): **{ws_api}** … **{we_api}** — giá đóng cuối window dùng cho hybrid: **{current_price:.4f}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Neighbors", ctx.get("total_neighbors", 0))
c2.metric("Avg cosine dist", f"{ctx.get('avg_cosine_dist') or 0:.4f}")
c3.metric("Dominant label", _label_text(ctx.get("dominant_label")))
c4.metric("KNN confidence", f"{(ctx.get('knn_confidence') or 0) * 100:.1f}%")

ld = ctx.get("label_distribution") or {}
nld = ctx.get("neighbor_label_distances") or []
if ld or nld:
    pie_mode = st.radio(
        "Phân bố nhãn (pie)",
        options=("count", "weighted"),
        format_func=lambda m: (
            "Đếm đều — theo label_distribution (DB)"
            if m == "count"
            else "Trọng số theo cos distance — cùng tập KNN với full-context"
        ),
        horizontal=True,
        help="Weighted dùng `neighbor_label_distances` từ API (w = 1/(dist+0.02)), khớp thứ tự KNN.",
    )
    if pie_mode == "weighted" and nld:
        pie_values = _rac_pie_weighted(nld)
        pie_title = "Label distribution (KNN, trọng số 1/(cos_dist+0.02))"
    else:
        pie_values = _rac_pie_counts(ld) if ld else {}
        pie_title = "Label distribution (neighbors, đếm đều)"
    if pie_values:
        st.plotly_chart(
            px.pie(
                names=list(pie_values.keys()),
                values=list(pie_values.values()),
                title=pie_title,
            ),
            use_container_width=True,
        )
        if pie_mode == "weighted" and nld:
            st.caption(
                "Cùng embedding + **k** với bảng metrics; trọng số ưu tiên láng giềng có vector **gần query** hơn."
            )

st.subheader("Support / Resistance (hybrid)")
ds = ctx.get("dist_to_support")
dr = ctx.get("dist_to_resistance")
ratio = ctx.get("sr_position_ratio")
c_sr1, c_sr2, c_sr3 = st.columns(3)
c_sr1.metric("Khoảng cách tới support (giá)", f"{ds:.4f}" if ds is not None else "—")
c_sr2.metric("Khoảng cách tới resistance (giá)", f"{dr:.4f}" if dr is not None else "—")
c_sr3.metric("S/R position ratio (0–1)", f"{float(ratio):.4f}" if ratio is not None else "—")

with st.expander("Chú thích: S/R position ratio là gì?", expanded=False):
    st.markdown(
        """
        Giá dùng cho hybrid là **đóng cửa phiên cuối** của cửa sổ 30 phiên (xem caption phía trên).

        - **`dist_to_support`**: khoảng cách tuyệt đối tới **mức support active gần nhất** trong bảng `support_resistance_zones`.
        - **`dist_to_resistance`**: tương tự tới **resistance** active gần nhất.
        - **`sr_position_ratio`** (hàm SQL `compute_full_rac_context`, migration 0007):  
          `d_support / (d_support + d_resistance)` khi cả hai khoảng cách đều có và tổng dương.

        **Cách đọc gauge (%):** hiển thị `ratio × 100`.

        - Gần **0%** → giá **nghiêng về phía support** (gần support hơn so với resistance theo tỷ lệ hai khoảng cách).
        - Gần **100%** → giá **nghiêng về phía resistance**.
        - **Khoảng 50%** → giữa hai mức theo *tỷ lệ khoảng cách*, không hẳn là “giữa sàn” theo nghĩa địa lý.

        Nếu thiếu zone một bên hoặc không có S/R active, ratio có thể **NULL**; gauge khi đó hiển thị 0 chỉ là giá trị mặc định trên UI.
        """
    )

fig_g = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=float(ratio) * 100 if ratio is not None else 0.0,
        title={"text": "S/R position ratio (%)"},
        gauge={"axis": {"range": [0, 100]}},
    )
)
fig_g.update_layout(height=280)
st.plotly_chart(fig_g, use_container_width=True)
st.caption("0% ≈ gần support hơn, 100% ≈ gần resistance hơn (theo công thức trên).")

st.subheader("Evidence (minh chứng KNN)")
nids = ctx.get("neighbor_ids") or []
st.write("**neighbor_ids** (thứ tự gần → xa theo embedding):", nids)
with st.expander("Chú thích: Evidence / neighbor_ids là gì?", expanded=False):
    st.markdown(
        """
        Đây là danh sách **`id` trong bảng `pattern_embeddings`**: các **cửa sổ mẫu lịch sử** mà pgvector chọn làm
        **K láng giềng gần nhất** so với vector truy vấn hiện tại (cosine distance nhỏ nhất trước).

        - Thứ tự trong mảng ≈ thứ tự **từ giống nhất đến kém giống hơn** (khớp `ORDER BY embedding <=> query` trong `compute_full_rac_context`).
        - Các ID này được đưa thêm vào **vector đặc trưng hybrid** khi chạy `predict` (cùng KNN stats, S/R ratio, …).
        - Để **xem biểu đồ** từng láng giềng: mở trang **Similar patterns**, cùng symbol và **kết thúc cửa sổ** như trên, rồi chọn từng neighbor trong dropdown.

        Không phải mã chứng khoán: đây là **khóa chính nội bộ** của embedding đã lưu trong DB.
        """
    )

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
            pj = pr.json()
            pl = _label_text(pj.get("predicted_label"))
            st.success(
                f"**{pl}** (raw {pj.get('predicted_label')!r}), "
                f"confidence {pj.get('confidence_score')!s}"
            )
            with st.expander("Raw JSON"):
                st.json(pj)
        else:
            st.error(pr.text)
