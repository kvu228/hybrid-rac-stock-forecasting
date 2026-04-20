"""RAC full context + optional predict."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import plotly.graph_objects as go
import streamlit as st

from streamlit_app.common import (
    api_base_url_input,
    hist_sessions_slider,
    http_client,
    label_text,
    pick_30_session_window,
    pick_symbol,
    render_label_pie,
)


st.set_page_config(page_title="RAC prediction", layout="wide")
st.title("RAC prediction (hybrid context)")

api = api_base_url_input()
k = st.sidebar.slider("K neighbors", 5, 50, 20)
hist_sessions = hist_sessions_slider(
    help_text="Số nến gần nhất để chọn cửa sổ 30 phiên (kéo sâu về quá khứ).",
)

with http_client(api) as client:
    symbol = pick_symbol(client)
    window = pick_30_session_window(
        client,
        symbol,
        hist_sessions,
        end_caption_fmt="Đủ 30 phiên: **{start}** UTC → **{end}** UTC",
    )
    assert window is not None
    current_price = window.current_price

    qe = client.post(
        "/api/rac/query-embedding",
        json={"symbol": symbol, "window_end": window.window_end.isoformat()},
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

st.caption(
    f"Cửa sổ embedding (API): **{qemb.get('window_start')}** … **{qemb.get('window_end')}** — "
    f"giá đóng cuối window dùng cho hybrid: **{current_price:.4f}**"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Neighbors", ctx.get("total_neighbors", 0))
c2.metric("Avg cosine dist", f"{ctx.get('avg_cosine_dist') or 0:.4f}")
c3.metric("Dominant label", label_text(ctx.get("dominant_label")))
c4.metric("KNN confidence", f"{(ctx.get('knn_confidence') or 0) * 100:.1f}%")

ld = ctx.get("label_distribution") or {}
nld = ctx.get("neighbor_label_distances") or []
if ld or nld:
    render_label_pie(
        neighbors=nld,
        count_distribution=ld,
        count_mode_label="Đếm đều — theo label_distribution (DB)",
        weighted_mode_label="Trọng số theo cos distance — cùng tập KNN với full-context",
        weighted_title="Label distribution (KNN, trọng số 1/(cos_dist+0.02))",
        weighted_caption=(
            "Cùng embedding + **k** với bảng metrics; trọng số ưu tiên láng giềng có vector "
            "**gần query** hơn."
        ),
        radio_help="Weighted dùng `neighbor_label_distances` từ API (w = 1/(dist+0.02)), khớp thứ tự KNN.",
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

if st.button("Run predict (persist)"):
    with http_client(api) as client:
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
            st.success(
                f"**{label_text(pj.get('predicted_label'))}** (raw {pj.get('predicted_label')!r}), "
                f"confidence {pj.get('confidence_score')!s}"
            )
            with st.expander("Raw JSON"):
                st.json(pj)
        else:
            st.error(pr.text)
