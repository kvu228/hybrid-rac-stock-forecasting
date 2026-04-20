"""Shared pie chart for label distributions (count vs. distance-weighted)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import plotly.express as px
import streamlit as st

from streamlit_app.common.labels import (
    remap_label_keys,
    weighted_label_distribution,
)

Mode = Literal["count", "weighted"]


def render_label_pie(
    *,
    neighbors: Sequence[Mapping[str, object]] | None = None,
    count_distribution: Mapping[object, object] | None = None,
    count_label_fn: Callable[[Sequence[Mapping[str, object]] | None], dict[str, float]] | None = None,
    count_mode_label: str = "Đếm đều — mỗi neighbor một phiếu",
    weighted_mode_label: str = "Trọng số theo cos distance — gần hơn (dist nhỏ) nặng hơn",
    count_title: str = "Label distribution (neighbors, đếm đều)",
    weighted_title: str = "Label distribution (neighbors, trọng số 1/(cos_dist+0.02))",
    weighted_caption: str | None = None,
    radio_key: str | None = None,
    radio_help: str = (
        "Weighted: w = 1/(dist+ε), hợp lý nếu bạn tin láng giềng rất gần "
        "đáng tin hơn (giống KNN có trọng số)."
    ),
) -> Mode:
    """Render the mode selector + pie chart and return the active mode.

    Either ``count_distribution`` (a ``{label_code: count}`` mapping from the API) or a
    ``count_label_fn`` callable must be provided for the count mode; ``neighbors`` powers
    the weighted mode (and the default count behaviour when ``count_label_fn`` handles it).
    """
    mode: Mode = st.radio(
        "Phân bố nhãn (pie)",
        options=("count", "weighted"),
        format_func=lambda m: count_mode_label if m == "count" else weighted_mode_label,
        horizontal=True,
        help=radio_help,
        key=radio_key,
    )  # type: ignore[assignment]

    if mode == "weighted":
        values = weighted_label_distribution(neighbors or [])
        title = weighted_title
    elif count_label_fn is not None:
        values = count_label_fn(neighbors)
        title = count_title
    else:
        values = remap_label_keys(count_distribution or {})
        title = count_title

    if values:
        st.plotly_chart(
            px.pie(names=list(values.keys()), values=list(values.values()), title=title),
            use_container_width=True,
        )
        if mode == "weighted" and weighted_caption:
            st.caption(weighted_caption)

    return mode
