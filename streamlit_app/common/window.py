"""Pick a 30-session window: low-level math + the shared Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx
import streamlit as st

WINDOW_LEN = 30


def end_index_for_calendar_day(times: list[datetime], end_day: date) -> int:
    """Largest index ``j >= WINDOW_LEN-1`` whose session calendar day is ``<= end_day``."""
    for cand in range(len(times) - 1, WINDOW_LEN - 2, -1):
        if times[cand].date() <= end_day:
            return cand
    return WINDOW_LEN - 1


def start_index_from_end_index(j: int) -> int:
    return j - (WINDOW_LEN - 1)


def window_end_date_bounds(times: list[datetime]) -> tuple[date, date]:
    """Earliest / latest valid end *dates* for a ``WINDOW_LEN``-bar window within ``times``."""
    return times[WINDOW_LEN - 1].date(), times[-1].date()


@dataclass(frozen=True)
class WindowSelection:
    """Result of :func:`pick_30_session_window`: all the derived data a page needs."""

    bars: list[dict[str, Any]]
    times: list[datetime]
    start_i: int
    window_end: datetime
    current_price: float


def hist_sessions_slider(
    *,
    label: str = "Lịch sử tải (phiên)",
    default: int = 480,
    help_text: str | None = None,
) -> int:
    """Sidebar slider controlling how many recent bars to load from the API."""
    return int(
        st.sidebar.slider(
            label,
            min_value=60,
            max_value=5000,
            value=default,
            step=40,
            help=help_text
            or "Số nến gần nhất nạp từ DB (càng lớn càng kéo được cửa sổ 30 phiên về quá khứ xa hơn).",
        )
    )


def _parse_bar_time(raw: object) -> datetime:
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def pick_30_session_window(
    client: httpx.Client,
    symbol: str,
    hist_sessions: int,
    *,
    end_caption_fmt: str = "Đủ 30 phiên giao dịch: **{start}** UTC → **{end}** UTC",
) -> WindowSelection | None:
    """Fetch the latest ``hist_sessions`` bars, render the date picker, return the selection.

    Returns ``None`` and calls :func:`streamlit.stop` when the request fails or fewer than
    30 sessions are available. On the happy path the sidebar caption summarising the loaded
    range is also rendered.
    """
    latest = client.get(f"/api/ohlcv/{symbol}/latest", params={"n": hist_sessions})
    if latest.status_code != 200:
        st.error(latest.text)
        st.stop()
    bars = latest.json()
    if len(bars) < WINDOW_LEN:
        st.warning(f"Need at least {WINDOW_LEN} sessions.")
        st.stop()

    times = [_parse_bar_time(b["time"]) for b in bars]
    st.sidebar.caption(
        f"Dữ liệu đã tải: **{times[0]:%Y-%m-%d}** → **{times[-1]:%Y-%m-%d}** "
        f"({len(times)} phiên, UTC)"
    )

    min_end_d, max_end_d = window_end_date_bounds(times)
    end_pick = st.date_input(
        "Ngày kết thúc cửa sổ (30 phiên)",
        value=max_end_d,
        min_value=min_end_d,
        max_value=max_end_d,
        help=(
            "Ngày theo lịch của **phiên cuối** trong 30 phiên (UTC). "
            "Không có phiên đúng ngày thì lấy phiên gần nhất trước đó."
        ),
    )
    j_end = end_index_for_calendar_day(times, end_pick)
    start_i = start_index_from_end_index(j_end)
    end_i = start_i + WINDOW_LEN - 1

    col_a, col_b = st.columns(2)
    col_a.metric("Phiên đầu window", times[start_i].strftime("%Y-%m-%d"))
    col_b.metric("Phiên cuối window", times[end_i].strftime("%Y-%m-%d"))
    st.caption(
        end_caption_fmt.format(
            start=times[start_i].strftime("%Y-%m-%d %H:%M"),
            end=times[end_i].strftime("%Y-%m-%d %H:%M"),
        )
    )

    return WindowSelection(
        bars=bars,
        times=times,
        start_i=start_i,
        window_end=times[end_i],
        current_price=float(bars[end_i]["close"]),
    )
