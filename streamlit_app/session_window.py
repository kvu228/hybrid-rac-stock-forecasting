"""Pick a 30-session window by the calendar day of the last bar (UTC date)."""

from __future__ import annotations

from datetime import date, datetime


def end_index_for_calendar_day(times: list[datetime], end_day: date) -> int:
    """Largest index j >= 29 with session calendar day <= end_day."""
    for cand in range(len(times) - 1, 28, -1):
        if times[cand].date() <= end_day:
            return cand
    return 29


def start_index_from_end_index(j: int) -> int:
    return j - 29


def window_end_date_bounds(times: list[datetime]) -> tuple[date, date]:
    """Earliest / latest valid end *dates* for a 30-bar window within ``times``."""
    return times[29].date(), times[-1].date()
