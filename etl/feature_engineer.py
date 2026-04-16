"""Feature engineering: normalization, windowing, and labeling for pattern embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

OHLCV_CHANNELS = ["open", "high", "low", "close", "volume"]
WINDOW_SIZE = 30
LABEL_HORIZON = 5  # T+5 forward return for labeling


@dataclass(frozen=True)
class WindowRecord:
    """Metadata + normalized data for a single sliding window."""

    symbol: str
    window_start: datetime
    window_end: datetime
    label: int  # 0=Down, 1=Neutral, 2=Up
    future_return: float  # raw T+5 close-to-close %
    data: np.ndarray  # shape (WINDOW_SIZE, 5) — z-score normalized OHLCV


def forward_fill_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill gaps in per-symbol OHLCV so every trading day has a row.

    Uses a business-day calendar (Mon-Fri) to fill gaps.  Price columns are
    forward-filled; volume is filled with 0 (no trading on missing days).

    Expects *sorted* input with columns: time, symbol, open, high, low, close, volume.
    """
    if df.empty:
        return df

    frames: list[pd.DataFrame] = []
    for symbol, grp in df.groupby("symbol", sort=False):
        grp = grp.set_index("time").sort_index()
        bday_idx = pd.bdate_range(start=grp.index.min(), end=grp.index.max(), freq="B")
        grp = grp.reindex(bday_idx)
        grp["symbol"] = symbol
        for col in ["open", "high", "low", "close"]:
            grp[col] = grp[col].ffill()
        grp["volume"] = grp["volume"].fillna(0).astype("int64")
        grp = grp.dropna(subset=["close"])
        grp = grp.reset_index().rename(columns={"index": "time"})
        frames.append(grp)

    if not frames:
        return df.iloc[:0]

    return pd.concat(frames, ignore_index=True)


def relative_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``close_ret`` column: session-over-session close-to-close % return.

    Computed per symbol.  First row of each symbol gets 0.0.
    """
    out = df.copy()
    out["close_ret"] = out.groupby("symbol")["close"].pct_change().fillna(0.0)
    return out


def zscore_normalize_window(window: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel (column) independently within a window.

    Args:
        window: shape (WINDOW_SIZE, n_channels)

    Returns:
        Normalized array of the same shape.  Constant channels (std=0) are
        zeroed out.
    """
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for constant channels
    return (window - mean) / std


def _compute_label(future_return: float, up_thresh: float, down_thresh: float) -> int:
    """Map a forward return to a class label.

    Returns:
        0 — Down  (return <= down_thresh)
        1 — Neutral
        2 — Up    (return >= up_thresh)
    """
    if future_return >= up_thresh:
        return 2
    if future_return <= down_thresh:
        return 0
    return 1


def generate_windows(
    df: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
    horizon: int = LABEL_HORIZON,
    stride: int = 1,
    up_threshold: float = 0.02,
    down_threshold: float = -0.02,
) -> list[WindowRecord]:
    """Create labeled sliding windows from cleaned OHLCV data.

    For each symbol the function slides a ``window_size``-session window with
    the given ``stride`` and computes a ``horizon``-day forward return from the
    last close in the window.  Each window's OHLCV matrix is z-score
    normalized independently.

    Args:
        df: Cleaned OHLCV DataFrame (must contain time, symbol, OHLCV cols).
        window_size: Number of sessions per window.
        horizon: Number of sessions ahead for the label return.
        stride: Step size between consecutive windows.
        up_threshold: Forward return >= this → label 2 (Up).
        down_threshold: Forward return <= this → label 0 (Down).

    Returns:
        List of ``WindowRecord`` with normalized data + label.
    """
    records: list[WindowRecord] = []

    for symbol, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("time").reset_index(drop=True)
        closes = grp["close"].values
        times = grp["time"].values
        ohlcv = grp[OHLCV_CHANNELS].values  # (N, 5)

        n = len(grp)
        max_start = n - window_size - horizon
        if max_start < 0:
            continue

        for i in range(0, max_start + 1, stride):
            w_end = i + window_size  # exclusive for slice, last index = w_end-1
            future_idx = w_end + horizon - 1

            last_close = closes[w_end - 1]
            if last_close == 0:
                continue
            future_return = (closes[future_idx] - last_close) / last_close

            label = _compute_label(future_return, up_threshold, down_threshold)
            raw_window = ohlcv[i:w_end].astype(np.float64)
            normed = zscore_normalize_window(raw_window)

            records.append(
                WindowRecord(
                    symbol=str(symbol),
                    window_start=pd.Timestamp(times[i]).to_pydatetime(),
                    window_end=pd.Timestamp(times[w_end - 1]).to_pydatetime(),
                    label=label,
                    future_return=float(future_return),
                    data=normed,
                )
            )

    return records


def train_test_split_by_time(
    records: list[WindowRecord],
    train_ratio: float = 0.8,
) -> tuple[list[WindowRecord], list[WindowRecord]]:
    """Split window records chronologically (NOT random).

    Windows are sorted by ``window_end`` across all symbols, then the first
    ``train_ratio`` fraction goes to the train set and the rest to test.

    Returns:
        (train_records, test_records)
    """
    if not records:
        return [], []

    sorted_recs = sorted(records, key=lambda r: r.window_end)
    split_idx = int(len(sorted_recs) * train_ratio)
    return sorted_recs[:split_idx], sorted_recs[split_idx:]