"""Feature engineering: normalization, windowing, and labeling for pattern embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

OHLCV_CHANNELS = ["open", "high", "low", "close", "volume"]
# Feature channels used by encoder input (adds close_ret so the window carries
# explicit directional/momentum signal that z-score within-window doesn't erase).
FEATURE_CHANNELS = ["open", "high", "low", "close", "volume", "close_ret"]
WINDOW_SIZE = 30
LABEL_HORIZON = 10  # T+10 forward return — wider horizon reduces path-dependency noise in triple-barrier labels


@dataclass(frozen=True)
class WindowRecord:
    """Metadata + normalized data for a single sliding window."""

    symbol: str
    window_start: datetime
    window_end: datetime
    label: int  # 0=Down, 1=Neutral, 2=Up
    future_return: float  # raw T+5 close-to-close %
    data: np.ndarray  # shape (WINDOW_SIZE, n_channels) — z-score normalized features


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


def zscore_normalize_window(window: np.ndarray, channel_names: list[str] | None = None) -> np.ndarray:
    """Normalize channels within a window.

    Price columns (OHLC) are converted to percentage changes relative to the
    first 'open' price in the window, scaled by 100. This preserves both shape
    and ABSOLUTE volatility (e.g. 2.0 = +2% move).
    Volume is z-scored. 'close_ret' is scaled by 100.

    Args:
        window: shape (WINDOW_SIZE, n_channels)
        channel_names: List of channel names.

    Returns:
        Normalized array of the same shape.
    """
    x = np.asarray(window, dtype=np.float64)
    out = np.zeros_like(x)

    if channel_names is None:
        channel_names = ["open", "high", "low", "close", "volume", "close_ret"]

    price_indices = [i for i, c in enumerate(channel_names) if c in ["open", "high", "low", "close"]]

    for i, col_name in enumerate(channel_names):
        if col_name in ["open", "high", "low", "close"]:
            continue  # Handled below
        elif col_name == "volume":
            v = x[:, i]
            v_mean = np.nanmean(v)
            v_std = np.nanstd(v)
            if v_std == 0 or ~np.isfinite(v_std):
                v_std = 1.0
            out[:, i] = (v - v_mean) / v_std
        elif col_name == "close_ret" or "ret" in col_name:
            out[:, i] = x[:, i] * 100.0
        else:
            v = x[:, i]
            v_mean = np.nanmean(v)
            v_std = np.nanstd(v)
            if v_std == 0 or ~np.isfinite(v_std):
                v_std = 1.0
            out[:, i] = (v - v_mean) / v_std

    if price_indices:
        if "open" in channel_names:
            base_price = x[0, channel_names.index("open")]
        else:
            base_price = x[0, price_indices[0]]

        if base_price == 0 or ~np.isfinite(base_price):
            base_price = 1.0

        price_data = x[:, price_indices]
        out[:, price_indices] = ((price_data - base_price) / base_price) * 100.0

    out[~np.isfinite(out)] = 0.0
    return out


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
    up_threshold: float = 0.03,
    down_threshold: float = -0.03,
    channels: list[str] | tuple[str, ...] = tuple(FEATURE_CHANNELS),
) -> list[WindowRecord]:
    """Create labeled sliding windows from cleaned OHLCV data.

    For each symbol the function slides a ``window_size``-session window with
    the given ``stride`` and computes a ``horizon``-day forward return from the
    last close in the window.  Each window's feature matrix is z-score
    normalized independently.

    By default the feature matrix has 6 channels (OHLCV + close_ret). The
    ``close_ret`` channel is session-over-session percentage return; unlike
    the raw OHLCV it is already differenced, so z-scoring within the window
    preserves directional information (up-trend vs down-trend) and helps the
    encoder distinguish momentum.

    Pass ``channels=OHLCV_CHANNELS`` for the legacy 5-channel behavior.

    Args:
        df: Cleaned OHLCV DataFrame (must contain time, symbol, OHLCV cols).
        window_size: Number of sessions per window.
        horizon: Number of sessions ahead for the label return.
        stride: Step size between consecutive windows.
        up_threshold: Forward return >= this → label 2 (Up).
        down_threshold: Forward return <= this → label 0 (Down).
        channels: Columns to include as feature channels.

    Returns:
        List of ``WindowRecord`` with normalized data + label. ``data`` has
        shape ``(window_size, len(channels))``.
    """
    channel_list = list(channels)
    records: list[WindowRecord] = []
    
    for symbol, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("time").reset_index(drop=True)
        # Compute close_ret on demand if requested and not already present.
        if "close_ret" in channel_list and "close_ret" not in grp.columns:
            cr = grp["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
            grp = grp.assign(close_ret=cr)
        closes = grp["close"].values
        highs = grp["high"].values
        lows = grp["low"].values
        times = grp["time"].values
        feature_matrix = grp[channel_list].values  # (N, C)

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
            
            # Triple-Barrier Labeling
            label = 1  # Default to Neutral (Time stop)
            for j in range(w_end, future_idx + 1):
                # Check pessimistic case first: assume low hits stop-loss before high hits take-profit in the same session
                low_ret = (lows[j] - last_close) / last_close
                high_ret = (highs[j] - last_close) / last_close
                
                if low_ret <= down_threshold:
                    label = 0  # Down (Stop-loss hit)
                    break
                elif high_ret >= up_threshold:
                    label = 2  # Up (Take-profit hit)
                    break
            
            # Keep future_return as the point-in-time return for logging/metadata
            future_return = (closes[future_idx] - last_close) / last_close

            raw_window = feature_matrix[i:w_end].astype(np.float64)
            normed = zscore_normalize_window(raw_window, channel_names=channel_list)

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