"""Build a 30-session normalized window + CNN embedding for RAC queries (API / Streamlit)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from etl.feature_engineer import (
    FEATURE_CHANNELS,
    WINDOW_SIZE,
    _compute_bb_pct,
    _compute_macd_signal,
    _compute_rsi,
    forward_fill_trading_days,
    zscore_normalize_window,
)
from ml.cnn_encoder import encode_batch
from ml.embedding_generator import load_encoder


def ohlcv_rows_to_dataframe(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    cols = ["time", "symbol", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _normalize_window_end(window_end: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(window_end)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def build_normalized_query_window(
    df: pd.DataFrame,
    window_end: datetime,
    *,
    channels: list[str] | tuple[str, ...] = tuple(FEATURE_CHANNELS),
) -> tuple[np.ndarray, datetime, datetime, pd.DataFrame]:
    """Return ``(normalized_window [T, C], window_start, window_end, raw_ohlcv_slice DataFrame)``.

    By default includes the ``close_ret`` channel so the produced window
    matches what the encoder expects. Pass ``channels=OHLCV_CHANNELS`` only
    for backwards-compat with the legacy 5-channel encoder.
    """
    if df.empty:
        raise ValueError("no OHLCV rows for symbol")

    df = forward_fill_trading_days(df)
    df = df.sort_values("time").reset_index(drop=True)
    channel_list = list(channels)
    _INDICATOR_COLS = {"rsi_14", "bb_pct", "macd_signal"}
    if "close_ret" in channel_list and "close_ret" not in df.columns:
        cr = df["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df = df.assign(close_ret=cr)
    # Compute technical indicators if needed.
    if _INDICATOR_COLS & set(channel_list):
        c = df["close"].values.astype(float)
        if "rsi_14" in channel_list and "rsi_14" not in df.columns:
            df = df.assign(rsi_14=_compute_rsi(c))
        if "bb_pct" in channel_list and "bb_pct" not in df.columns:
            df = df.assign(bb_pct=_compute_bb_pct(c))
        if "macd_signal" in channel_list and "macd_signal" not in df.columns:
            df = df.assign(macd_signal=_compute_macd_signal(c))

    target = _normalize_window_end(window_end)
    match = df.index[df["time"] == target].tolist()
    if match:
        i = int(match[-1])
    else:
        le = df["time"] <= target
        if not le.any():
            raise ValueError("no rows on or before window_end")
        i = int(df.index[le][-1])

    if i < WINDOW_SIZE - 1:
        raise ValueError(f"need at least {WINDOW_SIZE} sessions ending at window_end; got {i + 1}")

    start_i = i - (WINDOW_SIZE - 1)
    slice_df = df.loc[start_i:i].copy()
    raw = slice_df[channel_list].to_numpy(dtype=np.float64)
    normed = zscore_normalize_window(raw, channel_names=channel_list)
    w_start = slice_df["time"].iloc[0].to_pydatetime()
    w_end = slice_df["time"].iloc[-1].to_pydatetime()
    # Keep the raw OHLCV slice (without close_ret) for visualization callers.
    display_cols = ["time", "open", "high", "low", "close", "volume"]
    present_cols = [c for c in display_cols if c in slice_df.columns]
    return normed, w_start, w_end, slice_df[present_cols]


def embedding_from_normalized_window(
    normed: np.ndarray,
    model_path: Path,
    *,
    device: str = "cpu",
) -> list[float]:
    model = load_encoder(model_path)
    emb = encode_batch(model, normed[np.newaxis, :, :], device=device, batch_size=1)[0]
    return [float(x) for x in emb.tolist()]
