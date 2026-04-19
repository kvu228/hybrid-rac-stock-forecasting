"""Build a 30-session normalized window + CNN embedding for RAC queries (API / Streamlit)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from etl.feature_engineer import OHLCV_CHANNELS, WINDOW_SIZE, forward_fill_trading_days, zscore_normalize_window
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
) -> tuple[np.ndarray, datetime, datetime, pd.DataFrame]:
    """Return (normalized_window [30,5], window_start, window_end, raw_ohlcv_slice DataFrame)."""
    if df.empty:
        raise ValueError("no OHLCV rows for symbol")

    df = forward_fill_trading_days(df)
    df = df.sort_values("time").reset_index(drop=True)
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
    raw = slice_df[list(OHLCV_CHANNELS)].to_numpy(dtype=np.float64)
    normed = zscore_normalize_window(raw)
    w_start = slice_df["time"].iloc[0].to_pydatetime()
    w_end = slice_df["time"].iloc[-1].to_pydatetime()
    out_cols = ["time", *OHLCV_CHANNELS]
    return normed, w_start, w_end, slice_df[out_cols]


def embedding_from_normalized_window(
    normed: np.ndarray,
    model_path: Path,
    *,
    device: str = "cpu",
) -> list[float]:
    model = load_encoder(model_path)
    emb = encode_batch(model, normed[np.newaxis, :, :], device=device, batch_size=1)[0]
    return [float(x) for x in emb.tolist()]
