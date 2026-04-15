from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = ["time", "symbol", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class CleanResult:
    df: pd.DataFrame
    dropped_duplicates: int


def clean_ohlcv(df: pd.DataFrame) -> CleanResult:
    """
    Normalize and validate OHLCV rows for ingestion into `stock_ohlcv`.

    Output guarantees:
    - columns: time,symbol,open,high,low,close,volume (in order)
    - time: timezone-aware datetime64[ns, UTC]
    - symbol: non-empty string
    - numeric columns are coerced; volume is int64 (non-negative if possible)
    - rows sorted by (symbol, time) and deduped on (symbol, time)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Parse time and normalize to UTC tz-aware timestamps.
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"])

    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out[out["symbol"] != ""]

    for col in ["open", "high", "low", "close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out["volume"] = out["volume"].astype("int64")
    out.loc[out["volume"] < 0, "volume"] = 0

    out = out[REQUIRED_COLUMNS]
    out = out.sort_values(["symbol", "time"], kind="mergesort").reset_index(drop=True)

    before = len(out)
    out = out.drop_duplicates(subset=["symbol", "time"], keep="last").reset_index(drop=True)
    dropped = before - len(out)

    return CleanResult(df=out, dropped_duplicates=dropped)

