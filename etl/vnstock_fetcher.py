from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

import pandas as pd


Provider = Literal["vci"]


@dataclass(frozen=True)
class FetchRequest:
    symbol: str
    start: date
    end: date
    provider: Provider = "vci"


def fetch_ohlcv(req: FetchRequest) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a single symbol and date range.

    Returns a DataFrame with columns:
      time, symbol, open, high, low, close, volume
    """
    if req.provider != "vci":
        raise ValueError(f"Unsupported provider: {req.provider}")

    # Use vnstock's unified Quote adapter (VCI provider) to fetch OHLCV.
    # This is the stable API surface in vnstock 3.5.x.
    from vnstock import Quote

    q = Quote(source="vci", symbol=req.symbol)
    df = q.history(start=req.start.isoformat(), end=req.end.isoformat(), interval="1D")
    if df.empty:
        return pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])

    rename_map = {
        # vnstock Quote.history() (VCI) returns standardized OHLCV already,
        # but we keep a small normalization layer.
        "date": "time",
        "time": "time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "vol": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "time" not in df.columns:
        # Some vnstock responses may use 't' or similar internal names; fail loudly.
        raise ValueError(f"vnstock Quote.history output missing time column; got columns={list(df.columns)}")

    # Ensure symbol column exists and matches request.
    if "symbol" not in df.columns:
        df["symbol"] = req.symbol
    else:
        df["symbol"] = req.symbol

    df = df[["time", "symbol", "open", "high", "low", "close", "volume"]]
    return df


def fetch_many_ohlcv(
    symbols: Iterable[str],
    start: date,
    end: date,
    provider: Provider = "vci",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        frames.append(fetch_ohlcv(FetchRequest(symbol=sym, start=start, end=end, provider=provider)))
    if not frames:
        return pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])
    return pd.concat(frames, ignore_index=True)

