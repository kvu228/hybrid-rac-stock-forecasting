from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

import pandas as pd


Provider = Literal["vci"]


def _is_no_data_error(err: BaseException) -> bool:
    """
    vnstock may raise ValueError("Không tìm thấy dữ liệu...") for invalid symbol/date range.
    Sometimes this is wrapped by tenacity.RetryError; we treat the whole chain as "no data".
    """

    def _has_no_data_text(msg: str) -> bool:
        return (
            "Không tìm thấy dữ liệu" in msg
            or "Khong tim thay du lieu" in msg
            or "No data" in msg
        )

    # 1) Direct message checks.
    if _has_no_data_text(str(err)):
        return True

    # 2) Unwrap tenacity RetryError if present.
    try:
        import tenacity

        if isinstance(err, tenacity.RetryError):
            last = getattr(err, "last_attempt", None)
            if last is not None:
                last_exc = getattr(last, "exception", lambda: None)()
                if last_exc is not None and _has_no_data_text(str(last_exc)):
                    return True
    except Exception:
        pass

    # 3) Walk cause/context chain.
    seen: set[int] = set()
    cur: BaseException | None = err
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _has_no_data_text(str(cur)):
            return True
        cur = cur.__cause__ or cur.__context__

    return False


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
    try:
        df = q.history(start=req.start.isoformat(), end=req.end.isoformat(), interval="1D")
    except Exception as e:
        # vnstock may raise (and wrap with tenacity RetryError) when there is no data for
        # the symbol or date range. Treat it as an empty result so ETL can proceed.
        if _is_no_data_error(e):
            return pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])
        raise
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

