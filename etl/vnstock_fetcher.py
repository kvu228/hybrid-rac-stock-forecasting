from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import random
import threading
import time
from typing import Iterable, Literal

import pandas as pd


Provider = Literal["vci"]


class _RateLimiter:
    """Thread-safe process-wide request rate limiter."""

    def __init__(self, rpm: int, burst: int = 1) -> None:
        if rpm <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self._interval_s = 60.0 / float(rpm)
        self._burst = max(1, int(burst))
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()
        self._available = self._burst

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()

                # Refill burst tokens based on time elapsed.
                if self._available < self._burst:
                    # Number of full intervals elapsed since last scheduled time.
                    intervals = int(max(0.0, now - self._next_allowed) / self._interval_s)
                    if intervals > 0:
                        self._available = min(self._burst, self._available + intervals)
                        self._next_allowed = now

                if self._available > 0 and now >= self._next_allowed:
                    self._available -= 1
                    self._next_allowed = max(self._next_allowed, now) + self._interval_s
                    return

                sleep_for = max(0.0, self._next_allowed - now)

            time.sleep(min(sleep_for, 1.0))


_rate_limiter: _RateLimiter | None = None


def configure_rate_limiter(*, requests_per_minute: int, burst: int = 1) -> None:
    """Configure the global limiter used by `fetch_ohlcv`."""

    global _rate_limiter
    _rate_limiter = _RateLimiter(rpm=requests_per_minute, burst=burst)


def _is_rate_limited_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return (
        "429" in msg
        or "too many request" in msg
        or "too many requests" in msg
        or "rate limit" in msg
        or "ratelimit" in msg
        or "request limit" in msg
    )


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

    df = pd.DataFrame()
    max_attempts = 6
    base_backoff_s = 1.0
    for attempt in range(1, max_attempts + 1):
        if _rate_limiter is not None:
            _rate_limiter.acquire()
        try:
            df = q.history(start=req.start.isoformat(), end=req.end.isoformat(), interval="1D")
            break
        except Exception as e:
            # vnstock may raise (and wrap with tenacity RetryError) when there is no data for
            # the symbol or date range. Treat it as an empty result so ETL can proceed.
            if _is_no_data_error(e):
                return pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])

            if _is_rate_limited_error(e) and attempt < max_attempts:
                sleep_s = min(60.0, base_backoff_s * (2 ** (attempt - 1)))
                sleep_s *= 0.75 + 0.5 * random.random()  # jitter in [0.75, 1.25]
                time.sleep(sleep_s)
                continue

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

