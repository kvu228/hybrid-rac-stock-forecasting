from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd
import psycopg


@dataclass(frozen=True)
class IngestStats:
    staged_rows: int
    inserted_rows: int


def _normalize_psycopg_url(database_url: str) -> str:
    # CI/dev may use SQLAlchemy-style URLs: postgresql+psycopg://...
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url.removeprefix("postgresql+psycopg://")
    return database_url


def ingest_stock_ohlcv(df: pd.DataFrame, database_url: str) -> IngestStats:
    """
    Bulk ingest cleaned OHLCV into `stock_ohlcv` using COPY into a temporary staging
    table, then upsert into the hypertable.

    Requires unique index on (symbol, time): `ux_stock_ohlcv_symbol_time`.
    """
    if df.empty:
        return IngestStats(staged_rows=0, inserted_rows=0)

    required = ["time", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # COPY expects text; emit ISO timestamps and tab-separated values.
    buf = io.StringIO()
    tmp = df.copy()
    tmp["time"] = pd.to_datetime(tmp["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = tmp[required]
    tmp.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
    buf.seek(0)

    with psycopg.connect(_normalize_psycopg_url(database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
CREATE TEMP TABLE tmp_stock_ohlcv (
  time   TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  open   DOUBLE PRECISION NOT NULL,
  high   DOUBLE PRECISION NOT NULL,
  low    DOUBLE PRECISION NOT NULL,
  close  DOUBLE PRECISION NOT NULL,
  volume BIGINT NOT NULL
) ON COMMIT DROP;
"""
            )

            with cur.copy(
                "COPY tmp_stock_ohlcv (time, symbol, open, high, low, close, volume) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')"
            ) as copy:
                copy.write(buf.read())

            cur.execute(
                """
INSERT INTO stock_ohlcv (time, symbol, open, high, low, close, volume)
SELECT time, symbol, open, high, low, close, volume
FROM tmp_stock_ohlcv
ON CONFLICT (symbol, time) DO UPDATE
SET open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume;
"""
            )
            inserted = cur.rowcount if cur.rowcount is not None else 0

    return IngestStats(staged_rows=len(df), inserted_rows=inserted)

