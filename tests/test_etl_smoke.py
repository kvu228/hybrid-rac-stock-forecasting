import os
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from etl.data_cleaner import clean_ohlcv
from etl.ingestion import ingest_stock_ohlcv


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    assert url, "DATABASE_URL is required for integration tests"
    return url


def _engine() -> sa.Engine:
    return sa.create_engine(_database_url(), pool_pre_ping=True)


def test_cleaner_normalizes_and_dedups() -> None:
    df = pd.DataFrame(
        [
            {"time": "2024-01-02", "symbol": "VCB", "open": "100", "high": 105, "low": 99, "close": 104, "volume": 1000},
            {"time": "2024-01-02", "symbol": "VCB", "open": "100", "high": 105, "low": 99, "close": 104, "volume": 1000},
            {"time": "bad-date", "symbol": "VCB", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ]
    )
    res = clean_ohlcv(df)
    assert len(res.df) == 1
    assert res.dropped_duplicates == 1
    assert list(res.df.columns) == ["time", "symbol", "open", "high", "low", "close", "volume"]


def test_ingestion_idempotent_with_unique_key() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "ohlcv_small.tsv"
    raw = pd.read_csv(fixture, sep="\t")
    cleaned = clean_ohlcv(raw)

    symbols = sorted(cleaned.df["symbol"].unique().tolist())

    # Ensure a clean slate even if the DB already contains rows
    # (e.g., after running a backfill locally).
    with _engine().begin() as conn:
        conn.execute(
            sa.text(
                """
DELETE FROM stock_ohlcv
WHERE symbol = ANY(:symbols)
  AND time >= '2024-01-01'
  AND time < '2024-01-10';
"""
            ),
            {"symbols": symbols},
        )

    # First ingest
    ingest_stock_ohlcv(cleaned.df, _database_url())
    # Second ingest should not create duplicates (upsert by (symbol,time))
    ingest_stock_ohlcv(cleaned.df, _database_url())

    with _engine().connect() as conn:
        count = conn.execute(
            sa.text(
                """
SELECT COUNT(*)
FROM stock_ohlcv
WHERE symbol = ANY(:symbols)
  AND time >= '2024-01-01'
  AND time < '2024-01-10';
"""
            ),
            {"symbols": symbols},
        ).scalar_one()
        assert int(count) == len(cleaned.df)

