from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from etl.data_cleaner import clean_ohlcv
from etl.ingestion import IngestStats, ingest_stock_ohlcv


def seed_fixture_ohlcv(*, database_url: str | None = None) -> IngestStats:
    """Load ``tests/fixtures/ohlcv_small.tsv`` into ``stock_ohlcv`` (idempotent upsert)."""
    url = (database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is required")

    fixture_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ohlcv_small.tsv"
    df = pd.read_csv(fixture_path, sep="\t")
    cleaned = clean_ohlcv(df)
    return ingest_stock_ohlcv(cleaned.df, url)


def main() -> None:
    stats = seed_fixture_ohlcv()
    print(f"Seeded: staged={stats.staged_rows} inserted_or_updated={stats.inserted_rows}")


if __name__ == "__main__":
    main()

