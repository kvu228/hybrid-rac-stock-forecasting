from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from etl.data_cleaner import clean_ohlcv
from etl.ingestion import ingest_stock_ohlcv


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    fixture_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ohlcv_small.tsv"
    df = pd.read_csv(fixture_path, sep="\t")
    cleaned = clean_ohlcv(df)
    stats = ingest_stock_ohlcv(cleaned.df, database_url)
    print(f"Seeded: staged={stats.staged_rows} inserted_or_updated={stats.inserted_rows}")


if __name__ == "__main__":
    main()

