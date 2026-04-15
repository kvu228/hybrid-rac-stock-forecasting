from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from db.seed_small_dataset import main as seed_small_dataset
from etl.data_cleaner import clean_ohlcv
from etl.ingestion import ingest_stock_ohlcv
from etl.vnstock_fetcher import fetch_ohlcv


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _register_vnstock_user_from_env() -> None:
    api_key = (os.getenv("VNSTOCK_API_KEY") or "").strip()
    debug = (os.getenv("VNSTOCK_DEBUG_AUTH") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if not api_key:
        if debug:
            print("vnstock auth: guest (VNSTOCK_API_KEY not set)")
        return

    try:
        from vnstock import register_user
    except Exception:
        # If vnstock isn't available (or API changed), keep the pipeline runnable.
        if debug:
            print("vnstock auth: skipped (could not import vnstock.register_user)")
        return

    try:
        ok = bool(register_user(api_key=api_key))
        if debug:
            print(f"vnstock auth: register_user returned {ok}")
    except Exception:
        # Don't fail ETL just because auth couldn't be performed.
        if debug:
            print("vnstock auth: failed (exception during register_user)")
        return


def _load_dotenv_if_present() -> None:
    """
    Load `.env` into process environment if available.

    This repo commonly stores `DATABASE_URL` and `VNSTOCK_API_KEY` in `.env`, but Python
    does not automatically read it unless we do.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    # Only load local `.env` (if present) and do not override existing env vars.
    load_dotenv(override=False)


def _load_symbols(symbols: list[str] | None, symbols_file: str | None) -> list[str]:
    if symbols and symbols_file:
        raise SystemExit("Use either --symbols or --symbols-file, not both")

    if symbols_file:
        p = Path(symbols_file)
        raw = p.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        for line in raw:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
        if not out:
            raise SystemExit("No symbols found in --symbols-file")
        return out

    if not symbols:
        raise SystemExit("Missing symbols. Provide --symbols or --symbols-file")
    return [s.strip() for s in symbols if s.strip()]


def _date_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    if chunk_days <= 0:
        return [(start, end)]
    cur = start
    chunks: list[tuple[date, date]] = []
    while cur <= end:
        nxt = min(end, cur + timedelta(days=chunk_days - 1))
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return chunks


def _engine(database_url: str) -> sa.Engine:
    return sa.create_engine(database_url, pool_pre_ping=True)

def _get_db_today(database_url: str) -> date:
    with _engine(database_url).connect() as conn:
        return conn.execute(sa.text("SELECT NOW()::date;")).scalar_one()


def _get_max_date_per_symbol(database_url: str, symbols: list[str]) -> dict[str, date | None]:
    if not symbols:
        return {}

    # Note: stock_ohlcv.time is TIMESTAMPTZ; we cast to date for incremental start.
    stmt = sa.text(
        """
SELECT symbol, MAX(time)::date AS max_date
FROM stock_ohlcv
WHERE symbol = ANY(:symbols)
GROUP BY symbol;
"""
    )

    with _engine(database_url).connect() as conn:
        rows = conn.execute(stmt, {"symbols": symbols}).fetchall()

    out: dict[str, date | None] = {s: None for s in symbols}
    for sym, max_d in rows:
        out[str(sym)] = max_d
    return out


def _fetch_clean_ingest_one(
    symbol: str,
    start: date,
    end: date,
    database_url: str,
    chunk_days: int,
) -> tuple[str, int, int, int, int]:
    fetched = 0
    cleaned_rows = 0
    dropped_dupes = 0
    inserted_or_updated = 0

    frames: list[pd.DataFrame] = []
    for a, b in _date_chunks(start, end, chunk_days):
        # Import here to avoid a heavier dependency surface at module import time.
        from etl.vnstock_fetcher import FetchRequest

        df = fetch_ohlcv(req=FetchRequest(symbol=symbol, start=a, end=b, provider="vci"))
        fetched += len(df)
        if not df.empty:
            frames.append(df)

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])
    cleaned = clean_ohlcv(raw)
    cleaned_rows = len(cleaned.df)
    dropped_dupes = cleaned.dropped_duplicates
    stats = ingest_stock_ohlcv(cleaned.df, database_url)
    inserted_or_updated = stats.inserted_rows

    return (symbol, fetched, cleaned_rows, dropped_dupes, inserted_or_updated)


def main() -> None:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="ETL CLI: seed | backfill | incremental")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seed", help="Seed a small deterministic dataset into stock_ohlcv")

    p_backfill = sub.add_parser("backfill", help="Fetch → clean → ingest for a date range")
    p_backfill.add_argument("--symbols", nargs="+", help="Ticker symbols (e.g., VCB FPT VNM)")
    p_backfill.add_argument("--symbols-file", help="Text file with one symbol per line")
    p_backfill.add_argument("--start", type=_parse_date, required=True, help="Start date (YYYY-MM-DD)")
    p_backfill.add_argument("--end", type=_parse_date, required=True, help="End date (YYYY-MM-DD)")
    p_backfill.add_argument("--chunk-days", type=int, default=365, help="Split fetch range into N-day chunks (default: 365)")
    p_backfill.add_argument("--concurrency", type=int, default=1, help="Number of symbols to fetch in parallel (default: 1)")
    p_backfill.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="SQLAlchemy/psycopg URL")

    p_incr = sub.add_parser("incremental", help="Fetch only missing dates since max(time) in DB per symbol")
    p_incr.add_argument("--symbols", nargs="+", help="Ticker symbols (e.g., VCB FPT VNM)")
    p_incr.add_argument("--symbols-file", help="Text file with one symbol per line")
    p_incr.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="End date (YYYY-MM-DD). If omitted, uses DB server date (SELECT now()::date).",
    )
    p_incr.add_argument("--chunk-days", type=int, default=365, help="Split fetch range into N-day chunks (default: 365)")
    p_incr.add_argument("--concurrency", type=int, default=1, help="Number of symbols to fetch in parallel (default: 1)")
    p_incr.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="SQLAlchemy/psycopg URL")

    args = parser.parse_args()

    # If VNSTOCK_API_KEY is set, register once to raise rate limits.
    _register_vnstock_user_from_env()

    if args.cmd == "seed":
        seed_small_dataset()
        return

    database_url = args.database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required (or pass --database-url)")

    if args.cmd == "backfill":
        symbols = _load_symbols(args.symbols, args.symbols_file)
        start = args.start
        end = args.end
    else:
        symbols = _load_symbols(args.symbols, args.symbols_file)
        end = args.end if args.end is not None else _get_db_today(database_url)
        max_dates = _get_max_date_per_symbol(database_url, symbols)
        # Start at (max_date + 1) if present, else default to a conservative baseline.
        start = None

    concurrency = max(1, int(args.concurrency))
    chunk_days = int(args.chunk_days)

    results: list[tuple[str, int, int, int, int]] = []

    if args.cmd == "backfill":
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(_fetch_clean_ingest_one, sym, start, end, database_url, chunk_days): sym
                for sym in symbols
            }
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        max_dates = _get_max_date_per_symbol(database_url, symbols)
        jobs: list[tuple[str, date, date]] = []
        for sym in symbols:
            max_d = max_dates.get(sym)
            if max_d is None:
                # If symbol has no data yet, default to 2010 baseline.
                s = date(2010, 1, 1)
            else:
                s = max_d + timedelta(days=1)
            if s > end:
                continue
            jobs.append((sym, s, end))

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(_fetch_clean_ingest_one, sym, s, end, database_url, chunk_days): sym
                for sym, s, end in jobs
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda x: x[0])
    total_fetched = sum(r[1] for r in results)
    total_cleaned = sum(r[2] for r in results)
    total_dropped = sum(r[3] for r in results)
    total_upserted = sum(r[4] for r in results)

    print(
        f"symbols={len(results)} fetched={total_fetched} cleaned={total_cleaned} dropped_dupes={total_dropped} "
        f"inserted_or_updated={total_upserted}"
    )
    for sym, fetched, cleaned_rows, dropped, upserted in results:
        print(f"{sym}: fetched={fetched} cleaned={cleaned_rows} dropped_dupes={dropped} inserted_or_updated={upserted}")


if __name__ == "__main__":
    main()

