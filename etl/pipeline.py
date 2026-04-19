from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa

from db.seed_small_dataset import seed_fixture_ohlcv
from etl.data_cleaner import clean_ohlcv
from etl.ingestion import ingest_stock_ohlcv
from etl.vnstock_fetcher import configure_rate_limiter, fetch_ohlcv
from dotenv import load_dotenv


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

    # Only load local `.env` (if present) and do not override existing env vars.
    load_dotenv(override=False)


def parse_symbols_list(symbols: list[str] | None, symbols_file: str | None) -> list[str]:
    """Resolve symbol list from CLI/API. Raises ``ValueError`` on invalid input."""
    if symbols and symbols_file:
        raise ValueError("Use either symbols or symbols_file, not both")

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
            raise ValueError("No symbols found in symbols_file")
        return out

    if not symbols:
        raise ValueError("Missing symbols. Provide symbols or symbols_file")
    return [s.strip() for s in symbols if s.strip()]


def _load_symbols(symbols: list[str] | None, symbols_file: str | None) -> list[str]:
    try:
        return parse_symbols_list(symbols, symbols_file)
    except ValueError as e:
        raise SystemExit(str(e)) from e


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


def _fetch_ohlcv_from_db(
    database_url: str,
    symbols: list[str],
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    """Read OHLCV rows from the DB for the given symbols and optional date range."""
    engine = _engine(database_url)
    clauses = ["symbol = ANY(:symbols)"]
    params: dict[str, object] = {"symbols": symbols}
    if start is not None:
        clauses.append("time >= :start")
        params["start"] = str(start)
    if end is not None:
        clauses.append("time <= :end")
        params["end"] = str(end)

    where = " AND ".join(clauses)
    query = sa.text(
        f"SELECT time, symbol, open, high, low, close, volume FROM stock_ohlcv WHERE {where} ORDER BY symbol, time"  # noqa: S608
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume"])


def _cmd_generate_windows(args: argparse.Namespace, database_url: str) -> None:
    from etl.feature_engineer import (
        forward_fill_trading_days,
        generate_windows,
        train_test_split_by_time,
    )

    import numpy as np

    symbols = _load_symbols(args.symbols, args.symbols_file)
    df = _fetch_ohlcv_from_db(database_url, symbols, args.start, args.end)
    if df.empty:
        print("No OHLCV data found for the given symbols/date range.")
        return

    df = forward_fill_trading_days(df)
    records = generate_windows(
        df,
        window_size=args.window_size,
        horizon=args.horizon,
        stride=args.stride,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
    )
    if not records:
        print("Not enough data to generate windows.")
        return

    train, test = train_test_split_by_time(records, train_ratio=args.train_ratio)

    # Export to disk
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_recs in [("train", train), ("test", test)]:
        if not split_recs:
            continue
        data = np.stack([r.data for r in split_recs])  # (N, window_size, 5)
        meta = pd.DataFrame(
            {
                "symbol": [r.symbol for r in split_recs],
                "window_start": [r.window_start for r in split_recs],
                "window_end": [r.window_end for r in split_recs],
                "label": [r.label for r in split_recs],
                "future_return": [r.future_return for r in split_recs],
            }
        )
        np.savez_compressed(out_dir / f"{split_name}_windows.npz", data=data)
        meta.to_csv(out_dir / f"{split_name}_metadata.csv", index=False)

    label_counts = {0: 0, 1: 0, 2: 0}
    for r in records:
        label_counts[r.label] += 1

    print(
        f"windows={len(records)} train={len(train)} test={len(test)} "
        f"labels={{Down={label_counts[0]}, Neutral={label_counts[1]}, Up={label_counts[2]}}}"
    )
    print(f"Saved to {out_dir}/")


def _cmd_detect_sr(args: argparse.Namespace, database_url: str) -> None:
    from etl.sr_detector import detect_sr_zones, ingest_sr_zones

    symbols = _load_symbols(args.symbols, args.symbols_file)
    df = _fetch_ohlcv_from_db(database_url, symbols, start=None, end=None)
    if df.empty:
        print("No OHLCV data found for the given symbols.")
        return

    zones = detect_sr_zones(df, order=args.order)
    inserted = ingest_sr_zones(zones, database_url)
    support_count = sum(1 for z in zones if z.zone_type == "SUPPORT")
    resist_count = sum(1 for z in zones if z.zone_type == "RESISTANCE")
    print(f"symbols={len(symbols)} zones={len(zones)} support={support_count} resistance={resist_count} inserted={inserted}")


def _cmd_purge_inactive_sr(args: argparse.Namespace, database_url: str) -> None:
    from etl.sr_detector import purge_inactive_sr_zones

    if args.all_inactive:
        n = purge_inactive_sr_zones(database_url, symbols=None)
    else:
        syms = _load_symbols(args.symbols, args.symbols_file)
        n = purge_inactive_sr_zones(database_url, symbols=syms)
    print(f"purge-inactive-sr deleted_rows={n}")


def run_seed_small_dataset(*, database_url: str | None = None) -> dict[str, int]:
    """Load the small OHLCV fixture into ``stock_ohlcv``. Requires ``DATABASE_URL`` unless passed explicitly."""
    _load_dotenv_if_present()
    stats = seed_fixture_ohlcv(database_url=database_url)
    return {"staged_rows": stats.staged_rows, "inserted_rows": stats.inserted_rows}


def _aggregate_ingest_results(results: list[tuple[str, int, int, int, int]]) -> dict[str, Any]:
    results = sorted(results, key=lambda x: x[0])
    per_symbol = [
        {
            "symbol": sym,
            "fetched": fetched,
            "cleaned_rows": cleaned_rows,
            "dropped_duplicates": dropped,
            "inserted_or_updated": upserted,
        }
        for sym, fetched, cleaned_rows, dropped, upserted in results
    ]
    return {
        "symbols_processed": len(results),
        "total_fetched": sum(r[1] for r in results),
        "total_cleaned_rows": sum(r[2] for r in results),
        "total_dropped_duplicates": sum(r[3] for r in results),
        "total_inserted_or_updated": sum(r[4] for r in results),
        "per_symbol": per_symbol,
    }


def run_backfill(
    database_url: str,
    symbols: list[str],
    start: date,
    end: date,
    *,
    chunk_days: int = 365,
    concurrency: int = 1,
    requests_per_minute: int | None = None,
    rate_limit_burst: int | None = None,
) -> dict[str, Any]:
    _load_dotenv_if_present()
    _register_vnstock_user_from_env()
    rpm = (
        int(requests_per_minute)
        if requests_per_minute is not None
        else int((os.getenv("VNSTOCK_REQUESTS_PER_MINUTE") or "55").strip() or "55")
    )
    burst = (
        int(rate_limit_burst)
        if rate_limit_burst is not None
        else int((os.getenv("VNSTOCK_RATE_LIMIT_BURST") or "1").strip() or "1")
    )
    configure_rate_limiter(requests_per_minute=max(1, rpm), burst=max(1, burst))
    concurrency = max(1, int(concurrency))
    chunk_days = int(chunk_days)
    results: list[tuple[str, int, int, int, int]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(_fetch_clean_ingest_one, sym, start, end, database_url, chunk_days): sym for sym in symbols
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    out = _aggregate_ingest_results(results)
    out["mode"] = "backfill"
    out["start"] = str(start)
    out["end"] = str(end)
    return out


def run_incremental(
    database_url: str,
    symbols: list[str],
    end: date,
    *,
    chunk_days: int = 365,
    concurrency: int = 1,
    requests_per_minute: int | None = None,
    rate_limit_burst: int | None = None,
) -> dict[str, Any]:
    _load_dotenv_if_present()
    _register_vnstock_user_from_env()
    rpm = (
        int(requests_per_minute)
        if requests_per_minute is not None
        else int((os.getenv("VNSTOCK_REQUESTS_PER_MINUTE") or "55").strip() or "55")
    )
    burst = (
        int(rate_limit_burst)
        if rate_limit_burst is not None
        else int((os.getenv("VNSTOCK_RATE_LIMIT_BURST") or "1").strip() or "1")
    )
    configure_rate_limiter(requests_per_minute=max(1, rpm), burst=max(1, burst))
    concurrency = max(1, int(concurrency))
    chunk_days = int(chunk_days)
    max_dates = _get_max_date_per_symbol(database_url, symbols)
    jobs: list[tuple[str, date, date]] = []
    for sym in symbols:
        max_d = max_dates.get(sym)
        s = date(2010, 1, 1) if max_d is None else max_d + timedelta(days=1)
        if s > end:
            continue
        jobs.append((sym, s, end))

    results: list[tuple[str, int, int, int, int]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(_fetch_clean_ingest_one, sym, s, e, database_url, chunk_days): sym for sym, s, e in jobs
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    out = _aggregate_ingest_results(results)
    out["mode"] = "incremental"
    out["end"] = str(end)
    out["jobs_planned"] = len(jobs)
    return out


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
    p_backfill.add_argument(
        "--requests-per-minute",
        type=int,
        default=int((os.getenv("VNSTOCK_REQUESTS_PER_MINUTE") or "55").strip() or "55"),
        help="Global VNStock request budget across threads (default: env VNSTOCK_REQUESTS_PER_MINUTE or 55)",
    )
    p_backfill.add_argument(
        "--rate-limit-burst",
        type=int,
        default=int((os.getenv("VNSTOCK_RATE_LIMIT_BURST") or "1").strip() or "1"),
        help="Allow short bursts up to N requests (default: env VNSTOCK_RATE_LIMIT_BURST or 1)",
    )
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
    p_incr.add_argument(
        "--requests-per-minute",
        type=int,
        default=int((os.getenv("VNSTOCK_REQUESTS_PER_MINUTE") or "55").strip() or "55"),
        help="Global VNStock request budget across threads (default: env VNSTOCK_REQUESTS_PER_MINUTE or 55)",
    )
    p_incr.add_argument(
        "--rate-limit-burst",
        type=int,
        default=int((os.getenv("VNSTOCK_RATE_LIMIT_BURST") or "1").strip() or "1"),
        help="Allow short bursts up to N requests (default: env VNSTOCK_RATE_LIMIT_BURST or 1)",
    )
    p_incr.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="SQLAlchemy/psycopg URL")

    # --- Phase 3: Feature Engineering commands ---

    p_windows = sub.add_parser("generate-windows", help="Generate labeled sliding windows from OHLCV data in DB")
    p_windows.add_argument("--symbols", nargs="+", help="Ticker symbols")
    p_windows.add_argument("--symbols-file", help="Text file with one symbol per line")
    p_windows.add_argument("--start", type=_parse_date, default=None, help="Start date filter (optional)")
    p_windows.add_argument("--end", type=_parse_date, default=None, help="End date filter (optional)")
    p_windows.add_argument("--window-size", type=int, default=30, help="Sessions per window (default: 30)")
    p_windows.add_argument("--horizon", type=int, default=5, help="Forward return horizon in sessions (default: 5)")
    p_windows.add_argument("--stride", type=int, default=1, help="Window stride (default: 1)")
    p_windows.add_argument("--up-threshold", type=float, default=0.02, help="Return >= this → Up label (default: 0.02)")
    p_windows.add_argument("--down-threshold", type=float, default=-0.02, help="Return <= this → Down label (default: -0.02)")
    p_windows.add_argument("--train-ratio", type=float, default=0.8, help="Chronological train ratio (default: 0.8)")
    p_windows.add_argument("--output-dir", default="data/windows", help="Directory for output .npz + metadata (default: data/windows)")
    p_windows.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="SQLAlchemy/psycopg URL")

    p_sr = sub.add_parser("detect-sr", help="Detect support/resistance zones and insert into DB")
    p_sr.add_argument("--symbols", nargs="+", help="Ticker symbols")
    p_sr.add_argument("--symbols-file", help="Text file with one symbol per line")
    p_sr.add_argument("--order", type=int, default=5, help="Pivot half-window size (default: 5)")
    p_sr.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="SQLAlchemy/psycopg URL")

    p_purge = sub.add_parser(
        "purge-inactive-sr",
        help="DELETE support_resistance_zones rows where is_active=FALSE",
    )
    p_purge_mx = p_purge.add_mutually_exclusive_group(required=True)
    p_purge_mx.add_argument(
        "--all-inactive",
        action="store_true",
        help="Delete every inactive row (all symbols)",
    )
    p_purge_mx.add_argument("--symbols", nargs="+", help="Ticker symbols")
    p_purge_mx.add_argument("--symbols-file", help="Text file with one symbol per line")
    p_purge.add_argument("--database-url", default=os.getenv("DATABASE_URL"), help="SQLAlchemy/psycopg URL")

    args = parser.parse_args()

    # If VNSTOCK_API_KEY is set, register once to raise rate limits.
    _register_vnstock_user_from_env()

    if args.cmd == "seed":
        stats = run_seed_small_dataset()
        print(
            f"Seeded: staged={stats['staged_rows']} inserted_or_updated={stats['inserted_rows']}",
        )
        return

    database_url = args.database_url
    if not database_url:
        raise SystemExit("DATABASE_URL is required (or pass --database-url)")

    if args.cmd == "generate-windows":
        _cmd_generate_windows(args, database_url)
        return

    if args.cmd == "detect-sr":
        _cmd_detect_sr(args, database_url)
        return

    if args.cmd == "purge-inactive-sr":
        _cmd_purge_inactive_sr(args, database_url)
        return

    if args.cmd == "backfill":
        symbols = _load_symbols(args.symbols, args.symbols_file)
        res = run_backfill(
            database_url,
            symbols,
            args.start,
            args.end,
            chunk_days=args.chunk_days,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            rate_limit_burst=args.rate_limit_burst,
        )
    elif args.cmd == "incremental":
        symbols = _load_symbols(args.symbols, args.symbols_file)
        end = args.end if args.end is not None else _get_db_today(database_url)
        res = run_incremental(
            database_url,
            symbols,
            end,
            chunk_days=args.chunk_days,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            rate_limit_burst=args.rate_limit_burst,
        )
    else:
        raise SystemExit(f"unexpected cmd {args.cmd!r}")

    print(
        f"symbols={res['symbols_processed']} fetched={res['total_fetched']} "
        f"cleaned={res['total_cleaned_rows']} dropped_dupes={res['total_dropped_duplicates']} "
        f"inserted_or_updated={res['total_inserted_or_updated']}",
    )
    for row in res["per_symbol"]:
        print(
            f"{row['symbol']}: fetched={row['fetched']} cleaned={row['cleaned_rows']} "
            f"dropped_dupes={row['dropped_duplicates']} inserted_or_updated={row['inserted_or_updated']}",
        )


if __name__ == "__main__":
    main()

