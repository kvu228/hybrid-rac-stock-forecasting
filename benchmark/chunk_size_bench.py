"""Compare TimescaleDB chunk intervals using separate hypertables (same OHLCV data copy)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import psycopg

from benchmark.common import (
    BenchRunMeta,
    ensure_results_dirs,
    parse_explain_planning_execution_ms,
    require_database_url,
    run_explain_analyze,
    server_metadata,
    timestamp_slug,
    write_csv,
    write_explain_file,
    write_json,
)

BENCH_SPECS: tuple[tuple[str, str, str], ...] = (
    ("bench_stock_ohlcv_1w", "1 week", "7 days"),
    ("bench_stock_ohlcv_1m", "1 month", "1 month"),
    ("bench_stock_ohlcv_3m", "3 months", "3 months"),
)


def parse_chunks_excluded(plan: str) -> int | None:
    m = re.search(r"Chunks excluded:\s*(\d+)", plan, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r"chunks excluded[:\s]+(\d+)", plan, re.IGNORECASE)
    return int(m2.group(1)) if m2 else None


def compression_ratio_estimate(conn: psycopg.Connection[Any], hypertable: str) -> float | None:
    """Best-effort compressed vs uncompressed size from TimescaleDB helper views."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s;
            """,
            (hypertable,),
        )
        if cur.fetchone() is None:
            return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  COALESCE(SUM(before_compression_total_bytes), 0)::bigint AS before_b,
                  COALESCE(SUM(after_compression_total_bytes), 0)::bigint AS after_b
                FROM chunk_compression_stats(%s);
                """,
                (hypertable,),
            )
            row = cur.fetchone()
            if row is None or row[0] == 0:
                return None
            before_b, after_b = int(row[0]), int(row[1])
            if after_b == 0:
                return None
            return round(before_b / float(after_b), 4)
    except Exception:
        return None


def pick_range_params(conn: psycopg.Connection[Any]) -> tuple[str, str, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT symbol, MIN(time)::text, MAX(time)::text
            FROM stock_ohlcv
            GROUP BY symbol
            ORDER BY COUNT(*) DESC
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        if not row:
            raise SystemExit("stock_ohlcv is empty; load OHLCV before chunk benchmark.")
        return str(row[0]), str(row[1]), str(row[2])


def setup_bench_hypertable(
    conn: psycopg.Connection[Any],
    table_name: str,
    interval_sql: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"DROP TABLE IF EXISTS {table_name} CASCADE;",
        )
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
              time TIMESTAMPTZ NOT NULL,
              symbol TEXT NOT NULL,
              open DOUBLE PRECISION NOT NULL,
              high DOUBLE PRECISION NOT NULL,
              low DOUBLE PRECISION NOT NULL,
              close DOUBLE PRECISION NOT NULL,
              volume BIGINT NOT NULL
            );
            """
        )
        cur.execute(
            f"INSERT INTO {table_name} SELECT * FROM stock_ohlcv;",
        )
        cur.execute(
            f"""
            SELECT create_hypertable(
              '{table_name}', 'time',
              chunk_time_interval => INTERVAL '{interval_sql}',
              if_not_exists => TRUE
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX idx_{table_name}_symbol_time
            ON {table_name} (symbol, time DESC);
            """
        )
        cur.execute(
            f"""
            ALTER TABLE {table_name} SET (
              timescaledb.compress,
              timescaledb.compress_segmentby = 'symbol',
              timescaledb.compress_orderby = 'time DESC'
            );
            """
        )
        try:
            cur.execute(
                f"SELECT add_compression_policy('{table_name}', INTERVAL '90 days');",
            )
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teardown-only", action="store_true", help="Drop bench_* hypertables and exit.")
    args = p.parse_args()
    url = require_database_url()
    slug = timestamp_slug()
    ensure_results_dirs()
    rows_out: list[dict[str, Any]] = []

    with psycopg.connect(url, autocommit=True) as conn:
        meta = server_metadata(conn)
        if args.teardown_only:
            with conn.cursor() as cur:
                for tbl, _label, _iv in BENCH_SPECS:
                    cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE;")
            print("Dropped bench hypertables.")
            return

        sym, t_start, t_end = pick_range_params(conn)

        for tbl, label, interval_sql in BENCH_SPECS:
            setup_bench_hypertable(conn, tbl, interval_sql)
            inner = f"""
SELECT time, open, high, low, close, volume
FROM {tbl}
WHERE symbol = %(sym)s AND time BETWEEN %(ts)s::timestamptz AND %(te)s::timestamptz
ORDER BY time;
"""
            plan = run_explain_analyze(
                conn,
                inner,
                {"sym": sym, "ts": t_start, "te": t_end},
            )
            write_explain_file(f"chunk_bench_{tbl}", slug, plan)
            _, exec_ms = parse_explain_planning_execution_ms(plan)
            n_excl = parse_chunks_excluded(plan)
            ratio = compression_ratio_estimate(conn, tbl)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)::bigint
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = %s;
                    """,
                    (tbl,),
                )
                crow = cur.fetchone()
                if crow is None:
                    raise RuntimeError("expected chunk count row")
                n_chunks = int(crow[0])
            rows_out.append(
                {
                    "table": tbl,
                    "chunk_interval_label": label,
                    "symbol": sym,
                    "time_start": t_start,
                    "time_end": t_end,
                    "num_chunks": n_chunks,
                    "chunks_excluded": n_excl,
                    "execution_time_ms": exec_ms,
                    "compression_ratio_estimate": ratio,
                }
            )

    summary = BenchRunMeta(
        script="chunk_size_bench",
        seed=0,
        k=0,
        n_queries=len(rows_out),
        extra={"server": meta},
    )
    base = Path(__file__).resolve().parent / "results" / f"chunk_size_bench_{slug}"
    write_csv(
        base.with_suffix(".csv"),
        [
            "table",
            "chunk_interval_label",
            "symbol",
            "time_start",
            "time_end",
            "num_chunks",
            "chunks_excluded",
            "execution_time_ms",
            "compression_ratio_estimate",
        ],
        rows_out,
    )
    write_json(base.with_suffix(".json"), {"summary": summary.as_dict(), "rows": rows_out})
    print(f"Wrote {base.with_suffix('.csv')} and {base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
