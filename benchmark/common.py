"""Shared DB helpers, EXPLAIN parsing, vector formatting, and result export."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.sql import SQL, Identifier, Literal

# Matches production migration 0006_indexes.py default (rebuild after DROP).
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 200
HNSW_INDEX_NAME = "idx_embedding_hnsw"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXPLAIN_DIR = RESULTS_DIR / "explain"


def normalize_database_url(raw: str) -> str:
    url = raw.strip()
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


def require_database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        msg = "DATABASE_URL is required for benchmarks."
        raise SystemExit(msg)
    return normalize_database_url(url)


def format_vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection[tuple[Any, ...]]]:
    conn = psycopg.connect(require_database_url(), autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()


def git_commit_short() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except OSError:
        pass
    return None


def server_metadata(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    with conn.cursor() as cur:
        cur.execute("SELECT version() AS v;")
        vrow = cur.fetchone()
        if vrow is None:
            raise RuntimeError("expected version() row")
        meta["postgresql_version"] = str(vrow[0])
        cur.execute(
            """
            SELECT extname, extversion
            FROM pg_extension
            WHERE extname IN ('vector', 'timescaledb')
            ORDER BY extname;
            """
        )
        meta["extensions"] = {str(r[0]): str(r[1]) for r in cur.fetchall()}
        cur.execute("SHOW shared_buffers;")
        sb = cur.fetchone()
        if sb is None:
            raise RuntimeError("expected SHOW shared_buffers row")
        meta["shared_buffers"] = str(sb[0])
        cur.execute("SHOW work_mem;")
        wm = cur.fetchone()
        if wm is None:
            raise RuntimeError("expected SHOW work_mem row")
        meta["work_mem"] = str(wm[0])
    meta["git_commit"] = git_commit_short()
    return meta


def parse_explain_planning_execution_ms(plan: str) -> tuple[float | None, float | None]:
    planning: float | None = None
    execution: float | None = None
    m = re.search(r"Planning Time:\s*([\d.]+)\s*ms", plan)
    if m:
        planning = float(m.group(1))
    m = re.search(r"Execution Time:\s*([\d.]+)\s*ms", plan)
    if m:
        execution = float(m.group(1))
    return planning, execution


def parse_buffers_shared_hit_read(plan: str) -> tuple[int | None, int | None]:
    """Sum shared hit/read across all buffer lines in EXPLAIN (BUFFERS) output."""
    hit_total = 0
    read_total = 0
    found = False
    for line in plan.splitlines():
        if "shared hit=" in line or "shared read=" in line:
            mh = re.search(r"shared hit=(\d+)", line)
            mr = re.search(r"shared read=(\d+)", line)
            if mh:
                hit_total += int(mh.group(1))
                found = True
            if mr:
                read_total += int(mr.group(1))
                found = True
    if not found:
        return None, None
    return hit_total, read_total


def percentile_sorted(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def percentiles_p50_p95_p99(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    s = sorted(values)
    return (
        percentile_sorted(s, 50),
        percentile_sorted(s, 95),
        percentile_sorted(s, 99),
    )


def recall_at_k(approx_ids: Sequence[int], exact_ids: Sequence[int]) -> float:
    if not exact_ids:
        return float("nan")
    exact_set = set(exact_ids)
    return len(exact_set.intersection(approx_ids)) / len(exact_set)


def fetch_sample_embeddings(
    conn: psycopg.Connection[Any],
    *,
    n: int,
    seed: int,
) -> list[tuple[int, list[float]]]:
    """Return (id, embedding as list of float) using reproducible ORDER BY id OFFSET."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM pattern_embeddings;")
        cnt = cur.fetchone()
        if cnt is None:
            return []
        total = int(cnt[0])
        if total == 0:
            return []
        take = min(n, total)
        offset = 0 if take >= total else int(seed) % (total - take + 1)
        cur.execute(
            """
            SELECT id, embedding::text
            FROM pattern_embeddings
            ORDER BY id
            OFFSET %s
            LIMIT %s;
            """,
            (offset, take),
        )
        rows = cur.fetchall()
    out: list[tuple[int, list[float]]] = []
    for eid, emb_text in rows:
        inner = str(emb_text).strip("[]")
        parts = [float(x) for x in inner.split(",") if x.strip()]
        out.append((int(eid), parts))
    return out


def run_explain_analyze(
    conn: psycopg.Connection[Any],
    inner_sql: str,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Run EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT TEXT) wrapped around inner_sql."""
    params = params or {}
    explain = (
        "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT TEXT)\n" + inner_sql.strip().rstrip(";")
    )
    with conn.cursor() as cur:
        cur.execute(explain, params)
        parts = [row[0] for row in cur.fetchall()]
    return "\n".join(parts)


def run_sql_timing(
    conn: psycopg.Connection[Any],
    sql: str,
    params: Mapping[str, Any] | None = None,
) -> tuple[float, Any]:
    """Wall-clock client timing for a statement (not server execution only)."""
    params = params or {}
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, rows


def ensure_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def write_explain_file(prefix: str, slug: str, body: str) -> Path:
    ensure_results_dirs()
    path = EXPLAIN_DIR / f"{prefix}_{slug}.txt"
    path.write_text(body, encoding="utf-8")
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


@dataclass
class BenchRunMeta:
    script: str
    seed: int
    k: int
    n_queries: int
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        base = {
            "script": self.script,
            "seed": self.seed,
            "k": self.k,
            "n_queries": self.n_queries,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        base.update(self.extra)
        return base


def hnsw_index_ddl(m: int, ef_construction: int) -> str:
    return (
        f"CREATE INDEX {HNSW_INDEX_NAME} ON pattern_embeddings "
        f"USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)});"
    )


def drop_hnsw_index(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(SQL("DROP INDEX IF EXISTS {}").format(Identifier(HNSW_INDEX_NAME)))


def create_hnsw_index(
    conn: psycopg.Connection[Any],
    *,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
) -> None:
    with conn.cursor() as cur:
        cur.execute(hnsw_index_ddl(m, ef_construction))


def index_size_bytes(conn: psycopg.Connection[Any], index_name: str = HNSW_INDEX_NAME) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(pg_relation_size(c.oid), 0)::bigint
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'I' AND c.relname = %s;
            """,
            (index_name,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return int(row[0])


def pick_dense_symbol(conn: psycopg.Connection[Any], *, min_count: int = 500) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT symbol, COUNT(*) AS c
            FROM pattern_embeddings
            GROUP BY symbol
            HAVING COUNT(*) >= %s
            ORDER BY c DESC
            LIMIT 1;
            """,
            (min_count,),
        )
        row = cur.fetchone()
        if row:
            return str(row[0])
        cur.execute(
            """
            SELECT symbol, COUNT(*) AS c
            FROM pattern_embeddings
            GROUP BY symbol
            ORDER BY c DESC
            LIMIT 1;
            """
        )
        row2 = cur.fetchone()
        return str(row2[0]) if row2 else None


def knn_top_ids(
    conn: psycopg.Connection[Any],
    query_literal: str,
    k: int,
    *,
    symbol: str | None = None,
) -> list[int]:
    """Top-K ids by cosine distance (exact ordering; use with/without HNSW index for recall)."""
    if symbol:
        sql = """
            SELECT id FROM pattern_embeddings
            WHERE symbol = %(sym)s
            ORDER BY embedding <=> %(qv)s::vector
            LIMIT %(k)s;
        """
        params = {"sym": symbol, "qv": query_literal, "k": k}
    else:
        sql = """
            SELECT id FROM pattern_embeddings
            ORDER BY embedding <=> %(qv)s::vector
            LIMIT %(k)s;
        """
        params = {"qv": query_literal, "k": k}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [int(r[0]) for r in cur.fetchall()]


def set_hnsw_ef_search(conn: psycopg.Connection[Any], ef_search: int) -> None:
    """Set session GUC. Must not use %%s/$1 — PostgreSQL rejects parameters on SET."""
    with conn.cursor() as cur:
        cur.execute(SQL("SET hnsw.ef_search = {}").format(Literal(int(ef_search))))


def stderr_print(msg: str) -> None:
    print(msg, flush=True)
