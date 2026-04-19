"""Benchmark helpers: EXPLAIN ANALYZE (whitelisted), pg_stat_statements, result JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException

from api.deps import DbConn
from api.schemas import BenchmarkExplainRequest, BenchmarkExplainResponse, BenchmarkResultItem, BenchmarkStatsResponse

router = APIRouter()

_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "benchmark" / "results"
_SAFE_JSON = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.json$")

ALLOWED_EXPLAIN_KINDS = frozenset({"hnsw_knn", "seqscan_knn", "hybrid_context"})


async def _fetch_sample_vector_text(conn: Any) -> str:
    cur = await conn.execute(
        "SELECT embedding::text FROM pattern_embeddings ORDER BY id LIMIT 1",
    )
    row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=400, detail="pattern_embeddings is empty")
    return str(row[0])


async def _explain_tx(conn: Any, inner_sql: str, params: dict[str, Any], *, seqscan_only: bool = False) -> str:
    await conn.execute("BEGIN")
    try:
        if seqscan_only:
            await conn.execute("SET LOCAL enable_indexscan = OFF")
            await conn.execute("SET LOCAL enable_bitmapscan = OFF")
        explain = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT TEXT)\n" + inner_sql.strip().rstrip(";")
        cur = await conn.execute(explain, params)
        lines = [cast(str, r[0]) for r in await cur.fetchall()]
        return "\n".join(lines)
    finally:
        await conn.execute("COMMIT")


@router.post("/benchmark/explain", response_model=BenchmarkExplainResponse)
async def benchmark_explain(conn: DbConn, req: BenchmarkExplainRequest) -> dict[str, str]:
    if req.query_kind not in ALLOWED_EXPLAIN_KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"query_kind must be one of: {', '.join(sorted(ALLOWED_EXPLAIN_KINDS))}",
        )

    qv = await _fetch_sample_vector_text(conn)
    k = int(req.k)
    params: dict[str, Any] = {"qv": qv, "k": k}

    await conn.execute("BEGIN")
    try:
        if req.ef_search is not None:
            await conn.execute("SET LOCAL hnsw.ef_search = %s", (str(int(req.ef_search)),))

        if req.query_kind == "hnsw_knn":
            inner = """
SELECT id, embedding <=> %(qv)s::vector AS dist
FROM pattern_embeddings
ORDER BY embedding <=> %(qv)s::vector
LIMIT %(k)s
"""
            explain = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT TEXT)\n" + inner.strip().rstrip(";")
            cur = await conn.execute(explain, params)
            lines = [cast(str, r[0]) for r in await cur.fetchall()]
            plan = "\n".join(lines)
            return {"query_kind": req.query_kind, "plan_text": plan}

        if req.query_kind == "seqscan_knn":
            await conn.execute("SET LOCAL enable_indexscan = OFF")
            await conn.execute("SET LOCAL enable_bitmapscan = OFF")
            inner = """
SELECT id, embedding <=> %(qv)s::vector AS dist
FROM pattern_embeddings
ORDER BY embedding <=> %(qv)s::vector
LIMIT %(k)s
"""
            explain = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT TEXT)\n" + inner.strip().rstrip(";")
            cur = await conn.execute(explain, params)
            lines = [cast(str, r[0]) for r in await cur.fetchall()]
            plan = "\n".join(lines)
            return {"query_kind": req.query_kind, "plan_text": plan}

        cur_sym = await conn.execute(
            "SELECT symbol FROM pattern_embeddings ORDER BY id LIMIT 1",
        )
        r2 = await cur_sym.fetchone()
        if r2 is None:
            raise HTTPException(status_code=400, detail="pattern_embeddings is empty")
        sym = str(r2[0])
        cur_px = await conn.execute(
            "SELECT close::float8 FROM stock_ohlcv WHERE symbol = %s ORDER BY time DESC LIMIT 1",
            (sym,),
        )
        r3 = await cur_px.fetchone()
        price = float(cast(Any, r3[0])) if r3 and r3[0] is not None else 1.0

        inner = """
SELECT * FROM compute_full_rac_context(%(qv)s::vector, %(sym)s, %(px)s, %(k)s)
"""
        explain = "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT TEXT)\n" + inner.strip().rstrip(";")
        cur = await conn.execute(explain, {"qv": qv, "sym": sym, "px": price, "k": k})
        lines = [cast(str, r[0]) for r in await cur.fetchall()]
        plan = "\n".join(lines)
        return {"query_kind": req.query_kind, "plan_text": plan}
    finally:
        await conn.execute("COMMIT")


@router.get("/benchmark/stats", response_model=BenchmarkStatsResponse)
async def benchmark_stats(conn: DbConn) -> dict[str, object]:
    cur = await conn.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements')",
    )
    row = await cur.fetchone()
    if not row or not bool(row[0]):
        return {
            "available": False,
            "hint": "Extension pg_stat_statements is not installed. Add it to shared_preload_libraries and CREATE EXTENSION.",
            "statements": [],
        }

    try:
        cur2 = await conn.execute(
            """
            SELECT query::text, calls, total_exec_time, mean_exec_time, rows
            FROM pg_stat_statements
            ORDER BY total_exec_time DESC
            LIMIT 20
            """
        )
        rows = await cur2.fetchall()
    except Exception as e:  # noqa: BLE001
        return {
            "available": False,
            "hint": f"Could not read pg_stat_statements: {e}",
            "statements": [],
        }

    statements: list[dict[str, object]] = []
    for r in rows:
        q = str(r[0])
        statements.append(
            {
                "query": (q[:500] + "…") if len(q) > 500 else q,
                "calls": int(cast(Any, r[1])) if r[1] is not None else 0,
                "total_exec_time": float(cast(Any, r[2])) if r[2] is not None else 0.0,
                "mean_exec_time": float(cast(Any, r[3])) if r[3] is not None else 0.0,
                "rows": int(cast(Any, r[4])) if r[4] is not None else 0,
            }
        )
    return {"available": True, "hint": None, "statements": statements}


@router.get("/benchmark/results", response_model=list[BenchmarkResultItem])
async def benchmark_results_list() -> list[dict[str, object]]:
    root = _RESULTS_ROOT
    if not root.is_dir():
        return []
    out: list[dict[str, object]] = []
    for p in sorted(root.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file():
            continue
        if not _SAFE_JSON.match(p.name):
            continue
        out.append({"name": p.name, "size_bytes": p.stat().st_size})
    return out


@router.get("/benchmark/results/{name}")
async def benchmark_results_get(name: str) -> dict[str, object]:
    if not _SAFE_JSON.match(name):
        raise HTTPException(status_code=400, detail="invalid result file name")
    path = (_RESULTS_ROOT / name).resolve()
    if not str(path).startswith(str(_RESULTS_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="invalid path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
