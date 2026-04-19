"""Compare global KNN vs B-tree symbol filter + KNN on pattern_embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import psycopg

from benchmark.common import (
    BenchRunMeta,
    ensure_results_dirs,
    fetch_sample_embeddings,
    format_vector_literal,
    parse_buffers_shared_hit_read,
    parse_explain_planning_execution_ms,
    percentiles_p50_p95_p99,
    pick_dense_symbol,
    require_database_url,
    run_explain_analyze,
    server_metadata,
    set_hnsw_ef_search,
    timestamp_slug,
    write_csv,
    write_explain_file,
    write_json,
)

SQL_GLOBAL = """
SELECT id, symbol, embedding <=> %(qv)s::vector AS dist
FROM pattern_embeddings
ORDER BY embedding <=> %(qv)s::vector
LIMIT %(k)s
"""

SQL_FILTERED = """
SELECT id, symbol, embedding <=> %(qv)s::vector AS dist
FROM pattern_embeddings
WHERE symbol = %(sym)s
ORDER BY embedding <=> %(qv)s::vector
LIMIT %(k)s
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ef-search", type=int, default=100)
    p.add_argument("--symbol", default="", help="Override symbol for filtered query.")
    args = p.parse_args()
    url = require_database_url()
    slug = timestamp_slug()
    ensure_results_dirs()
    rows_out: list[dict[str, Any]] = []

    with psycopg.connect(url, autocommit=True) as conn:
        meta = server_metadata(conn)
        sym = (args.symbol or "").strip() or pick_dense_symbol(conn, min_count=50)
        if not sym:
            raise SystemExit("No symbol found in pattern_embeddings.")
        samples = fetch_sample_embeddings(conn, n=args.n_queries, seed=args.seed)
        if not samples:
            raise SystemExit("pattern_embeddings is empty.")
        set_hnsw_ef_search(conn, args.ef_search)

        global_ms: list[float] = []
        filtered_ms: list[float] = []

        for i, (_eid, vec) in enumerate(samples):
            qv = format_vector_literal(vec)
            pg = run_explain_analyze(conn, SQL_GLOBAL, {"qv": qv, "k": args.k})
            e_ms = parse_explain_planning_execution_ms(pg)[1]
            if e_ms is not None:
                global_ms.append(e_ms)
            gh, gr = parse_buffers_shared_hit_read(pg)
            write_explain_file(f"hybrid_global_q{i}", slug, pg)
            rows_out.append(
                {
                    "variant": "global_knn",
                    "query_idx": i,
                    "symbol_filter": "",
                    "execution_time_ms": e_ms,
                    "shared_hit": gh,
                    "shared_read": gr,
                }
            )

            pf = run_explain_analyze(conn, SQL_FILTERED, {"qv": qv, "k": args.k, "sym": sym})
            f_ms = parse_explain_planning_execution_ms(pf)[1]
            if f_ms is not None:
                filtered_ms.append(f_ms)
            fh, fr = parse_buffers_shared_hit_read(pf)
            write_explain_file(f"hybrid_filtered_{sym}_q{i}", slug, pf)
            rows_out.append(
                {
                    "variant": "filtered_symbol_knn",
                    "query_idx": i,
                    "symbol_filter": sym,
                    "execution_time_ms": f_ms,
                    "shared_hit": fh,
                    "shared_read": fr,
                }
            )

    g50, g95, g99 = percentiles_p50_p95_p99(global_ms)
    f50, f95, f99 = percentiles_p50_p95_p99(filtered_ms)
    summary = BenchRunMeta(
        script="hybrid_search_bench",
        seed=args.seed,
        k=args.k,
        n_queries=len(samples),
        extra={
            "symbol_used": sym,
            "ef_search": args.ef_search,
            "global_ms_p50_p95_p99": [g50, g95, g99],
            "filtered_ms_p50_p95_p99": [f50, f95, f99],
            "server": meta,
        },
    )

    base = Path(__file__).resolve().parent / "results" / f"hybrid_search_bench_{slug}"
    write_csv(
        base.with_suffix(".csv"),
        ["variant", "query_idx", "symbol_filter", "execution_time_ms", "shared_hit", "shared_read"],
        rows_out,
    )
    write_json(base.with_suffix(".json"), {"summary": summary.as_dict(), "rows": rows_out})
    print(f"Wrote {base.with_suffix('.csv')} and {base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
