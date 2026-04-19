"""Benchmark HNSW index vs exact sequential scan (index dropped) for KNN on pattern_embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import psycopg

from benchmark.common import (
    BenchRunMeta,
    create_hnsw_index,
    drop_hnsw_index,
    ensure_results_dirs,
    fetch_sample_embeddings,
    format_vector_literal,
    knn_top_ids,
    parse_buffers_shared_hit_read,
    parse_explain_planning_execution_ms,
    percentiles_p50_p95_p99,
    recall_at_k,
    require_database_url,
    run_explain_analyze,
    server_metadata,
    set_hnsw_ef_search,
    timestamp_slug,
    write_csv,
    write_explain_file,
    write_json,
)

INNER_KNN_SQL = """
SELECT id, embedding <=> %(qv)s::vector AS dist
FROM pattern_embeddings
ORDER BY embedding <=> %(qv)s::vector
LIMIT %(k)s
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one EXPLAIN for current index state only; no DROP/CREATE.",
    )
    args = parser.parse_args()
    url = require_database_url()

    slug = timestamp_slug()
    ensure_results_dirs()
    rows_out: list[dict[str, Any]] = []

    with psycopg.connect(url, autocommit=True) as conn:
        meta = server_metadata(conn)
        samples = fetch_sample_embeddings(conn, n=args.n_queries, seed=args.seed)
        if not samples:
            raise SystemExit("pattern_embeddings is empty; load data before benchmarking.")

        if args.dry_run:
            qv = format_vector_literal(samples[0][1])
            set_hnsw_ef_search(conn, args.ef_search)
            plan = run_explain_analyze(conn, INNER_KNN_SQL, {"qv": qv, "k": args.k})
            print(plan)
            path = write_explain_file("hnsw_vs_seqscan_dry", slug, plan)
            print(f"Wrote {path}")
            return

        exact_ids_per_query: list[list[int]] = []
        exact_exec_ms: list[float] = []

        drop_hnsw_index(conn)
        try:
            for i, (_eid, vec) in enumerate(samples):
                qv = format_vector_literal(vec)
                plan_ex = run_explain_analyze(conn, INNER_KNN_SQL, {"qv": qv, "k": args.k})
                _p, ex_ms = parse_explain_planning_execution_ms(plan_ex)
                if ex_ms is not None:
                    exact_exec_ms.append(ex_ms)
                hit, read = parse_buffers_shared_hit_read(plan_ex)
                exact_ids = knn_top_ids(conn, qv, args.k)
                exact_ids_per_query.append(exact_ids)
                write_explain_file(f"hnsw_vs_seqscan_exact_q{i}", slug, plan_ex)
                rows_out.append(
                    {
                        "phase": "exact_no_index",
                        "query_idx": i,
                        "execution_time_ms": ex_ms,
                        "shared_hit": hit,
                        "shared_read": read,
                        "recall_at_k": "",
                    }
                )
        finally:
            create_hnsw_index(conn)

        set_hnsw_ef_search(conn, args.ef_search)
        hnsw_exec_ms: list[float] = []
        recalls: list[float] = []

        for i, (_eid, vec) in enumerate(samples):
            qv = format_vector_literal(vec)
            plan_h = run_explain_analyze(conn, INNER_KNN_SQL, {"qv": qv, "k": args.k})
            _p, h_ms = parse_explain_planning_execution_ms(plan_h)
            if h_ms is not None:
                hnsw_exec_ms.append(h_ms)
            hit, read = parse_buffers_shared_hit_read(plan_h)
            approx_ids = knn_top_ids(conn, qv, args.k)
            truth = exact_ids_per_query[i]
            recalls.append(recall_at_k(approx_ids, truth))
            write_explain_file(f"hnsw_vs_seqscan_hnsw_q{i}", slug, plan_h)
            rows_out.append(
                {
                    "phase": "hnsw_index",
                    "query_idx": i,
                    "execution_time_ms": h_ms,
                    "shared_hit": hit,
                    "shared_read": read,
                    "recall_at_k": recalls[-1],
                }
            )

    p50_e, p95_e, p99_e = percentiles_p50_p95_p99(exact_exec_ms)
    p50_h, p95_h, p99_h = percentiles_p50_p95_p99(hnsw_exec_ms)
    summary = BenchRunMeta(
        script="hnsw_vs_seqscan",
        seed=args.seed,
        k=args.k,
        n_queries=len(samples),
        extra={
            "ef_search": args.ef_search,
            "exact_execution_ms_p50_p95_p99": [p50_e, p95_e, p99_e],
            "hnsw_execution_ms_p50_p95_p99": [p50_h, p95_h, p99_h],
            "recall_mean": float(sum(recalls) / len(recalls)) if recalls else None,
            "server": meta,
        },
    )

    csv_path = Path(__file__).resolve().parent / "results" / f"hnsw_vs_seqscan_{slug}.csv"
    json_path = csv_path.with_suffix(".json")
    fields = [
        "phase",
        "query_idx",
        "execution_time_ms",
        "shared_hit",
        "shared_read",
        "recall_at_k",
    ]
    write_csv(csv_path, fields, rows_out)
    write_json(json_path, {"summary": summary.as_dict(), "rows": rows_out})
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
