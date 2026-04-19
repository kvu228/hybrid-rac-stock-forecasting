"""Sweep pgvector HNSW build params (m, ef_construction) and query ef_search; measure recall vs exact."""

from __future__ import annotations

import argparse
import time
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
    hnsw_index_ddl,
    index_size_bytes,
    knn_top_ids,
    percentiles_p50_p95_p99,
    recall_at_k,
    require_database_url,
    server_metadata,
    set_hnsw_ef_search,
    timestamp_slug,
    write_csv,
    write_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-queries", type=int, default=12, help="Queries per ef_search cell (keep small).")
    p.add_argument("--quick", action="store_true", help="Smaller grid for smoke runs.")
    args = p.parse_args()
    url = require_database_url()
    slug = timestamp_slug()
    ensure_results_dirs()

    if args.quick:
        m_list = [8, 16]
        ef_c_list = [64, 128]
        ef_s_list = [40, 100]
    else:
        m_list = [8, 16, 32, 64]
        ef_c_list = [64, 128, 200, 400]
        ef_s_list = [40, 100, 200, 400]

    k_recall = 20
    rows_out: list[dict[str, Any]] = []

    with psycopg.connect(url, autocommit=True) as conn:
        meta = server_metadata(conn)
        samples = fetch_sample_embeddings(conn, n=args.n_queries, seed=args.seed)
        if not samples:
            raise SystemExit("pattern_embeddings is empty.")

        exact_per_q: list[list[int]] = []
        drop_hnsw_index(conn)
        try:
            for _eid, vec in samples:
                qv = format_vector_literal(vec)
                exact_per_q.append(knn_top_ids(conn, qv, k_recall))
        finally:
            create_hnsw_index(conn)

        for m in m_list:
            for ef_c in ef_c_list:
                drop_hnsw_index(conn)
                t0 = time.perf_counter()
                with conn.cursor() as cur:
                    cur.execute(hnsw_index_ddl(m, ef_c))
                build_s = time.perf_counter() - t0
                idx_bytes = index_size_bytes(conn) or 0

                for ef_s in ef_s_list:
                    set_hnsw_ef_search(conn, ef_s)
                    latencies: list[float] = []
                    recalls10: list[float] = []
                    recalls20: list[float] = []
                    for qi, (_eid, vec) in enumerate(samples):
                        qv = format_vector_literal(vec)
                        tq0 = time.perf_counter()
                        approx = knn_top_ids(conn, qv, k_recall)
                        latencies.append((time.perf_counter() - tq0) * 1000.0)
                        truth = exact_per_q[qi]
                        recalls10.append(recall_at_k(approx[:10], truth[:10]))
                        recalls20.append(recall_at_k(approx, truth))

                    p50, p95, p99 = percentiles_p50_p95_p99(latencies)
                    rows_out.append(
                        {
                            "m": m,
                            "ef_construction": ef_c,
                            "ef_search": ef_s,
                            "index_build_s": round(build_s, 3),
                            "index_size_bytes": idx_bytes,
                            "query_ms_p50": round(p50, 4),
                            "query_ms_p95": round(p95, 4),
                            "query_ms_p99": round(p99, 4),
                            "recall_at_10_mean": round(float(sum(recalls10) / len(recalls10)), 6),
                            "recall_at_20_mean": round(float(sum(recalls20) / len(recalls20)), 6),
                        }
                    )

        create_hnsw_index(conn)

    summary = BenchRunMeta(
        script="hnsw_param_sweep",
        seed=args.seed,
        k=k_recall,
        n_queries=len(samples),
        extra={
            "grid_m": m_list,
            "grid_ef_construction": ef_c_list,
            "grid_ef_search": ef_s_list,
            "quick": args.quick,
            "server": meta,
        },
    )
    base = Path(__file__).resolve().parent / "results" / f"hnsw_param_sweep_{slug}"
    fields = list(rows_out[0].keys()) if rows_out else []
    write_csv(base.with_suffix(".csv"), fields, rows_out)
    write_json(base.with_suffix(".json"), {"summary": summary.as_dict(), "rows": rows_out})
    print(f"Wrote {base.with_suffix('.csv')} and {base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
