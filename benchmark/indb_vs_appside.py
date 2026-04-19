"""Compare compute_rac_context() in-DB vs fetching K rows + Python aggregation."""

from __future__ import annotations

import argparse
import pickle
import time
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import psycopg

from benchmark.common import (
    BenchRunMeta,
    ensure_results_dirs,
    fetch_sample_embeddings,
    format_vector_literal,
    require_database_url,
    server_metadata,
    set_hnsw_ef_search,
    timestamp_slug,
    write_csv,
    write_json,
)


def compute_rac_python(
    rows: list[tuple[Any, ...]],
    *,
    k_neighbors: int,
) -> dict[str, Any]:
    """Mirror compute_rac_context from stored proc (same semantics for non-null labels)."""
    # id, label, future_return, embedding, cos_dist
    labels = [int(r[1]) for r in rows if r[1] is not None]
    returns = [float(r[2]) for r in rows if r[2] is not None]
    dists = [float(r[4]) for r in rows]
    n = len(rows)
    ctr = Counter(labels)
    label_dist = {str(k): v for k, v in sorted(ctr.items())}
    dom_label: int | None = None
    conf = 0.0
    if ctr:
        dom_label, mc = ctr.most_common(1)[0]
        conf = float(mc) / float(k_neighbors)
    std_ret: float | None
    if len(returns) > 1:
        std_ret = stdev(returns)
    elif len(returns) == 1:
        std_ret = 0.0
    else:
        std_ret = None
    return {
        "total_neighbors": n,
        "avg_cosine_dist": mean(dists) if dists else None,
        "label_distribution": label_dist,
        "avg_future_return": mean(returns) if returns else None,
        "stddev_future_return": std_ret,
        "dominant_label": dom_label,
        "confidence": conf,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ef-search", type=int, default=100)
    args = p.parse_args()
    url = require_database_url()
    slug = timestamp_slug()
    ensure_results_dirs()
    rows_csv: list[dict[str, Any]] = []

    sql_app = """
        SELECT id, label, future_return, embedding, embedding <=> %(qv)s::vector AS cos_dist
        FROM pattern_embeddings
        ORDER BY embedding <=> %(qv)s::vector
        LIMIT %(k)s;
    """
    sql_indb = "SELECT * FROM compute_rac_context(%(qv)s::vector, %(k)s);"

    with psycopg.connect(url, autocommit=True) as conn:
        meta = server_metadata(conn)
        samples = fetch_sample_embeddings(conn, n=args.n_queries, seed=args.seed)
        if not samples:
            raise SystemExit("pattern_embeddings is empty.")
        set_hnsw_ef_search(conn, args.ef_search)

        for i, (_eid, vec) in enumerate(samples):
            qv = format_vector_literal(vec)

            t0 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(sql_app, {"qv": qv, "k": args.k})
                app_rows = cur.fetchall()
            app_ms = (time.perf_counter() - t0) * 1000.0
            app_bytes = len(pickle.dumps(app_rows, protocol=pickle.HIGHEST_PROTOCOL))
            py_stats = compute_rac_python(app_rows, k_neighbors=args.k)

            t1 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(sql_indb, {"qv": qv, "k": args.k})
                indb_rows = cur.fetchall()
            indb_ms = (time.perf_counter() - t1) * 1000.0
            indb_bytes = len(pickle.dumps(indb_rows, protocol=pickle.HIGHEST_PROTOCOL))

            rows_csv.append(
                {
                    "query_idx": i,
                    "variant": "app_side",
                    "wall_ms": round(app_ms, 3),
                    "payload_bytes": app_bytes,
                    "round_trips": 1,
                    "total_neighbors": py_stats["total_neighbors"],
                }
            )
            rows_csv.append(
                {
                    "query_idx": i,
                    "variant": "in_db",
                    "wall_ms": round(indb_ms, 3),
                    "payload_bytes": indb_bytes,
                    "round_trips": 1,
                    "total_neighbors": indb_rows[0][0] if indb_rows else None,
                }
            )

    summary = BenchRunMeta(
        script="indb_vs_appside",
        seed=args.seed,
        k=args.k,
        n_queries=len(samples),
        extra={"server": meta},
    )
    base = Path(__file__).resolve().parent / "results" / f"indb_vs_appside_{slug}"
    write_csv(
        base.with_suffix(".csv"),
        ["query_idx", "variant", "wall_ms", "payload_bytes", "round_trips", "total_neighbors"],
        rows_csv,
    )
    write_json(base.with_suffix(".json"), {"summary": summary.as_dict(), "rows": rows_csv})
    print(f"Wrote {base.with_suffix('.csv')} and {base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
