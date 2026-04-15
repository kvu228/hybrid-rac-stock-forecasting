import os

import sqlalchemy as sa


def _engine() -> sa.Engine:
    url = os.environ.get("DATABASE_URL")
    assert url, "DATABASE_URL is required for DB integration tests"
    return sa.create_engine(url, pool_pre_ping=True)


def test_extensions_installed() -> None:
    with _engine().connect() as conn:
        rows = conn.execute(
            sa.text(
                """
SELECT extname
FROM pg_extension
WHERE extname IN ('timescaledb', 'vector')
ORDER BY extname;
"""
            )
        ).fetchall()
        assert [r[0] for r in rows] == ["timescaledb", "vector"]


def test_stock_ohlcv_is_hypertable() -> None:
    with _engine().connect() as conn:
        rows = conn.execute(
            sa.text(
                """
SELECT hypertable_name
FROM timescaledb_information.hypertables
WHERE hypertable_name = 'stock_ohlcv';
"""
            )
        ).fetchall()
        assert rows, "stock_ohlcv should be a TimescaleDB hypertable"


def test_hnsw_index_exists() -> None:
    with _engine().connect() as conn:
        rows = conn.execute(
            sa.text(
                """
SELECT indexname
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename = 'pattern_embeddings'
  AND indexname = 'idx_embedding_hnsw';
"""
            )
        ).fetchall()
        assert rows, "idx_embedding_hnsw should exist on pattern_embeddings"


def test_core_functions_exist() -> None:
    with _engine().connect() as conn:
        rows = conn.execute(
            sa.text(
                """
SELECT proname
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
  AND proname IN (
    'find_similar_patterns',
    'compute_rac_context',
    'get_distance_to_nearest_sr',
    'compute_full_rac_context'
  )
ORDER BY proname;
"""
            )
        ).fetchall()

        assert [r[0] for r in rows] == [
            "compute_full_rac_context",
            "compute_rac_context",
            "find_similar_patterns",
            "get_distance_to_nearest_sr",
        ]

