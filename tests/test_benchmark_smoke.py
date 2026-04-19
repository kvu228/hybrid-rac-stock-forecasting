"""Lightweight tests for benchmark helpers (no heavy DB work unless DATABASE_URL is set)."""

from __future__ import annotations

import os

import pytest

from benchmark.common import (
    connect,
    parse_buffers_shared_hit_read,
    parse_explain_planning_execution_ms,
    percentile_sorted,
    percentiles_p50_p95_p99,
    recall_at_k,
    server_metadata,
)


def test_parse_explain_planning_execution_ms() -> None:
    text = """
Seq Scan on pattern_embeddings
Planning Time: 0.123 ms
Execution Time: 45.678 ms
""".strip()
    p, e = parse_explain_planning_execution_ms(text)
    assert p == pytest.approx(0.123)
    assert e == pytest.approx(45.678)


def test_parse_buffers_shared_hit_read() -> None:
    text = "  Buffers: shared hit=100 shared read=5"
    hit, read = parse_buffers_shared_hit_read(text)
    assert hit == 100
    assert read == 5


def test_percentiles_and_recall() -> None:
    assert percentile_sorted([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    p50, p95, p99 = percentiles_p50_p95_p99([1, 2, 3, 4, 100])
    assert p50 == pytest.approx(3.0)
    assert recall_at_k([1, 2, 3], [1, 2, 9]) == pytest.approx(2 / 3)


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_server_metadata_live() -> None:
    with connect(autocommit=True) as conn:
        meta = server_metadata(conn)
        assert "postgresql_version" in meta
        assert "extensions" in meta
