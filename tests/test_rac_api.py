"""Integration tests for RAC endpoints.

Requires a running DB with migrations applied.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import httpx
import pytest
import sqlalchemy as sa

from api.main import app, lifespan


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient]:
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; skipping DB integration tests")
    return url


def _engine() -> sa.Engine:
    return sa.create_engine(_database_url(), pool_pre_ping=True)


def _require_db() -> None:
    try:
        with _engine().connect() as _:
            return
    except Exception:
        pytest.skip("Database not reachable; skipping DB integration tests")


def _vec(i: int, dim: int = 128) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


@pytest.mark.anyio
async def test_rac_similar_and_full_context_and_prediction_roundtrip(client: httpx.AsyncClient) -> None:
    _require_db()
    symbol = "__TEST_RAC__"

    v0 = _vec(0)
    v1 = _vec(1)
    v2 = _vec(2)

    # Insert a few embeddings directly.
    with _engine().begin() as conn:
        conn.execute(sa.text("DELETE FROM rac_predictions WHERE query_embedding_id IN (SELECT id FROM pattern_embeddings WHERE symbol = :sym)"), {"sym": symbol})
        conn.execute(sa.text("DELETE FROM pattern_embeddings WHERE symbol = :sym"), {"sym": symbol})

        # Minimal window metadata; we don't need real OHLCV for stored-proc tests.
        conn.execute(
            sa.text(
                """
                INSERT INTO pattern_embeddings (symbol, window_start, window_end, embedding, label, future_return)
                VALUES
                  (:sym, NOW() - INTERVAL '40 days', NOW() - INTERVAL '11 days', CAST(:e0 AS vector), 2, 0.03),
                  (:sym, NOW() - INTERVAL '39 days', NOW() - INTERVAL '10 days', CAST(:e1 AS vector), 1, 0.00),
                  (:sym, NOW() - INTERVAL '38 days', NOW() - INTERVAL '9 days',  CAST(:e2 AS vector), 0, -0.02)
                """
            ),
            {
                "sym": symbol,
                "e0": "[" + ",".join(str(x) for x in v0) + "]",
                "e1": "[" + ",".join(str(x) for x in v1) + "]",
                "e2": "[" + ",".join(str(x) for x in v2) + "]",
            },
        )

    # similar-patterns: with threshold 0.0, only the identical vector should match (dist=0 < 1.0)
    resp = await client.post(
        "/api/rac/similar-patterns",
        json={"query_embedding": v0, "k": 5, "similarity_threshold": 0.0, "filter_symbol": symbol},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "neighbors" in data
    assert data["neighbors"]
    assert data["neighbors"][0]["symbol"] == symbol
    assert data["neighbors"][0]["cosine_distance"] == pytest.approx(0.0)

    # full-context should return neighbor_ids including the inserted embeddings
    resp = await client.post(
        "/api/rac/full-context",
        json={"query_embedding": v0, "symbol": symbol, "current_price": 100.0, "k": 3},
    )
    assert resp.status_code == 200
    ctx = resp.json()
    assert ctx["total_neighbors"] == 3
    assert len(ctx["neighbor_ids"]) == 3

    # predict should persist into rac_predictions (linked to closest neighbor id)
    resp = await client.post(
        "/api/rac/predict",
        json={"query_embedding": v0, "symbol": symbol, "current_price": 100.0, "k": 3},
    )
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["predicted_label"] in (0, 1, 2)

    resp = await client.get(f"/api/rac/predictions/{symbol}", params={"limit": 5})
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list)
    assert rows, "Expected at least one persisted prediction row"

