"""Integration tests for the FastAPI REST API.

Requires a running DB with migrations applied and seed data loaded.
"""

from __future__ import annotations

import os

from collections.abc import AsyncGenerator

import pytest
import httpx
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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_symbols(client: httpx.AsyncClient) -> None:
    _require_db()
    resp = await client.get("/api/symbols")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        item = data[0]
        assert "symbol" in item
        assert "row_count" in item
        assert "min_time" in item
        assert "max_time" in item


@pytest.mark.anyio
async def test_get_ohlcv_with_fixture_data(client: httpx.AsyncClient) -> None:
    """Query OHLCV for VCB (present in seed fixture)."""
    _require_db()
    resp = await client.get("/api/ohlcv/VCB", params={"limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        row = data[0]
        assert row["symbol"] == "VCB"
        for field in ("time", "open", "high", "low", "close", "volume"):
            assert field in row


@pytest.mark.anyio
async def test_get_ohlcv_latest(client: httpx.AsyncClient) -> None:
    _require_db()
    resp = await client.get("/api/ohlcv/VCB/latest", params={"n": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) <= 3
    # Verify oldest→newest ordering
    if len(data) >= 2:
        assert data[0]["time"] <= data[-1]["time"]


@pytest.mark.anyio
async def test_get_ohlcv_unknown_symbol(client: httpx.AsyncClient) -> None:
    _require_db()
    resp = await client.get("/api/ohlcv/__NONEXISTENT__")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Metadata — S/R zones
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_sr_zones_empty(client: httpx.AsyncClient) -> None:
    """Query S/R zones for a symbol that likely has none."""
    _require_db()
    resp = await client.get("/api/sr-zones/__NONEXISTENT__")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.anyio
async def test_get_sr_distance(client: httpx.AsyncClient) -> None:
    """Call the stored procedure — even with no zones it should return null distances."""
    _require_db()
    resp = await client.get("/api/sr-zones/__NONEXISTENT__/distance", params={"price": 100.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "__NONEXISTENT__"
    assert data["current_price"] == 100.0


@pytest.mark.anyio
async def test_sr_zones_roundtrip(client: httpx.AsyncClient) -> None:
    """Insert test S/R zones, query via API, then clean up."""
    _require_db()
    symbol = "__TEST_API_SR__"

    # Insert test zones directly
    with _engine().begin() as conn:
        conn.execute(
            sa.text("DELETE FROM support_resistance_zones WHERE symbol = :sym"),
            {"sym": symbol},
        )
        conn.execute(
            sa.text(
                "INSERT INTO support_resistance_zones (symbol, zone_type, price_level, strength, is_active) "
                "VALUES (:sym, 'SUPPORT', 90.0, 3.0, TRUE), (:sym, 'RESISTANCE', 110.0, 5.0, TRUE)"
            ),
            {"sym": symbol},
        )

    try:
        # List zones
        resp = await client.get(f"/api/sr-zones/{symbol}")
        assert resp.status_code == 200
        zones = resp.json()
        assert len(zones) == 2
        types = {z["zone_type"] for z in zones}
        assert types == {"SUPPORT", "RESISTANCE"}

        # Distance
        resp = await client.get(f"/api/sr-zones/{symbol}/distance", params={"price": 100.0})
        assert resp.status_code == 200
        dist = resp.json()
        assert dist["dist_to_support"] == pytest.approx(10.0)
        assert dist["dist_to_resistance"] == pytest.approx(10.0)
    finally:
        with _engine().begin() as conn:
            conn.execute(
                sa.text("DELETE FROM support_resistance_zones WHERE symbol = :sym"),
                {"sym": symbol},
            )


@pytest.mark.anyio
async def test_purge_inactive_sr_via_api(client: httpx.AsyncClient) -> None:
    """POST purge-inactive removes inactive rows for listed symbols."""
    _require_db()
    symbol = "__TEST_API_PURGE_SR__"

    with _engine().begin() as conn:
        conn.execute(sa.text("DELETE FROM support_resistance_zones WHERE symbol = :sym"), {"sym": symbol})
        conn.execute(
            sa.text(
                "INSERT INTO support_resistance_zones (symbol, zone_type, price_level, strength, is_active) VALUES "
                "(:sym, 'SUPPORT', 1.0, 1.0, FALSE), (:sym, 'RESISTANCE', 9.0, 2.0, TRUE)"
            ),
            {"sym": symbol},
        )

    try:
        resp = await client.post("/api/sr-zones/purge-inactive", json={"symbols": [symbol]})
        assert resp.status_code == 200
        assert resp.json() == {"deleted_count": 1}

        listed = await client.get(f"/api/sr-zones/{symbol}", params={"active_only": False})
        assert listed.status_code == 200
        zones = listed.json()
        assert len(zones) == 1
        assert zones[0]["is_active"] is True
    finally:
        with _engine().begin() as conn:
            conn.execute(sa.text("DELETE FROM support_resistance_zones WHERE symbol = :sym"), {"sym": symbol})


@pytest.mark.anyio
async def test_purge_inactive_sr_validation(client: httpx.AsyncClient) -> None:
    _require_db()
    resp = await client.post("/api/sr-zones/purge-inactive", json={})
    assert resp.status_code == 422