"""Metadata endpoints: S/R zones and distance calculations."""

from __future__ import annotations

from fastapi import APIRouter, Query

from api.deps import DbConn
from api.schemas import SRDistanceResult, SRZoneRow

router = APIRouter()


@router.get("/sr-zones/{symbol}", response_model=list[SRZoneRow])
async def get_sr_zones(
    conn: DbConn,
    symbol: str,
    active_only: bool = Query(default=True, description="Return only active zones"),
) -> list[dict[str, object]]:
    """Return support/resistance zones for a symbol."""
    if active_only:
        query = """
            SELECT id, symbol, zone_type, price_level, strength, detected_at, is_active
            FROM support_resistance_zones
            WHERE symbol = %(symbol)s AND is_active = TRUE
            ORDER BY price_level
        """
    else:
        query = """
            SELECT id, symbol, zone_type, price_level, strength, detected_at, is_active
            FROM support_resistance_zones
            WHERE symbol = %(symbol)s
            ORDER BY price_level
        """

    rows = await (await conn.execute(query, {"symbol": symbol})).fetchall()
    return [
        {
            "id": r[0],
            "symbol": r[1],
            "zone_type": r[2],
            "price_level": r[3],
            "strength": r[4],
            "detected_at": r[5],
            "is_active": r[6],
        }
        for r in rows
    ]


@router.get("/sr-zones/{symbol}/distance", response_model=SRDistanceResult)
async def get_sr_distance(
    conn: DbConn,
    symbol: str,
    price: float = Query(..., description="Current price to measure distance from"),
) -> dict[str, object]:
    """Call stored procedure ``get_distance_to_nearest_sr()`` and return distances."""
    row = await (
        await conn.execute(
            "SELECT * FROM get_distance_to_nearest_sr(%(symbol)s, %(price)s)",
            {"symbol": symbol, "price": price},
        )
    ).fetchone()

    dist_support = row[0] if row else None
    dist_resistance = row[1] if row else None

    return {
        "symbol": symbol,
        "current_price": price,
        "dist_to_support": dist_support,
        "dist_to_resistance": dist_resistance,
    }