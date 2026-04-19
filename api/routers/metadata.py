"""Metadata endpoints.

Exposes support/resistance (S/R) zones and a helper endpoint that calls the
in-DB function `get_distance_to_nearest_sr()`.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from api.deps import DbConn
from api.schemas import PurgeInactiveSRRequest, PurgeInactiveSRResponse, SRDistanceResult, SRZoneRow

router = APIRouter()


@router.get("/sr-zones/{symbol}", response_model=list[SRZoneRow])
async def get_sr_zones(
    conn: DbConn,
    symbol: str,
    active_only: bool = Query(default=True, description="Return only active zones"),
) -> list[dict[str, object]]:
    """Return support/resistance zones for `symbol`."""
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
    """Call `get_distance_to_nearest_sr()` and return distances for `symbol`."""
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


@router.post("/sr-zones/purge-inactive", response_model=PurgeInactiveSRResponse)
async def purge_inactive_sr_zones(conn: DbConn, req: PurgeInactiveSRRequest) -> dict[str, int]:
    """DELETE ``support_resistance_zones`` rows where ``is_active`` is FALSE.

    ``detect-sr`` deactivates prior zones and inserts new ones; inactive rows
    otherwise stay in the table. Use this for periodic maintenance.

    Restrict in production (auth / internal-only), like any destructive admin API.
    """
    if req.all_inactive:
        cur = await conn.execute("DELETE FROM support_resistance_zones WHERE is_active = FALSE", {})
    else:
        cur = await conn.execute(
            "DELETE FROM support_resistance_zones WHERE is_active = FALSE AND symbol = ANY(%(symbols)s)",
            {"symbols": req.symbols},
        )
    rc = cur.rowcount
    deleted = int(rc) if rc is not None and rc >= 0 else 0
    return {"deleted_count": deleted}