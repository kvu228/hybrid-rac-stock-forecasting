"""OHLCV endpoints: symbol listing, time-range query, latest sessions."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from api.deps import DbConn
from api.schemas import OHLCVRow, SymbolItem

router = APIRouter()


@router.get("/symbols", response_model=list[SymbolItem])
async def list_symbols(conn: DbConn) -> list[dict[str, object]]:
    """Return all symbols with row counts and date ranges."""
    rows = await conn.execute(
        """
        SELECT symbol, COUNT(*) AS row_count, MIN(time) AS min_time, MAX(time) AS max_time
        FROM stock_ohlcv
        GROUP BY symbol
        ORDER BY symbol
        """
    )
    return [
        {"symbol": r[0], "row_count": r[1], "min_time": r[2], "max_time": r[3]}
        for r in await rows.fetchall()
    ]


@router.get("/ohlcv/{symbol}", response_model=list[OHLCVRow])
async def get_ohlcv(
    conn: DbConn,
    symbol: str,
    start: date | None = Query(default=None, description="Start date (inclusive)"),
    end: date | None = Query(default=None, description="End date (inclusive)"),
    limit: int = Query(default=5000, ge=1, le=50000, description="Max rows"),
) -> list[dict[str, object]]:
    """Return OHLCV rows for a symbol within an optional date range."""
    clauses = ["symbol = %(symbol)s"]
    params: dict[str, object] = {"symbol": symbol, "limit": limit}

    if start is not None:
        clauses.append("time >= %(start)s")
        params["start"] = str(start)
    if end is not None:
        clauses.append("time <= %(end)s")
        params["end"] = str(end)

    where = " AND ".join(clauses)
    query = f"SELECT time, symbol, open, high, low, close, volume FROM stock_ohlcv WHERE {where} ORDER BY time LIMIT %(limit)s"  # noqa: S608

    rows = await (await conn.execute(query, params)).fetchall()
    return [
        {
            "time": r[0],
            "symbol": r[1],
            "open": r[2],
            "high": r[3],
            "low": r[4],
            "close": r[5],
            "volume": r[6],
        }
        for r in rows
    ]


@router.get("/ohlcv/{symbol}/latest", response_model=list[OHLCVRow])
async def get_ohlcv_latest(
    conn: DbConn,
    symbol: str,
    n: int = Query(default=30, ge=1, le=1000, description="Number of latest sessions"),
) -> list[dict[str, object]]:
    """Return the N most recent OHLCV rows for a symbol (ordered oldest→newest)."""
    rows = await (
        await conn.execute(
            """
            SELECT time, symbol, open, high, low, close, volume
            FROM stock_ohlcv
            WHERE symbol = %(symbol)s
            ORDER BY time DESC
            LIMIT %(n)s
            """,
            {"symbol": symbol, "n": n},
        )
    ).fetchall()

    # Reverse to oldest→newest for charting convenience
    rows = list(reversed(rows))
    return [
        {
            "time": r[0],
            "symbol": r[1],
            "open": r[2],
            "high": r[3],
            "low": r[4],
            "close": r[5],
            "volume": r[6],
        }
        for r in rows
    ]