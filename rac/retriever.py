"""RAC retriever: pgvector KNN search via stored procedures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import psycopg


@dataclass(frozen=True)
class Neighbor:
    id: int
    symbol: str
    label: int | None
    future_return: float | None
    cosine_distance: float
    window_start: datetime
    window_end: datetime


def _format_vector(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


async def find_similar_patterns(
    conn: psycopg.AsyncConnection[tuple[object, ...]],
    *,
    query_embedding: list[float],
    k: int = 20,
    similarity_threshold: float = 0.7,
    filter_symbol: str | None = None,
) -> list[Neighbor]:
    """Call `find_similar_patterns()` and return neighbors."""
    qv = _format_vector(query_embedding)
    rows = await (
        await conn.execute(
            """
            SELECT *
            FROM find_similar_patterns(
                %(query_vec)s::vector,
                %(k_neighbors)s,
                %(similarity_threshold)s,
                %(filter_symbol)s
            )
            """,
            {
                "query_vec": qv,
                "k_neighbors": k,
                "similarity_threshold": similarity_threshold,
                "filter_symbol": filter_symbol,
            },
        )
    ).fetchall()

    out: list[Neighbor] = []
    for r in rows:
        rr = cast(tuple[Any, ...], r)
        out.append(
            Neighbor(
                id=int(rr[0]),
                symbol=str(rr[1]),
                label=int(rr[2]) if rr[2] is not None else None,
                future_return=float(rr[3]) if rr[3] is not None else None,
                cosine_distance=float(rr[4]),
                window_start=cast(datetime, rr[5]),
                window_end=cast(datetime, rr[6]),
            )
        )
    return out

