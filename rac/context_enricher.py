"""Context enrichment wrappers for RAC stored procedures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import psycopg


def _format_vector(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


@dataclass(frozen=True)
class RacContext:
    total_neighbors: int
    avg_cosine_dist: float | None
    label_distribution: dict[str, int] | None
    avg_future_return: float | None
    stddev_future_return: float | None
    dominant_label: int | None
    confidence: float | None


@dataclass(frozen=True)
class FullRacContext:
    # Block 1
    total_neighbors: int
    avg_cosine_dist: float | None
    label_distribution: dict[str, int] | None
    avg_future_return: float | None
    stddev_future_return: float | None
    dominant_label: int | None
    knn_confidence: float | None
    # Block 2
    dist_to_support: float | None
    dist_to_resistance: float | None
    sr_position_ratio: float | None
    # Block 3
    neighbor_ids: list[int]


async def compute_rac_context(
    conn: psycopg.AsyncConnection[tuple[object, ...]],
    *,
    query_embedding: list[float],
    k: int = 20,
) -> RacContext:
    qv = _format_vector(query_embedding)
    row = await (
        await conn.execute(
            "SELECT * FROM compute_rac_context(%(query_vec)s::vector, %(k)s)",
            {"query_vec": qv, "k": k},
        )
    ).fetchone()
    if row is None:
        return RacContext(0, None, None, None, None, None, None)

    rr = cast(tuple[Any, ...], row)
    label_dist = cast(dict[str, int], rr[2]) if isinstance(rr[2], dict) else None
    return RacContext(
        total_neighbors=int(rr[0]),
        avg_cosine_dist=float(rr[1]) if rr[1] is not None else None,
        label_distribution=label_dist,
        avg_future_return=float(rr[3]) if rr[3] is not None else None,
        stddev_future_return=float(rr[4]) if rr[4] is not None else None,
        dominant_label=int(rr[5]) if rr[5] is not None else None,
        confidence=float(rr[6]) if rr[6] is not None else None,
    )


async def compute_full_rac_context(
    conn: psycopg.AsyncConnection[tuple[object, ...]],
    *,
    query_embedding: list[float],
    symbol: str,
    current_price: float,
    k: int = 20,
) -> FullRacContext:
    qv = _format_vector(query_embedding)
    row = await (
        await conn.execute(
            """
            SELECT * FROM compute_full_rac_context(
                %(query_vec)s::vector,
                %(symbol)s,
                %(current_price)s,
                %(k)s
            )
            """,
            {"query_vec": qv, "symbol": symbol, "current_price": current_price, "k": k},
        )
    ).fetchone()
    if row is None:
        return FullRacContext(0, None, None, None, None, None, None, None, None, None, [])

    rr = cast(tuple[Any, ...], row)
    label_dist = cast(dict[str, int], rr[2]) if isinstance(rr[2], dict) else None
    neighbor_ids = cast(list[Any], rr[10]) if rr[10] is not None else []
    return FullRacContext(
        total_neighbors=int(rr[0]),
        avg_cosine_dist=float(rr[1]) if rr[1] is not None else None,
        label_distribution=label_dist,
        avg_future_return=float(rr[3]) if rr[3] is not None else None,
        stddev_future_return=float(rr[4]) if rr[4] is not None else None,
        dominant_label=int(rr[5]) if rr[5] is not None else None,
        knn_confidence=float(rr[6]) if rr[6] is not None else None,
        dist_to_support=float(rr[7]) if rr[7] is not None else None,
        dist_to_resistance=float(rr[8]) if rr[8] is not None else None,
        sr_position_ratio=float(rr[9]) if rr[9] is not None else None,
        neighbor_ids=[int(x) for x in neighbor_ids],
    )

