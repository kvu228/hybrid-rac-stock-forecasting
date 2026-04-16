"""Support / Resistance zone detection and ingestion into PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import psycopg


@dataclass(frozen=True)
class SRZone:
    symbol: str
    zone_type: str  # "SUPPORT" or "RESISTANCE"
    price_level: float
    strength: float  # number of times price tested this level


def _normalize_psycopg_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url.removeprefix("postgresql+psycopg://")
    return database_url


def detect_pivot_points(
    df: pd.DataFrame,
    *,
    order: int = 5,
) -> list[SRZone]:
    """Detect support and resistance zones using local pivot highs/lows.

    A **pivot high** at index *i* means ``high[i]`` is the maximum in the
    window ``[i-order, i+order]``.  Similarly a **pivot low** uses ``low[i]``
    as the minimum.  Nearby pivots are clustered (within ``tolerance`` of
    each other) to form zones, and each cluster's strength equals the number
    of pivots that fell into it.

    Args:
        df: OHLCV DataFrame for a *single symbol*, sorted by time.
            Must contain columns: symbol, high, low, close.
        order: Half-window size for pivot detection (default 5 → 11-bar window).

    Returns:
        List of ``SRZone`` records.
    """
    if len(df) < 2 * order + 1:
        return []

    symbol = str(df["symbol"].iloc[0])
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)

    pivot_highs: list[float] = []
    pivot_lows: list[float] = []

    for i in range(order, len(df) - order):
        window_h = highs[i - order : i + order + 1]
        if highs[i] == window_h.max():
            pivot_highs.append(float(highs[i]))

        window_l = lows[i - order : i + order + 1]
        if lows[i] == window_l.min():
            pivot_lows.append(float(lows[i]))

    # Use 1.5% of the median close as default clustering tolerance
    median_close = float(np.median(df["close"].values))
    tolerance = median_close * 0.015

    zones: list[SRZone] = []
    zones.extend(_cluster_levels(pivot_highs, symbol, "RESISTANCE", tolerance))
    zones.extend(_cluster_levels(pivot_lows, symbol, "SUPPORT", tolerance))
    return zones


def _cluster_levels(
    levels: list[float],
    symbol: str,
    zone_type: str,
    tolerance: float,
) -> list[SRZone]:
    """Cluster nearby price levels into zones.

    Simple greedy algorithm: sort levels, walk through them and merge any
    level within ``tolerance`` of the running cluster mean.
    """
    if not levels:
        return []

    sorted_lvls = sorted(levels)
    clusters: list[list[float]] = [[sorted_lvls[0]]]

    for lvl in sorted_lvls[1:]:
        cluster_mean = float(np.mean(clusters[-1]))
        if abs(lvl - cluster_mean) <= tolerance:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    return [
        SRZone(
            symbol=symbol,
            zone_type=zone_type,
            price_level=float(np.mean(c)),
            strength=float(len(c)),
        )
        for c in clusters
    ]


def detect_sr_zones(df: pd.DataFrame, *, order: int = 5) -> list[SRZone]:
    """Detect S/R zones across all symbols in a DataFrame.

    Delegates to :func:`detect_pivot_points` per symbol.
    """
    zones: list[SRZone] = []
    for _symbol, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("time").reset_index(drop=True)
        zones.extend(detect_pivot_points(grp, order=order))
    return zones


def ingest_sr_zones(zones: list[SRZone], database_url: str) -> int:
    """Insert S/R zones into ``support_resistance_zones``.

    Deactivates previous zones for affected symbols before inserting new ones
    so the table always reflects the latest detection run.

    Returns:
        Number of rows inserted.
    """
    if not zones:
        return 0

    symbols = list({z.symbol for z in zones})

    with psycopg.connect(_normalize_psycopg_url(database_url)) as conn:
        with conn.cursor() as cur:
            # Deactivate old zones for these symbols
            cur.execute(
                "UPDATE support_resistance_zones SET is_active = FALSE WHERE symbol = ANY(%s)",
                (symbols,),
            )

            for z in zones:
                cur.execute(
                    """
INSERT INTO support_resistance_zones (symbol, zone_type, price_level, strength, is_active)
VALUES (%s, %s, %s, %s, TRUE)
""",
                    (z.symbol, z.zone_type, z.price_level, z.strength),
                )

        conn.commit()

    return len(zones)