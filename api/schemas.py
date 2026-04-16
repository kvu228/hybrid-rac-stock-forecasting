"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


# --- OHLCV ---


class OHLCVRow(BaseModel):
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class SymbolItem(BaseModel):
    symbol: str
    row_count: int
    min_time: datetime
    max_time: datetime


# --- S/R Zones ---


class SRZoneRow(BaseModel):
    id: int
    symbol: str
    zone_type: str
    price_level: float
    strength: float | None
    detected_at: datetime
    is_active: bool


class SRDistanceResult(BaseModel):
    symbol: str
    current_price: float
    dist_to_support: float | None
    dist_to_resistance: float | None