"""Pydantic request/response models for the REST API.

Schemas are organized by domain:
- OHLCV (time-series reads)
- Support/Resistance metadata
- RAC (pgvector retrieval + stored-procedure context)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


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


# --- RAC ---


class RacSimilarPatternsRequest(BaseModel):
    query_embedding: list[float] = Field(..., min_length=128, max_length=128)
    k: int = Field(default=20, ge=1, le=200)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filter_symbol: str | None = None


class RacNeighborRow(BaseModel):
    id: int
    symbol: str
    label: int | None
    future_return: float | None
    cosine_distance: float
    window_start: datetime
    window_end: datetime


class RacSimilarPatternsResponse(BaseModel):
    neighbors: list[RacNeighborRow]
    query_time_ms: float


class RacContextRequest(BaseModel):
    query_embedding: list[float] = Field(..., min_length=128, max_length=128)
    k: int = Field(default=20, ge=1, le=200)


class RacContextResponse(BaseModel):
    total_neighbors: int
    avg_cosine_dist: float | None
    label_distribution: dict[str, int] | None
    avg_future_return: float | None
    stddev_future_return: float | None
    dominant_label: int | None
    confidence: float | None


class RacFullContextRequest(BaseModel):
    query_embedding: list[float] = Field(..., min_length=128, max_length=128)
    symbol: str
    current_price: float
    k: int = Field(default=20, ge=1, le=200)


class RacFullContextResponse(BaseModel):
    total_neighbors: int
    avg_cosine_dist: float | None
    label_distribution: dict[str, int] | None
    avg_future_return: float | None
    stddev_future_return: float | None
    dominant_label: int | None
    knn_confidence: float | None
    dist_to_support: float | None
    dist_to_resistance: float | None
    sr_position_ratio: float | None
    neighbor_ids: list[int]


class RacPredictionRow(BaseModel):
    id: int
    predicted_label: int
    confidence_score: float | None
    k_neighbors: int
    avg_neighbor_dist: float | None
    neighbor_label_dist: dict[str, int] | None
    neighbor_ids: list[int] | None
    predicted_at: datetime