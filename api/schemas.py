"""Pydantic request/response models for the REST API.

Schemas are organized by domain:
- OHLCV (time-series reads)
- Support/Resistance metadata
- RAC (pgvector retrieval + stored-procedure context)
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator


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


class PurgeInactiveSRRequest(BaseModel):
    """Delete historical S/R rows left after ``detect-sr`` (``is_active=FALSE``)."""

    all_inactive: bool = Field(
        default=False,
        description="If true, delete inactive rows for every symbol (omit ``symbols``).",
    )
    symbols: list[str] | None = Field(
        default=None,
        description="If set (non-empty), delete inactive rows only for these symbols.",
    )

    @model_validator(mode="after")
    def _one_mode(self) -> "PurgeInactiveSRRequest":
        if self.all_inactive:
            if self.symbols:
                raise ValueError("omit symbols when all_inactive is true")
            return self
        if not self.symbols:
            raise ValueError("provide a non-empty symbols list or set all_inactive to true")
        return self


class PurgeInactiveSRResponse(BaseModel):
    deleted_count: int


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


# --- ETL jobs ---


class EtlJobAcceptedResponse(BaseModel):
    job_id: str
    status: str


class EtlBackfillRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1)
    start: date
    end: date

    chunk_days: int = Field(default=365, ge=1, le=3650)
    concurrency: int = Field(default=1, ge=1, le=32)
    requests_per_minute: int = Field(default=55, ge=1, le=500)
    rate_limit_burst: int = Field(default=1, ge=1, le=50)


class EtlIncrementalRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1)
    end: date | None = Field(
        default=None,
        description="If omitted, server uses DB ``SELECT now()::date`` when the job runs.",
    )
    chunk_days: int = Field(default=365, ge=1, le=3650)
    concurrency: int = Field(default=1, ge=1, le=32)
    requests_per_minute: int = Field(default=55, ge=1, le=500)
    rate_limit_burst: int = Field(default=1, ge=1, le=50)


class EtlJobStatusResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    message: str
    result: dict[str, object] | None
    error: str | None
    created_at: datetime
    updated_at: datetime


# --- Benchmark ---


class BenchmarkExplainRequest(BaseModel):
    """Whitelisted EXPLAIN templates (no arbitrary SQL)."""

    query_kind: str = Field(
        ...,
        description="One of: hnsw_knn, seqscan_knn, hybrid_context (uses first row in pattern_embeddings).",
    )
    k: int = Field(default=20, ge=1, le=200)
    ef_search: int | None = Field(default=None, ge=1, le=1000, description="Optional SET hnsw.ef_search for this session.")


class BenchmarkExplainResponse(BaseModel):
    query_kind: str
    plan_text: str


class BenchmarkStatsResponse(BaseModel):
    available: bool
    hint: str | None = None
    statements: list[dict[str, object]] = Field(default_factory=list)


class BenchmarkResultItem(BaseModel):
    name: str
    size_bytes: int


# --- RAC query embedding ---


class RacQueryEmbeddingRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    window_end: datetime = Field(..., description="Last session timestamp of the 30-bar window (UTC).")


class OHLCVBar(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class RacQueryEmbeddingResponse(BaseModel):
    symbol: str
    window_start: datetime
    window_end: datetime
    query_embedding: list[float]
    ohlcv: list[OHLCVBar]