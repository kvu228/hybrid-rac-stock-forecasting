"""RAC endpoints.

These endpoints intentionally delegate most computation to PostgreSQL stored
procedures to demonstrate:
- pgvector KNN (HNSW) retrieval
- in-DB aggregation of neighbor statistics
- a hybrid query plan that combines vector search with structured metadata
  lookups (S/R zones) in a single round-trip.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query

from api.deps import DbConn
from api.schemas import (
    RacContextRequest,
    RacContextResponse,
    RacFullContextRequest,
    RacFullContextResponse,
    RacPredictionRow,
    RacQueryEmbeddingRequest,
    RacQueryEmbeddingResponse,
    RacSimilarPatternsRequest,
    RacSimilarPatternsResponse,
)
from rac.context_enricher import FullRacContext, compute_full_rac_context, compute_rac_context
from rac.query_window import (
    build_normalized_query_window,
    embedding_from_normalized_window,
    ohlcv_rows_to_dataframe,
)
from rac.rac_classifier import predict_and_persist
from rac.retriever import find_similar_patterns

router = APIRouter()


def _full_context_api_dict(ctx: FullRacContext) -> dict[str, object]:
    """JSON-friendly ``FullRacContext`` (neighbor_label_distances as dict rows)."""
    return {
        "total_neighbors": ctx.total_neighbors,
        "avg_cosine_dist": ctx.avg_cosine_dist,
        "label_distribution": ctx.label_distribution,
        "avg_future_return": ctx.avg_future_return,
        "stddev_future_return": ctx.stddev_future_return,
        "dominant_label": ctx.dominant_label,
        "knn_confidence": ctx.knn_confidence,
        "dist_to_support": ctx.dist_to_support,
        "dist_to_resistance": ctx.dist_to_resistance,
        "sr_position_ratio": ctx.sr_position_ratio,
        "neighbor_ids": ctx.neighbor_ids,
        "neighbor_label_distances": [
            {"label": lab, "cosine_distance": d} for lab, d in ctx.neighbor_label_distances
        ],
    }


@router.post("/rac/query-embedding", response_model=RacQueryEmbeddingResponse)
async def rac_query_embedding(conn: DbConn, req: RacQueryEmbeddingRequest) -> dict[str, object]:
    """Build a 30-session normalized window ending at ``window_end`` and encode with the CNN."""
    model_path = Path(os.environ.get("CNN_ENCODER_PATH", "ml/model_store/cnn_encoder.pt"))
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"CNN encoder weights not found at {model_path}. Train with ml/train_pipeline.py or set CNN_ENCODER_PATH.",
        )

    symbol = req.symbol.strip().upper()
    cur = await conn.execute(
        """
        SELECT time, symbol, open, high, low, close, volume
        FROM stock_ohlcv
        WHERE symbol = %(sym)s AND time <= %(end)s
        ORDER BY time DESC
        LIMIT 500
        """,
        {"sym": symbol, "end": req.window_end},
    )
    rows = list(reversed(await cur.fetchall()))
    df = ohlcv_rows_to_dataframe(rows)
    try:
        normed, w_start, w_end, slice_df = build_normalized_query_window(df, req.window_end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    device = (os.environ.get("TORCH_DEVICE") or "cpu").strip() or "cpu"
    emb = await asyncio.to_thread(embedding_from_normalized_window, normed, model_path, device=device)

    ohlcv: list[dict[str, object]] = []
    for _, r in slice_df.iterrows():
        ts = r["time"]
        ohlcv.append(
            {
                "time": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r["volume"]),
            }
        )

    return {
        "symbol": symbol,
        "window_start": w_start,
        "window_end": w_end,
        "query_embedding": emb,
        "ohlcv": ohlcv,
    }


@router.post("/rac/similar-patterns", response_model=RacSimilarPatternsResponse)
async def rac_similar_patterns(conn: DbConn, req: RacSimilarPatternsRequest) -> dict[str, object]:
    """Return Top-K neighbors by cosine distance using `find_similar_patterns()`."""
    start = time.perf_counter()
    neighbors = await find_similar_patterns(
        conn,
        query_embedding=req.query_embedding,
        k=req.k,
        similarity_threshold=req.similarity_threshold,
        filter_symbol=req.filter_symbol,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "neighbors": [
            {
                "id": n.id,
                "symbol": n.symbol,
                "label": n.label,
                "future_return": n.future_return,
                "cosine_distance": n.cosine_distance,
                "window_start": n.window_start,
                "window_end": n.window_end,
            }
            for n in neighbors
        ],
        "query_time_ms": elapsed_ms,
    }


@router.post("/rac/context", response_model=RacContextResponse)
async def rac_context(conn: DbConn, req: RacContextRequest) -> dict[str, object]:
    """Return aggregated KNN statistics using `compute_rac_context()`."""
    ctx = await compute_rac_context(conn, query_embedding=req.query_embedding, k=req.k)
    return {
        "total_neighbors": ctx.total_neighbors,
        "avg_cosine_dist": ctx.avg_cosine_dist,
        "label_distribution": ctx.label_distribution,
        "avg_future_return": ctx.avg_future_return,
        "stddev_future_return": ctx.stddev_future_return,
        "dominant_label": ctx.dominant_label,
        "confidence": ctx.confidence,
    }


@router.post("/rac/full-context", response_model=RacFullContextResponse)
async def rac_full_context(conn: DbConn, req: RacFullContextRequest) -> dict[str, object]:
    """Return hybrid RAC context using `compute_full_rac_context()`."""
    ctx = await compute_full_rac_context(
        conn,
        query_embedding=req.query_embedding,
        symbol=req.symbol,
        current_price=req.current_price,
        k=req.k,
    )
    return _full_context_api_dict(ctx)


@router.post("/rac/predict", response_model=dict[str, object])
async def rac_predict(
    conn: DbConn,
    req: RacFullContextRequest,
    persist: bool = Query(default=True, description="Persist into rac_predictions"),
) -> dict[str, object]:
    """Predict a label from RAC context and optionally persist the result."""
    if not persist:
        ctx = await compute_full_rac_context(
            conn,
            query_embedding=req.query_embedding,
            symbol=req.symbol,
            current_price=req.current_price,
            k=req.k,
        )
        pred_label = int(ctx.dominant_label) if ctx.dominant_label is not None else 1
        conf = float(ctx.knn_confidence) if ctx.knn_confidence is not None else None
        return {"predicted_label": pred_label, "confidence_score": conf, "context": _full_context_api_dict(ctx)}

    res = await predict_and_persist(
        conn,
        query_embedding=req.query_embedding,
        symbol=req.symbol,
        current_price=req.current_price,
        k=req.k,
    )
    return {
        "predicted_label": res.predicted_label,
        "confidence_score": res.confidence_score,
        "k_neighbors": res.k_neighbors,
        "avg_neighbor_dist": res.avg_neighbor_dist,
        "neighbor_label_dist": res.neighbor_label_dist,
        "neighbor_ids": res.neighbor_ids,
        "context": _full_context_api_dict(res.context),
    }


@router.get("/rac/predictions/{symbol}", response_model=list[RacPredictionRow])
async def rac_predictions(
    conn: DbConn, symbol: str, limit: int = Query(default=20, ge=1, le=200)
) -> list[dict[str, object]]:
    """Return recent persisted predictions linked to embeddings for `symbol`."""
    rows = await (
        await conn.execute(
            """
            SELECT id, predicted_label, confidence_score, k_neighbors, avg_neighbor_dist,
                   neighbor_label_dist, neighbor_ids, predicted_at
            FROM rac_predictions
            WHERE query_embedding_id IN (
                SELECT id FROM pattern_embeddings WHERE symbol = %(symbol)s
            )
            ORDER BY predicted_at DESC
            LIMIT %(limit)s
            """,
            {"symbol": symbol, "limit": limit},
        )
    ).fetchall()

    out: list[dict[str, object]] = []
    for r in rows:
        rr = cast(tuple[Any, ...], r)
        out.append(
            {
                "id": int(rr[0]),
                "predicted_label": int(rr[1]),
                "confidence_score": float(rr[2]) if rr[2] is not None else None,
                "k_neighbors": int(rr[3]),
                "avg_neighbor_dist": float(rr[4]) if rr[4] is not None else None,
                "neighbor_label_dist": cast(dict[str, int], rr[5]) if isinstance(rr[5], dict) else None,
                "neighbor_ids": [int(x) for x in (cast(list[Any], rr[6]) or [])] if rr[6] is not None else None,
                "predicted_at": rr[7],
            }
        )
    return out

