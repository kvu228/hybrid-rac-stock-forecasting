"""RAC classification orchestrator.

Orchestrates retrieval + context enrichment + (optional) SVM prediction and
persists results to `rac_predictions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from ml.svm_classifier import SvmPrediction, load_svm, predict as svm_predict
from rac.context_enricher import FullRacContext, compute_full_rac_context


@dataclass(frozen=True)
class RacPredictionResult:
    predicted_label: int
    confidence_score: float | None
    k_neighbors: int
    avg_neighbor_dist: float | None
    neighbor_label_dist: dict[str, int] | None
    neighbor_ids: list[int]
    context: FullRacContext


def _features_from_context(embedding: list[float], ctx: FullRacContext) -> np.ndarray:
    """Assemble a numeric feature vector for the SVM.

    Feature layout (simple and stable):
    - 128 dims: original embedding
    - 1 dim: avg_cosine_dist
    - 3 dims: label distribution counts for labels 0/1/2 (missing -> 0)
    - 2 dims: dist_to_support, dist_to_resistance (missing -> nan -> 0)
    - 1 dim: sr_position_ratio
    - 1 dim: knn_confidence
    """
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.shape != (128,):
        raise ValueError("Expected embedding length 128")

    avg_dist = np.float32(ctx.avg_cosine_dist if ctx.avg_cosine_dist is not None else 0.0)
    ld = ctx.label_distribution or {}
    c0 = np.float32(ld.get("0", 0))
    c1 = np.float32(ld.get("1", 0))
    c2 = np.float32(ld.get("2", 0))
    d_sup = np.float32(ctx.dist_to_support if ctx.dist_to_support is not None else 0.0)
    d_res = np.float32(ctx.dist_to_resistance if ctx.dist_to_resistance is not None else 0.0)
    sr_ratio = np.float32(ctx.sr_position_ratio if ctx.sr_position_ratio is not None else 0.0)
    knn_conf = np.float32(ctx.knn_confidence if ctx.knn_confidence is not None else 0.0)
    return np.concatenate([emb, [avg_dist, c0, c1, c2, d_sup, d_res, sr_ratio, knn_conf]]).astype(np.float32)


async def predict_and_persist(
    conn: psycopg.AsyncConnection[tuple[object, ...]],
    *,
    query_embedding: list[float],
    symbol: str,
    current_price: float,
    k: int = 20,
    svm_model_path: Path | None = Path("ml/model_store/svm.pkl"),
    query_embedding_id: int | None = None,
) -> RacPredictionResult:
    ctx = await compute_full_rac_context(
        conn, query_embedding=query_embedding, symbol=symbol, current_price=current_price, k=k
    )

    # Default/fallback prediction: dominant label from KNN context.
    pred_label = int(ctx.dominant_label) if ctx.dominant_label is not None else 1
    conf: float | None = float(ctx.knn_confidence) if ctx.knn_confidence is not None else None

    # Optional SVM prediction if a model exists on disk.
    svm_pred: SvmPrediction | None = None
    if svm_model_path is not None and svm_model_path.exists():
        feats = _features_from_context(query_embedding, ctx)
        svm_pred = svm_predict(load_svm(svm_model_path), feats)
        pred_label = svm_pred.label
        conf = svm_pred.confidence

    # Persist to rac_predictions.
    #
    # If the caller doesn't have a `pattern_embeddings.id` for the query window,
    # we fall back to linking this prediction to the closest neighbor ID. This
    # keeps `GET /rac/predictions/{symbol}` usable without requiring the caller
    # to insert the query embedding first.
    if query_embedding_id is None and ctx.neighbor_ids:
        query_embedding_id = ctx.neighbor_ids[0]

    label_dist = ctx.label_distribution
    await conn.execute(
        """
        INSERT INTO rac_predictions (
            query_embedding_id,
            predicted_label,
            confidence_score,
            k_neighbors,
            avg_neighbor_dist,
            neighbor_label_dist,
            neighbor_ids
        )
        VALUES (
            %(query_embedding_id)s,
            %(predicted_label)s,
            %(confidence_score)s,
            %(k_neighbors)s,
            %(avg_neighbor_dist)s,
            %(neighbor_label_dist)s,
            %(neighbor_ids)s
        )
        """,
        {
            "query_embedding_id": query_embedding_id,
            "predicted_label": pred_label,
            "confidence_score": conf,
            "k_neighbors": k,
            "avg_neighbor_dist": ctx.avg_cosine_dist,
            "neighbor_label_dist": Jsonb(label_dist) if label_dist is not None else None,
            "neighbor_ids": ctx.neighbor_ids,
        },
    )

    return RacPredictionResult(
        predicted_label=pred_label,
        confidence_score=conf,
        k_neighbors=k,
        avg_neighbor_dist=ctx.avg_cosine_dist,
        neighbor_label_dist=label_dist,
        neighbor_ids=ctx.neighbor_ids,
        context=ctx,
    )

