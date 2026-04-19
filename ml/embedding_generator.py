"""Generate embeddings from OHLCV windows and insert into pgvector table.

This module bridges Phase 3 (windowing/labels) -> Phase 4 (CNN embeddings) ->
Postgres `pattern_embeddings`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import psycopg
import torch

from etl.feature_engineer import WindowRecord, generate_windows
from ml.cnn_encoder import CNNEncoder, EncoderConfig, encode_batch


@dataclass(frozen=True)
class InsertStats:
    windows: int
    inserted: int
    hnsw_index_used: bool | None = None


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    if not url:
        raise RuntimeError("DATABASE_URL is required (env var or --database-url)")
    return url


def _format_vector(vec: np.ndarray) -> str:
    # pgvector text format: [1,2,3]
    if vec.ndim != 1:
        raise ValueError("Expected 1D vector")
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"


def load_encoder(model_path: Path) -> CNNEncoder:
    payload = torch.load(model_path, map_location="cpu")
    cfg_dict = payload.get("config", {})
    cfg = EncoderConfig(**cfg_dict) if cfg_dict else EncoderConfig()
    model = CNNEncoder(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def fetch_ohlcv(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    symbol: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    clauses = ["symbol = %(symbol)s"]
    params: dict[str, object] = {"symbol": symbol}
    if start is not None:
        clauses.append("time >= %(start)s")
        params["start"] = start
    if end is not None:
        clauses.append("time <= %(end)s")
        params["end"] = end

    where = " AND ".join(clauses)
    query = f"""
        SELECT time, symbol, open, high, low, close, volume
        FROM stock_ohlcv
        WHERE {where}
        ORDER BY time
    """
    rows = conn.execute(query, params).fetchall()
    df = pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def verify_hnsw_index_used(conn: psycopg.Connection[tuple[object, ...]], *, k: int = 10) -> bool | None:
    """Return whether EXPLAIN for a sample KNN plan references the HNSW index (or None if no rows)."""
    with conn.cursor() as cur:
        cur.execute("SELECT embedding::text FROM pattern_embeddings ORDER BY id LIMIT 1")
        row = cur.fetchone()
        if row is None:
            return None
        qv = str(row[0])
        cur.execute(
            "EXPLAIN (FORMAT TEXT) SELECT id FROM pattern_embeddings "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (qv, k),
        )
        plan = "\n".join(str(r[0]) for r in cur.fetchall())
    low = plan.lower()
    return ("hnsw" in low) or ("idx_embedding_hnsw" in low)


def insert_embeddings(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    records: Sequence[WindowRecord],
    embeddings: np.ndarray,
) -> int:
    if len(records) != embeddings.shape[0]:
        raise ValueError("records and embeddings length mismatch")

    rows: list[tuple[object, ...]] = []
    for rec, emb in zip(records, embeddings, strict=True):
        rows.append(
            (
                rec.symbol,
                rec.window_start,
                rec.window_end,
                _format_vector(emb),
                int(rec.label),
                float(rec.future_return),
            )
        )

    inserted = 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO pattern_embeddings (symbol, window_start, window_end, embedding, label, future_return)
            VALUES (%s, %s, %s, %s::vector, %s, %s)
            """,
            rows,
        )
        inserted = cur.rowcount if cur.rowcount is not None else len(rows)
    conn.commit()
    return inserted


def generate_and_insert(
    *,
    database_url: str,
    symbol: str,
    model_path: Path,
    start: date | None = None,
    end: date | None = None,
    device: str = "cpu",
    batch_size: int = 512,
    truncate_symbol: bool = False,
) -> InsertStats:
    model = load_encoder(model_path)

    with psycopg.connect(database_url) as conn:
        if truncate_symbol:
            conn.execute("DELETE FROM pattern_embeddings WHERE symbol = %(symbol)s", {"symbol": symbol})
            conn.commit()

        df = fetch_ohlcv(conn, symbol=symbol, start=start, end=end)
        records = generate_windows(df)
        windows = np.stack([r.data for r in records], axis=0) if records else np.zeros((0, 30, 5), np.float32)
        embeddings = encode_batch(model, windows, device=device, batch_size=batch_size) if len(records) else np.zeros((0, 128), np.float32)
        inserted = insert_embeddings(conn, records=records, embeddings=embeddings) if len(records) else 0
        hnsw: bool | None = None
        if inserted > 0:
            hnsw = verify_hnsw_index_used(conn, k=10)
        return InsertStats(windows=len(records), inserted=inserted, hnsw_index_used=hnsw)


def main(argv: list[str] | None = None) -> int:
    def _parse_date(s: str) -> date:
        return date.fromisoformat(s)

    parser = argparse.ArgumentParser(description="Generate CNN embeddings and insert into pattern_embeddings.")
    parser.add_argument("--database-url", type=str, default="")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--model", type=Path, default=Path("ml/model_store/cnn_encoder.pt"))
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument("--device", type=str, default=os.environ.get("TORCH_DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--truncate-symbol",
        action="store_true",
        help="Delete existing embeddings for this symbol before inserting.",
    )
    args = parser.parse_args(argv)

    db_url = args.database_url or _database_url()
    stats = generate_and_insert(
        database_url=db_url,
        symbol=args.symbol,
        model_path=args.model,
        start=args.start,
        end=args.end,
        device=args.device,
        batch_size=args.batch_size,
        truncate_symbol=args.truncate_symbol,
    )
    print(stats)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

