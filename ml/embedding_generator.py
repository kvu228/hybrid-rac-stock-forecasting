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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psycopg
import torch
from dotenv import load_dotenv

from etl.feature_engineer import WindowRecord, generate_windows
from ml.cnn_encoder import CNNEncoder, EncoderConfig, LegacyCNNEncoder, LegacyEncoderConfig, encode_batch, TemporalTransformerEncoder, TransformerConfig


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


def load_encoder(model_path: Path) -> CNNEncoder | LegacyCNNEncoder | TemporalTransformerEncoder | torch.nn.Module:
    payload = torch.load(model_path, map_location="cpu")
    state_dict = payload.get("state_dict", {})
    cfg_dict = payload.get("config", {}) if isinstance(payload, dict) else {}
    encoder_type = payload.get("encoder_type", "") if isinstance(payload, dict) else ""

    keys = list(state_dict.keys()) if isinstance(state_dict, dict) else []

    # Detect Transformer by encoder_type field or by characteristic keys.
    is_transformer = encoder_type == "transformer" or any(k.startswith("input_proj.") for k in keys)
    is_multiscale = any(k.startswith("branch_macro.") for k in keys)
    is_legacy = any(k.startswith("net.") for k in keys)

    if is_transformer:
        cfg = TransformerConfig(**{k: v for k, v in cfg_dict.items() if k in TransformerConfig.__dataclass_fields__}) if cfg_dict else TransformerConfig()
        model = TemporalTransformerEncoder(cfg)
        model.load_state_dict(state_dict)
    elif is_multiscale:
        from ml.cnn_encoder import MultiScaleCNNEncoder
        cfg = LegacyEncoderConfig(**cfg_dict) if cfg_dict else LegacyEncoderConfig()
        model = MultiScaleCNNEncoder(cfg)
        model.load_state_dict(state_dict)
    elif is_legacy:
        cfg = LegacyEncoderConfig(**cfg_dict) if cfg_dict else LegacyEncoderConfig()
        model = LegacyCNNEncoder(cfg)
        model.load_state_dict(state_dict)
    else:
        cfg = EncoderConfig(**cfg_dict) if cfg_dict else EncoderConfig()
        model = CNNEncoder(cfg)
        model.load_state_dict(state_dict)

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
        windows = (
            np.stack([r.data for r in records], axis=0)
            if records
            else np.zeros((0, 30, int(model.cfg.n_channels)), np.float32)
        )
        embeddings = (
            encode_batch(model, windows, device=device, batch_size=batch_size)
            if len(records)
            else np.zeros((0, 128), np.float32)
        )
        inserted = insert_embeddings(conn, records=records, embeddings=embeddings) if len(records) else 0
        hnsw: bool | None = None
        if inserted > 0:
            hnsw = verify_hnsw_index_used(conn, k=10)
        return InsertStats(windows=len(records), inserted=inserted, hnsw_index_used=hnsw)


def _auto_device() -> str:
    env_dev = os.environ.get("TORCH_DEVICE")
    if env_dev and env_dev != "auto":
        return env_dev
    if getattr(torch.version, "cuda", None) is not None and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _worker_fn(
    database_url: str,
    symbol: str,
    model_path: Path,
    start: date | None,
    end: date | None,
    device: str,
    batch_size: int,
    truncate_symbol: bool,
) -> tuple[str, InsertStats | None, str | None]:
    try:
        stats = generate_and_insert(
            database_url=database_url,
            symbol=symbol,
            model_path=model_path,
            start=start,
            end=end,
            device=device,
            batch_size=batch_size,
            truncate_symbol=truncate_symbol,
        )
        return symbol, stats, None
    except Exception as e:
        import traceback
        return symbol, None, traceback.format_exc()


def main(argv: list[str] | None = None) -> int:
    def _parse_date(s: str) -> date:
        return date.fromisoformat(s)

    parser = argparse.ArgumentParser(description="Generate CNN embeddings and insert into pattern_embeddings.")
    parser.add_argument("--database-url", type=str, default="")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", type=str, default=None)
    parser.add_argument("--model", type=Path, default=Path("ml/model_store/cnn_encoder.pt"))
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--truncate-symbol",
        action="store_true",
        help="Delete existing embeddings for this symbol before inserting.",
    )
    args = parser.parse_args(argv)
    
    load_dotenv(override=False)
    if str(args.device).lower() == "auto":
        args.device = _auto_device()

    db_url = args.database_url or _database_url()

    from etl.pipeline import _load_symbols
    symbols = []
    if args.symbols or args.symbols_file:
        symbols = _load_symbols(list(args.symbols) if args.symbols else None, args.symbols_file)
    elif args.symbol:
        symbols = [args.symbol]
    else:
        raise SystemExit("Provide --symbol, --symbols, or --symbols-file")

    workers = max(1, min(args.workers, len(symbols)))
    if workers > 1:
        print(f"Generating embeddings for {len(symbols)} symbols using {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for sym in symbols:
                futures.append(
                    ex.submit(
                        _worker_fn, db_url, sym, args.model, args.start, args.end,
                        args.device, args.batch_size, args.truncate_symbol
                    )
                )
            for i, fut in enumerate(as_completed(futures)):
                sym, stats, err = fut.result()
                if err:
                    print(f"[{i + 1}/{len(symbols)}] ERROR {sym}: {err}")
                else:
                    print(f"[{i + 1}/{len(symbols)}] {sym}: {stats}")
    else:
        for i, sym in enumerate(symbols):
            sym, stats, err = _worker_fn(
                db_url, sym, args.model, args.start, args.end,
                args.device, args.batch_size, args.truncate_symbol
            )
            if err:
                print(f"[{i + 1}/{len(symbols)}] ERROR {sym}: {err}")
            else:
                print(f"[{i + 1}/{len(symbols)}] {sym}: {stats}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

