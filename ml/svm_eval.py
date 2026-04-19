"""Train/evaluate SVM on CNN embeddings from DB OHLCV (Phase 4.4)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report

from etl.feature_engineer import forward_fill_trading_days, generate_windows, train_test_split_by_time
from etl.pipeline import _fetch_ohlcv_from_db, _load_symbols
from ml.cnn_encoder import encode_batch
from ml.embedding_generator import load_encoder
from ml.svm_classifier import predict, save_svm, train_svm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SVM on encoder embeddings; evaluate on chronological test split.")
    parser.add_argument("--database-url", default="", help="Defaults to DATABASE_URL.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", default=None)
    parser.add_argument("--encoder", type=Path, default=Path("ml/model_store/cnn_encoder.pt"))
    parser.add_argument("--svm-out", type=Path, default=None, help="Optional path to save trained SVM (joblib).")
    parser.add_argument("--device", type=str, default=os.environ.get("TORCH_DEVICE", "cpu"))
    args = parser.parse_args(argv)

    load_dotenv(override=False)
    database_url = (args.database_url or "").strip() or (os.getenv("DATABASE_URL") or "").strip()
    if not database_url:
        raise SystemExit("DATABASE_URL is required (or pass --database-url).")
    if not args.encoder.is_file():
        raise SystemExit(f"Encoder not found: {args.encoder}")

    symbols = _load_symbols(list(args.symbols) if args.symbols else None, args.symbols_file)
    df = _fetch_ohlcv_from_db(database_url, symbols, None, None)
    if df.empty:
        raise SystemExit("No OHLCV rows for given symbols.")
    df = forward_fill_trading_days(df)
    records = generate_windows(df)
    if not records:
        raise SystemExit("No windows generated.")
    train_recs, test_recs = train_test_split_by_time(records, train_ratio=0.8)
    if not train_recs or not test_recs:
        raise SystemExit("Train or test split is empty; need more history.")

    model = load_encoder(args.encoder)
    x_tr = np.stack([r.data for r in train_recs], axis=0)
    y_tr = np.asarray([r.label for r in train_recs], dtype=np.int64)
    x_te = np.stack([r.data for r in test_recs], axis=0)
    y_te = np.asarray([r.label for r in test_recs], dtype=np.int64)

    z_tr = encode_batch(model, x_tr, device=args.device, batch_size=512)
    z_te = encode_batch(model, x_te, device=args.device, batch_size=512)

    clf = train_svm(z_tr, y_tr)
    preds = [predict(clf, z_te[i]).label for i in range(z_te.shape[0])]
    print(classification_report(y_te.tolist(), preds, digits=4, zero_division=0))

    if args.svm_out is not None:
        save_svm(clf, args.svm_out)
        print(f"Saved SVM to {args.svm_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
