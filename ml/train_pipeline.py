"""Training pipeline for the CNN encoder.

This is intentionally minimal. The thesis focus is the database layer; training
exists to produce embeddings with enough structure for retrieval experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Sequence

from etl.feature_engineer import WindowRecord, forward_fill_trading_days, generate_windows, train_test_split_by_time
from etl.pipeline import _fetch_ohlcv_from_db, _load_symbols
from ml.cnn_encoder import CNNEncoder, EncoderConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 42
    train_ratio: float = 0.8
    num_workers: int = 0
    loss: str = "ce"  # "ce" | "triplet"
    triplet_margin: float = 0.2


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        self.windows = windows.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)

    def __len__(self) -> int:  # pragma: no cover
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class EncoderWithHead(nn.Module):
    def __init__(self, encoder: CNNEncoder, n_classes: int = 3) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.cfg.embedding_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.head(z)
        return z, logits


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_ohlcv_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column in OHLCV TSV")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="raise")
    return df


def _make_synthetic_ohlcv(n: int = 120, symbol: str = "SYN") -> pd.DataFrame:
    """Generate synthetic OHLCV on business days for smoke-training."""
    dates = pd.bdate_range(start="2024-01-02", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    open_ = close - rng.uniform(0, 0.5, size=n)
    high = np.maximum(open_, close) + rng.uniform(0, 1.0, size=n)
    low = np.minimum(open_, close) - rng.uniform(0, 1.0, size=n)
    vol = rng.integers(1000, 5000, size=n, dtype=np.int64)
    return pd.DataFrame(
        {
            "time": dates,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def _records_to_arrays(records: Sequence[WindowRecord]) -> tuple[np.ndarray, np.ndarray]:
    windows = np.stack([r.data for r in records], axis=0)
    labels = np.asarray([r.label for r in records], dtype=np.int64)
    return windows, labels


def _train_encoder_ce(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    encoder: CNNEncoder,
    train_cfg: TrainConfig,
) -> tuple[CNNEncoder, float, float]:
    model = EncoderWithHead(encoder)
    model.to(train_cfg.device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    def eval_acc() -> float:
        if test_loader is None:
            return float("nan")
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for xb, yb in test_loader:
                xb = xb.to(train_cfg.device)
                yb = yb.to(train_cfg.device)
                _, logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        model.train()
        return (correct / total) if total else float("nan")

    last_loss = 0.0
    for _epoch in range(train_cfg.epochs):
        for xb, yb in train_loader:
            xb = xb.to(train_cfg.device)
            yb = yb.to(train_cfg.device)
            opt.zero_grad(set_to_none=True)
            _, logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    acc = float(eval_acc())
    return model.encoder, last_loss, acc


def _train_encoder_triplet(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    encoder: CNNEncoder,
    train_cfg: TrainConfig,
) -> tuple[CNNEncoder, float]:
    device = train_cfg.device
    encoder = encoder.to(device)
    loss_fn = nn.TripletMarginLoss(margin=float(train_cfg.triplet_margin), p=2)
    opt = torch.optim.AdamW(encoder.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    last_loss = 0.0
    for _epoch in range(train_cfg.epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            b = int(xb.size(0))
            if b < 2:
                continue
            anchors: list[torch.Tensor] = []
            positives: list[torch.Tensor] = []
            negatives: list[torch.Tensor] = []
            for i in range(b):
                same_idx = ((yb == yb[i]) & (torch.arange(b, device=yb.device) != i)).nonzero(as_tuple=False).view(-1)
                diff_idx = (yb != yb[i]).nonzero(as_tuple=False).view(-1)
                if same_idx.numel() == 0 or diff_idx.numel() == 0:
                    continue
                j = int(same_idx[torch.randint(same_idx.numel(), (1,), device=device)].item())
                k = int(diff_idx[torch.randint(diff_idx.numel(), (1,), device=device)].item())
                anchors.append(xb[i])
                positives.append(xb[j])
                negatives.append(xb[k])
            if len(anchors) < 2:
                continue
            a = torch.stack(anchors)
            p = torch.stack(positives)
            n = torch.stack(negatives)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(encoder(a), encoder(p), encoder(n))
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    return encoder, last_loss


def train_from_ohlcv(
    df: pd.DataFrame,
    *,
    encoder_cfg: EncoderConfig = EncoderConfig(),
    train_cfg: TrainConfig = TrainConfig(),
) -> tuple[CNNEncoder, dict[str, float | str]]:
    """Train encoder with CE+linear head (default) or triplet loss on embeddings."""
    _set_seed(train_cfg.seed)
    if train_cfg.loss not in {"ce", "triplet"}:
        raise ValueError("train_cfg.loss must be 'ce' or 'triplet'")

    records = generate_windows(df)
    train_recs, test_recs = train_test_split_by_time(records, train_ratio=train_cfg.train_ratio)
    if not train_recs:
        raise ValueError("No training windows generated; check OHLCV input range/quality.")

    x_train, y_train = _records_to_arrays(train_recs)
    x_test, y_test = _records_to_arrays(test_recs) if test_recs else (np.zeros((0, 30, 5)), np.zeros((0,)))

    train_ds = WindowDataset(x_train, y_train)
    test_ds = WindowDataset(x_test, y_test) if len(x_test) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    test_loader = (
        DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)
        if test_ds is not None
        else None
    )

    encoder = CNNEncoder(encoder_cfg)

    if train_cfg.loss == "triplet":
        encoder, last_loss = _train_encoder_triplet(train_loader, encoder, train_cfg)
        metrics: dict[str, float | str] = {
            "train_windows": float(len(train_ds)),
            "test_windows": float(len(test_ds)) if test_ds is not None else 0.0,
            "last_train_loss": float(last_loss),
            "test_accuracy": float("nan"),
            "loss_mode": "triplet",
        }
        return encoder, metrics

    encoder, last_loss, acc = _train_encoder_ce(train_loader, test_loader, encoder, train_cfg)
    metrics = {
        "train_windows": float(len(train_ds)),
        "test_windows": float(len(test_ds)) if test_ds is not None else 0.0,
        "last_train_loss": float(last_loss),
        "test_accuracy": acc,
        "loss_mode": "ce",
    }
    return encoder, metrics


def save_encoder(encoder: CNNEncoder, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": encoder.cfg.__dict__,
            "state_dict": encoder.state_dict(),
        },
        out_path,
    )


def main(argv: list[str] | None = None) -> int:
    def _parse_date(s: str) -> date:
        return date.fromisoformat(s)

    parser = argparse.ArgumentParser(description="Train CNN encoder for pgvector embeddings.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic OHLCV instead of reading from --ohlcv-tsv (for smoke runs).",
    )
    src.add_argument(
        "--from-db",
        action="store_true",
        help="Load OHLCV from Postgres table stock_ohlcv (uses DATABASE_URL or --database-url).",
    )
    parser.add_argument("--ohlcv-tsv", type=Path, default=Path("tests/fixtures/ohlcv_small.tsv"))
    parser.add_argument("--symbols", nargs="*", default=None, help="With --from-db: ticker symbols (space-separated).")
    parser.add_argument("--symbols-file", default=None, help="With --from-db: text file, one symbol per line (# comments ok).")
    parser.add_argument("--database-url", default="", help="With --from-db: SQLAlchemy URL (defaults to DATABASE_URL).")
    parser.add_argument("--start", type=_parse_date, default=None, help="With --from-db: inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=_parse_date, default=None, help="With --from-db: inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--out", type=Path, default=Path("ml/model_store/cnn_encoder.pt"))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--device", type=str, default=os.environ.get("TORCH_DEVICE", "cpu"))
    parser.add_argument("--loss", choices=("ce", "triplet"), default="ce", help="Training objective for the encoder.")
    parser.add_argument("--triplet-margin", type=float, default=0.2, help="Triplet margin (triplet loss only).")
    parser.add_argument("--metrics-out", type=Path, default=None, help="Optional path to write metrics JSON.")
    args = parser.parse_args(argv)

    load_dotenv(override=False)

    if str(args.device).lower().startswith("cuda"):
        if getattr(torch.version, "cuda", None) is None:
            raise SystemExit(
                "device=cuda but this PyTorch wheel is CPU-only (no CUDA build). "
                "Install PyTorch with CUDA from https://pytorch.org/get-started/locally/ "
                "(e.g. uv pip with the cu12* index URL) or use --device cpu."
            )
        if not torch.cuda.is_available():
            raise SystemExit(
                "device=cuda but torch.cuda.is_available() is False (GPU/driver?). Use --device cpu or fix CUDA."
            )

    if args.from_db:
        if not args.symbols and not args.symbols_file:
            raise SystemExit("With --from-db, provide --symbols and/or --symbols-file.")
        database_url = (args.database_url or "").strip() or (os.getenv("DATABASE_URL") or "").strip()
        if not database_url:
            raise SystemExit("DATABASE_URL is required for --from-db (or pass --database-url).")
        symbols = _load_symbols(list(args.symbols) if args.symbols else None, args.symbols_file)
        df = _fetch_ohlcv_from_db(database_url, symbols, args.start, args.end)
        if df.empty:
            raise SystemExit("No rows returned from stock_ohlcv for the given symbols/date range.")
        df = forward_fill_trading_days(df)
        if df.empty:
            raise SystemExit("OHLCV became empty after forward-fill; check input range.")
    elif args.synthetic:
        df = _make_synthetic_ohlcv()
    else:
        df = _load_ohlcv_tsv(args.ohlcv_tsv)
    encoder, metrics = train_from_ohlcv(
        df,
        train_cfg=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            loss=str(args.loss),
            triplet_margin=float(args.triplet_margin),
        ),
    )
    save_encoder(encoder, args.out)
    print(f"Saved encoder to {args.out}")
    print(metrics)
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

