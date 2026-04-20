"""Training pipeline for the CNN encoder.

Trains the encoder so that its L2-normalized 128-D output cluster by label
(``SupCon`` main loss) while a small linear head also learns to classify
(``CE`` auxiliary). This produces embeddings that are directly useful for
pgvector cosine KNN retrieval — which is exactly what the RAC pipeline
consumes — and fixes the label-blind retrieval seen with CE-only training.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import psycopg
import torch
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from etl.feature_engineer import WindowRecord, forward_fill_trading_days, generate_windows, train_test_split_by_time
from etl.pipeline import _load_symbols
from ml.cnn_encoder import CNNEncoder, EncoderConfig


@dataclass(frozen=True)
class TrainConfig:
    # Defaults tuned for retrieval-first training (SupCon shapes cosine neighborhoods).
    # CE head is auxiliary; keep it light so embeddings are not dominated by linear head.
    epochs: int = 60
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 42
    train_ratio: float = 0.8
    num_workers: int = 0
    loss: str = "supcon"  # "supcon" | "ce" | "triplet"
    triplet_margin: float = 0.2
    supcon_temperature: float = 0.2
    supcon_ce_weight: float = 0.2  # auxiliary CE weight on top of SupCon
    use_balanced_sampler: bool = True
    use_pk_sampler: bool = True  # for SupCon: enforce P classes x K samples per batch
    pk_classes_per_batch: int = 3
    early_stop_patience: int = 15  # epochs without macro-F1 improvement


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


def _normalize_db_url(url: str) -> str:
    # psycopg.connect expects "postgresql://..." (not SQLAlchemy's "+driver" form)
    u = (url or "").strip()
    if u.startswith("postgresql+psycopg://"):
        u = "postgresql://" + u.removeprefix("postgresql+psycopg://")
    return u


def _fetch_ohlcv_from_db(
    database_url: str,
    symbols: list[str],
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    """Read OHLCV rows from Postgres via psycopg (no SQLAlchemy/psycopg2)."""
    db_url = _normalize_db_url(database_url)
    clauses = ["symbol = ANY(%(symbols)s)"]
    params: dict[str, object] = {"symbols": list(symbols)}
    if start is not None:
        clauses.append("time >= %(start)s")
        params["start"] = start
    if end is not None:
        clauses.append("time <= %(end)s")
        params["end"] = end
    where = " AND ".join(clauses)
    sql = (
        "SELECT time, symbol, open, high, low, close, volume "
        f"FROM stock_ohlcv WHERE {where} ORDER BY symbol, time"
    )
    with psycopg.connect(db_url) as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
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


def _balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Per-sample weights inversely proportional to class frequency."""
    counts = np.bincount(labels)
    counts = np.where(counts == 0, 1, counts)
    inv = 1.0 / counts
    weights = inv[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights.astype(np.float64)),
        num_samples=int(labels.shape[0]),
        replacement=True,
    )


class PKBatchSampler(Sampler[list[int]]):
    """P-K batch sampler: pick P classes, K samples per class each batch.

    This is critical for supervised contrastive learning: it guarantees each
    anchor has positives in the batch, otherwise SupCon degenerates.
    """

    def __init__(
        self,
        labels: np.ndarray,
        *,
        classes_per_batch: int,
        samples_per_class: int,
        seed: int = 42,
    ) -> None:
        if labels.ndim != 1:
            raise ValueError("labels must be 1D")
        self.labels = labels.astype(np.int64, copy=False)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)

        self._rng = np.random.default_rng(self.seed)
        self._classes = np.unique(self.labels).astype(np.int64)
        self._indices_by_class: dict[int, np.ndarray] = {
            int(c): np.flatnonzero(self.labels == c).astype(np.int64) for c in self._classes
        }
        if self.classes_per_batch <= 0 or self.samples_per_class <= 0:
            raise ValueError("classes_per_batch and samples_per_class must be > 0")
        if len(self._classes) < self.classes_per_batch:
            raise ValueError(
                f"Not enough classes for PK sampling: have {len(self._classes)} need {self.classes_per_batch}"
            )

        # Approximate number of batches per epoch: cover dataset once.
        self._batches_per_epoch = max(1, int(len(self.labels) / (self.classes_per_batch * self.samples_per_class)))

    def __iter__(self):
        for _ in range(self._batches_per_epoch):
            chosen = self._rng.choice(self._classes, size=self.classes_per_batch, replace=False)
            batch: list[int] = []
            for c in chosen:
                pool = self._indices_by_class[int(c)]
                # Sample with replacement if class is small.
                replace = pool.size < self.samples_per_class
                idxs = self._rng.choice(pool, size=self.samples_per_class, replace=replace)
                batch.extend(int(i) for i in idxs.tolist())
            yield batch

    def __len__(self) -> int:  # pragma: no cover
        return self._batches_per_epoch


def _supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., 2020) on L2-normalized features.

    Pulls together same-label samples and pushes apart different-label samples
    on the unit sphere — directly shapes the cosine-distance neighborhoods
    used by pgvector at query time.
    """
    device = features.device
    b = features.size(0)
    if b < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Cosine similarity matrix (features are already L2-normalized by the encoder).
    sim = torch.matmul(features, features.t()) / float(temperature)

    # For numerical stability subtract per-row max, then mask self-contrast.
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    logits = sim
    self_mask = torch.eye(b, device=device, dtype=torch.bool)

    labels_col = labels.view(-1, 1)
    positive_mask = labels_col == labels_col.t()
    positive_mask = positive_mask & ~self_mask

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = positive_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / pos_count

    # Skip anchors that have no positive (class singletons in batch).
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    return -mean_log_prob_pos[valid].mean()


def _eval_macro_f1(
    model: EncoderWithHead,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    device: str,
) -> tuple[float, float, list[int], list[int]]:
    if loader is None:
        return float("nan"), float("nan"), [], []
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            _, logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.cpu().numpy().tolist())
    model.train()
    if not y_true:
        return float("nan"), float("nan"), [], []
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return f1m, acc, y_true, y_pred


def _train_encoder(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
    encoder: CNNEncoder,
    train_cfg: TrainConfig,
    *,
    class_weights: torch.Tensor | None = None,
) -> tuple[CNNEncoder, dict[str, float | str]]:
    device = train_cfg.device
    model = EncoderWithHead(encoder).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, train_cfg.epochs))

    use_supcon = train_cfg.loss == "supcon"
    use_ce_only = train_cfg.loss == "ce"
    if train_cfg.loss not in {"supcon", "ce"}:
        raise ValueError(f"Unsupported loss for _train_encoder: {train_cfg.loss}")

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    last_loss = 0.0
    last_supcon = 0.0
    last_ce = 0.0
    f1_history: list[float] = []

    for epoch in range(train_cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            z, logits = model(xb)
            loss_ce = ce_loss(logits, yb)
            if use_supcon:
                loss_supcon = _supcon_loss(z, yb, temperature=train_cfg.supcon_temperature)
                loss = loss_supcon + train_cfg.supcon_ce_weight * loss_ce
                last_supcon = float(loss_supcon.item())
            else:  # ce only
                loss = loss_ce
                last_supcon = 0.0
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            last_ce = float(loss_ce.item())
        scheduler.step()

        f1m, acc, _, _ = _eval_macro_f1(model, test_loader, device)
        f1_history.append(f1m)
        print(
            f"  epoch {epoch + 1:>3}/{train_cfg.epochs}  "
            f"loss={last_loss:.4f}  supcon={last_supcon:.4f}  ce={last_ce:.4f}  "
            f"val_macro_f1={f1m:.4f}  val_acc={acc:.4f}"
        )

        if f1m > best_f1 + 1e-4:
            best_f1 = f1m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if (
                train_cfg.early_stop_patience > 0
                and epochs_no_improve >= train_cfg.early_stop_patience
                and test_loader is not None
            ):
                print(f"  early stop at epoch {epoch + 1} (no macro-F1 improvement for {epochs_no_improve} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    f1m, acc, y_true, y_pred = _eval_macro_f1(model, test_loader, device)
    per_class_report = (
        classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=False)
        if y_true
        else ""
    )
    if per_class_report:
        print("  per-class classification report (best model):")
        for line in per_class_report.splitlines():
            print("    " + line)

    metrics: dict[str, float | str] = {
        "last_train_loss": float(last_loss),
        "last_supcon_loss": float(last_supcon),
        "last_ce_loss": float(last_ce),
        "best_val_macro_f1": float(best_f1 if best_f1 >= 0 else float("nan")),
        "final_val_macro_f1": float(f1m),
        "final_val_accuracy": float(acc),
        "epochs_run": float(len(f1_history)),
        "loss_mode": train_cfg.loss,
    }
    return model.encoder, metrics


def _train_encoder_triplet(
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    encoder: CNNEncoder,
    train_cfg: TrainConfig,
) -> tuple[CNNEncoder, float]:
    """Legacy triplet training. Kept for reproducibility; SupCon is recommended."""
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
    """Train encoder with SupCon (default), CE-only, or triplet loss."""
    _set_seed(train_cfg.seed)
    if train_cfg.loss not in {"supcon", "ce", "triplet"}:
        raise ValueError("train_cfg.loss must be 'supcon', 'ce', or 'triplet'")

    records = generate_windows(df)
    train_recs, test_recs = train_test_split_by_time(records, train_ratio=train_cfg.train_ratio)
    if not train_recs:
        raise ValueError("No training windows generated; check OHLCV input range/quality.")

    x_train, y_train = _records_to_arrays(train_recs)
    if test_recs:
        x_test, y_test = _records_to_arrays(test_recs)
    else:
        # Fallback to channel count inferred from train to avoid hard-coded 5.
        empty_channels = x_train.shape[2]
        x_test = np.zeros((0, encoder_cfg.window_size, empty_channels), dtype=np.float32)
        y_test = np.zeros((0,), dtype=np.int64)

    # Verify channels match the encoder config; auto-adjust encoder if needed.
    got_channels = int(x_train.shape[2])
    if got_channels != encoder_cfg.n_channels:
        encoder_cfg = EncoderConfig(**{**encoder_cfg.__dict__, "n_channels": got_channels})

    train_hist = Counter(int(y) for y in y_train)
    test_hist = Counter(int(y) for y in y_test)
    print(f"  train windows={len(y_train):,} label dist={dict(sorted(train_hist.items()))}")
    print(f"  test  windows={len(y_test):,} label dist={dict(sorted(test_hist.items()))}")

    train_ds = WindowDataset(x_train, y_train)
    test_ds = WindowDataset(x_test, y_test) if len(x_test) else None

    # For SupCon we strongly prefer PK sampling so each batch has positives.
    if train_cfg.loss == "supcon" and train_cfg.use_pk_sampler:
        n_classes = int(np.unique(y_train).size)
        p = min(int(train_cfg.pk_classes_per_batch), n_classes)
        if p < 2:
            print("  PK sampler disabled: need at least 2 classes in training split.")
            p = 0
        if p == 0:
            train_loader = DataLoader(
                train_ds,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=train_cfg.num_workers,
            )
        else:
        # Keep P=3 for {Down,Neutral,Up}; drop remainder if batch_size not divisible.
            k_per = max(2, int(train_cfg.batch_size // max(1, p)))
            effective_bs = p * k_per
            if effective_bs != int(train_cfg.batch_size):
                print(
                    f"  PK sampler: using batch_size={effective_bs} (= {p} classes x {k_per} samples) "
                    f"instead of {train_cfg.batch_size}"
                )
            batch_sampler = PKBatchSampler(
                y_train,
                classes_per_batch=p,
                samples_per_class=k_per,
                seed=train_cfg.seed,
            )
            train_loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=train_cfg.num_workers,
            )
    elif train_cfg.use_balanced_sampler and train_cfg.loss in {"supcon", "ce"}:
        sampler = _balanced_sampler(y_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg.batch_size,
            sampler=sampler,
            num_workers=train_cfg.num_workers,
        )
    else:
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
            "best_val_macro_f1": float("nan"),
            "loss_mode": "triplet",
        }
        return encoder, metrics

    # Class weights for CE (both as main loss and as SupCon auxiliary).
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = np.zeros(3, dtype=np.float32)
    for c, n in zip(classes, counts, strict=True):
        class_weights[int(c)] = float(len(y_train)) / (3.0 * float(n))
    cw_tensor = torch.from_numpy(class_weights).to(torch.float32)

    encoder, train_metrics = _train_encoder(train_loader, test_loader, encoder, train_cfg, class_weights=cw_tensor)
    metrics = {
        "train_windows": float(len(train_ds)),
        "test_windows": float(len(test_ds)) if test_ds is not None else 0.0,
        **train_metrics,
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
    parser.add_argument("--database-url", default="", help="With --from-db: Postgres URL (defaults to DATABASE_URL).")
    parser.add_argument("--start", type=_parse_date, default=None, help="With --from-db: inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=_parse_date, default=None, help="With --from-db: inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--out", type=Path, default=Path("ml/model_store/cnn_encoder.pt"))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--device", type=str, default=os.environ.get("TORCH_DEVICE", "cpu"))
    parser.add_argument(
        "--loss",
        choices=("supcon", "ce", "triplet"),
        default=TrainConfig.loss,
        help="Training objective for the encoder.",
    )
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--supcon-ce-weight", type=float, default=0.3)
    parser.add_argument(
        "--no-pk-sampler",
        action="store_true",
        help="Disable PK-batch sampling for SupCon (falls back to balanced sampler / shuffle).",
    )
    parser.add_argument("--pk-classes-per-batch", type=int, default=TrainConfig.pk_classes_per_batch)
    parser.add_argument(
        "--no-balanced-sampler",
        action="store_true",
        help="Disable WeightedRandomSampler (falls back to shuffle).",
    )
    parser.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
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
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=str(args.device),
            loss=str(args.loss),
            triplet_margin=float(args.triplet_margin),
            supcon_temperature=float(args.supcon_temperature),
            supcon_ce_weight=float(args.supcon_ce_weight),
            use_pk_sampler=not args.no_pk_sampler,
            pk_classes_per_batch=int(args.pk_classes_per_batch),
            use_balanced_sampler=not args.no_balanced_sampler,
            early_stop_patience=int(args.early_stop_patience),
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
