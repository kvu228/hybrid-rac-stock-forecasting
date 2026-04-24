"""1D-CNN encoder with positional encoding + attention pooling.

Encodes a ``[window_size x n_channels]`` normalized OHLCV(+close_ret) window
into a fixed-size L2-normalized embedding used for pgvector KNN retrieval.

Design choices (vs the previous averaging variant):

* **Positional encoding** — sinusoidal PE is added to the conv features so the
  network can tell *when* in the window a pattern occurred. Average pooling
  alone discards all temporal location and tends to collapse up-trends and
  down-trends onto the same region of the embedding sphere.
* **Attention pooling** — a learnable query token attends over the time axis
  to produce a single vector; importance is learned, not uniform as in mean
  pooling. This preserves the end-of-window momentum which is what drives the
  T+5 label.

Output is L2-normalized so pgvector cosine distance is well-behaved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    window_size: int = 30
    n_channels: int = 6  # OHLCV + close_ret
    embedding_dim: int = 128
    conv_channels: tuple[int, int, int] = (32, 64, 128)
    kernel_size: int = 3
    dropout: float = 0.1
    attn_heads: int = 4
    use_positional_encoding: bool = True


def _build_sinusoidal_pe(length: int, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal positional encoding, shape ``(length, dim)``."""
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term)[:, : (dim // 2)]
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe


@dataclass(frozen=True)
class LegacyEncoderConfig:
    """Legacy config for the v1 encoder (mean pooling, 5 channels)."""

    window_size: int = 30
    n_channels: int = 5
    embedding_dim: int = 128
    conv_channels: tuple[int, int, int] = (32, 64, 128)
    kernel_size: int = 3
    dropout: float = 0.1


class LegacyCNNEncoder(nn.Module):
    """Legacy v1 encoder: conv stack + global average pooling."""

    def __init__(self, cfg: LegacyEncoderConfig = LegacyEncoderConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = cfg.conv_channels
        k = cfg.kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(cfg.n_channels, c1, kernel_size=k, padding=k // 2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k, padding=k // 2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=k, padding=k // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(c3, cfg.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B,T,C), got shape={tuple(x.shape)}")
        z = self.net(x.transpose(1, 2))
        return nn.functional.normalize(z, p=2.0, dim=1)


class CNNEncoder(nn.Module):
    """1D-CNN backbone + attention pooling → 128-D L2-normalized embedding."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        c1, c2, c3 = cfg.conv_channels
        k = cfg.kernel_size

        # Use dilated convolutions to increase receptive field over the 30-day window
        p1 = 1 * (k - 1) // 2
        p2 = 2 * (k - 1) // 2
        p3 = 4 * (k - 1) // 2

        # Conv1d expects (B, channels, time); we transpose in forward().
        self.backbone = nn.Sequential(
            nn.Conv1d(cfg.n_channels, c1, kernel_size=k, padding=p1, dilation=1),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=k, padding=p2, dilation=2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.Conv1d(c2, c3, kernel_size=k, padding=p3, dilation=4),
            nn.BatchNorm1d(c3),
            nn.GELU(),
        )

        if cfg.use_positional_encoding:
            pe = _build_sinusoidal_pe(cfg.window_size, c3)
            # Registered buffer so it moves with .to(device) but isn't trained.
            self.register_buffer("pos_encoding", pe, persistent=False)
            self.pe_scale = nn.Parameter(torch.tensor(0.5))
        else:
            self.pos_encoding = None  # type: ignore[assignment]
            self.pe_scale = None

        # Self-attention layer to let sequence elements interact before pooling
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=c3,
            nhead=cfg.attn_heads,
            dim_feedforward=c3 * 2,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )

        # Learnable query token that attends over time to pool a single vector.
        self.pool_query = nn.Parameter(torch.randn(1, 1, c3) * 0.1)
        self.attn = nn.MultiheadAttention(
            embed_dim=c3,
            num_heads=cfg.attn_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(c3)
        self.proj = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(c3, cfg.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch.

        Args:
            x: Tensor of shape (B, window_size, n_channels).

        Returns:
            Embeddings of shape (B, embedding_dim), L2-normalized.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, T, C), got shape={tuple(x.shape)}")
        b = x.size(0)

        feats = self.backbone(x.transpose(1, 2))  # (B, c3, T)
        feats = feats.transpose(1, 2)  # (B, T, c3) for attention

        if self.pos_encoding is not None:
            feats = feats + self.pos_encoding.unsqueeze(0) * self.pe_scale

        # Allow all days to interact and form complex patterns
        feats = self.self_attn(feats)

        query = self.pool_query.expand(b, -1, -1)  # (B, 1, c3)
        pooled, _ = self.attn(query, feats, feats, need_weights=False)  # (B, 1, c3)
        pooled = self.norm(pooled.squeeze(1))  # (B, c3)

        emb = self.proj(pooled)  # (B, embedding_dim)
        return nn.functional.normalize(emb, p=2.0, dim=1)


@torch.inference_mode()
def encode_batch(
    model: CNNEncoder | LegacyCNNEncoder,
    windows: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Encode windows into float32 L2-normalized embeddings.

    Args:
        model: Trained encoder.
        windows: numpy array of shape ``(N, T, C)`` where ``C == cfg.n_channels``.
        device: torch device.
        batch_size: inference batch size.
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must have shape (N, T, C); got {windows.shape}")
    if windows.shape[2] != model.cfg.n_channels:
        raise ValueError(
            f"windows has {windows.shape[2]} channels but encoder expects {model.cfg.n_channels}."
            " Regenerate windows with matching feature channels."
        )

    model = model.to(device)
    model.eval()

    out: list[np.ndarray] = []
    n = windows.shape[0]
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(device=device, dtype=torch.float32)
        emb = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
        out.append(emb)
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.cfg.embedding_dim), dtype=np.float32)
