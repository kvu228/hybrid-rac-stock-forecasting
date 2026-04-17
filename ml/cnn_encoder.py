"""1D-CNN encoder: [30 x 5] OHLCV window -> 128-d embedding.

The focus of this repo is DB engineering; this encoder exists to generate stable
embeddings for pgvector indexing and hybrid retrieval experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    window_size: int = 30
    n_channels: int = 5
    embedding_dim: int = 128
    conv_channels: tuple[int, int, int] = (32, 64, 128)
    kernel_size: int = 3
    dropout: float = 0.1


class CNNEncoder(nn.Module):
    """A compact 1D CNN encoder for time-series windows."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        c1, c2, c3 = cfg.conv_channels
        k = cfg.kernel_size

        # Input expected as (batch, time, channels) from numpy/pandas pipelines.
        # Conv1d expects (batch, channels, time).
        self.net = nn.Sequential(
            nn.Conv1d(cfg.n_channels, c1, kernel_size=k, padding=k // 2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k, padding=k // 2),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=k, padding=k // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, c3, 1)
            nn.Flatten(),  # (batch, c3)
            nn.Dropout(cfg.dropout),
            nn.Linear(c3, cfg.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch.

        Args:
            x: Tensor of shape (batch, window_size, n_channels)

        Returns:
            Embeddings of shape (batch, embedding_dim), L2-normalized.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (batch,time,channels), got shape={tuple(x.shape)}")
        x = x.transpose(1, 2)  # (batch, channels, time)
        z = self.net(x)
        return nn.functional.normalize(z, p=2.0, dim=1)


@torch.inference_mode()
def encode_batch(
    model: CNNEncoder,
    windows: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Encode windows into float32 embeddings.

    Args:
        model: Trained encoder.
        windows: numpy array of shape (N, 30, 5), typically z-score normalized.
        device: torch device.
        batch_size: inference batch size.
    """
    if windows.ndim != 3:
        raise ValueError(f"windows must have shape (N,30,5); got {windows.shape}")

    model = model.to(device)
    model.eval()

    out: list[np.ndarray] = []
    n = windows.shape[0]
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(device=device, dtype=torch.float32)
        emb = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
        out.append(emb)
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.cfg.embedding_dim), dtype=np.float32)

