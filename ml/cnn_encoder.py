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
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class TransformerConfig:
    """Config for TemporalTransformerEncoder."""
    window_size: int = 30
    n_channels: int = 6          # input feature channels (OHLCV + close_ret)
    embedding_dim: int = 128     # output embedding dimension
    d_model: int = 128           # internal transformer width
    n_heads: int = 4             # attention heads (d_model must be divisible)
    n_layers: int = 4            # number of TransformerEncoder layers
    dim_feedforward: int = 256   # FFN hidden size (2x d_model)
    dropout: float = 0.1
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
    """Config for MultiScaleCNNEncoder (6 channels: OHLCV + close_ret)."""

    window_size: int = 30
    n_channels: int = 6
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
            self.pe_scale = nn.Parameter(torch.tensor(0.02))
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
        self.pool_query = nn.Parameter(torch.randn(1, 1, c3) * 0.02)
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


class MultiScaleCNNEncoder(nn.Module):
    """Multi-Scale CNN encoder for time-series pattern matching.
    Extracts features from the full window, half window, and 1/6 window
    to capture macro, meso, and micro trends respectively.
    """

    def __init__(self, cfg: LegacyEncoderConfig = LegacyEncoderConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = cfg.conv_channels
        k = cfg.kernel_size
        
        def make_branch() -> nn.Module:
            return nn.Sequential(
                nn.Conv1d(cfg.n_channels, c1, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.Conv1d(c1, c2, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.Conv1d(c2, c3, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(cfg.dropout),
            )
            
        self.branch_macro = make_branch()
        self.branch_meso = make_branch()
        self.branch_micro = make_branch()
        
        # 3 branches * c3 channels each -> embedding_dim
        self.proj = nn.Linear(c3 * 3, cfg.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B,T,C), got shape={tuple(x.shape)}")
        
        # Transpose for Conv1d: (B, C, T)
        x_t = x.transpose(1, 2)
        T = x_t.size(2)
        
        # Define multi-scale window lengths dynamically based on actual T
        k = self.cfg.kernel_size
        len_meso = max(k, T // 2)
        len_micro = max(k, T // 6)
        
        feat_macro = self.branch_macro(x_t)
        feat_meso = self.branch_meso(x_t[:, :, -len_meso:])
        feat_micro = self.branch_micro(x_t[:, :, -len_micro:])
        
        concat = torch.cat([feat_macro, feat_meso, feat_micro], dim=1)  # (B, c3*3)
        z = self.proj(concat)  # (B, embedding_dim)
        
        return nn.functional.normalize(z, p=2.0, dim=1)


class TemporalTransformerEncoder(nn.Module):
    """Pure Transformer encoder for financial time-series windows.

    Architecture:
      1. **Patch projection**: Linear(n_channels → d_model) applied per time-step
         (no convolution — every day is a "token").
      2. **CLS token**: A learnable [CLS] vector prepended to the sequence;
         its final hidden state is used as the sequence summary.
      3. **Sinusoidal PE**: Added to time-step tokens (not CLS) to encode order.
      4. **Transformer layers**: N × Pre-Norm TransformerEncoderLayer with
         multi-head self-attention + FFN.  Pre-norm (norm_first=True) is more
         stable than post-norm for shorter sequences.
      5. **Projection head**: CLS hidden state → MLP → embedding_dim, L2-norm.

    Compared to MultiScaleCNN:
      - Full receptive field from layer 1 (attention is global).
      - Learns which time-steps are important (dynamic, not fixed pooling).
      - Position-aware via PE: distinguishes "recent" vs "historical" days.
    """

    def __init__(self, cfg: TransformerConfig = TransformerConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        # Input projection: each day (n_channels) → d_model token
        self.input_proj = nn.Linear(cfg.n_channels, cfg.d_model)

        # Learnable CLS token (prepended at position 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Sinusoidal positional encoding for time-step tokens (not CLS)
        if cfg.use_positional_encoding:
            pe = _build_sinusoidal_pe(cfg.window_size, cfg.d_model)  # (T, d_model)
            self.register_buffer("pos_enc", pe, persistent=False)
        else:
            self.pos_enc = None  # type: ignore[assignment]

        # Stack of Transformer encoder layers (pre-norm for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable than post-LN on short sequences
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # Final layer-norm on CLS output
        self.norm = nn.LayerNorm(cfg.d_model)

        # MLP projection: d_model → embedding_dim
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch.

        Args:
            x: Tensor of shape (B, window_size, n_channels).

        Returns:
            Embeddings of shape (B, embedding_dim), L2-normalized.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, T, C), got {tuple(x.shape)}")
        b = x.size(0)

        # Project each day to d_model → (B, T, d_model)
        tokens = self.input_proj(x)

        # Add sinusoidal PE to time-step tokens
        if self.pos_enc is not None:
            tokens = tokens + self.pos_enc.unsqueeze(0)

        # Prepend CLS token → (B, T+1, d_model)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        # Transformer self-attention (all tokens attend to each other)
        out = self.transformer(seq)                # (B, T+1, d_model)

        # Extract CLS output (position 0) as sequence summary
        cls_out = self.norm(out[:, 0, :])          # (B, d_model)

        # Project to embedding space and L2-normalize
        emb = self.proj(cls_out)                   # (B, embedding_dim)
        return nn.functional.normalize(emb, p=2.0, dim=1)


class CNNAutoencoder(nn.Module):
    """Autoencoder wrapper for LegacyCNNEncoder to train via reconstruction."""

    def __init__(self, encoder: LegacyCNNEncoder | MultiScaleCNNEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        cfg = encoder.cfg
        
        # Decoder attempts to reverse the encoder.
        # Encoder uses AdaptiveAvgPool1d(1) which collapses the temporal dimension.
        # We need to project the 128D vector back to (C, T) sequence.
        # LegacyCNNEncoder channels: n_channels -> c1 -> c2 -> c3 -> pool
        c1, c2, c3 = cfg.conv_channels
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(cfg.embedding_dim, c3 * cfg.window_size),
            nn.ReLU(),
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(c3, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(c2, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(c1, cfg.n_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x shape: (B, T, C)
        Returns:
            z: (B, embedding_dim)
            x_recon: (B, T, C)
        """
        z = self.encoder(x)
        
        b = z.size(0)
        d = self.decoder_linear(z)
        d = d.view(b, self.encoder.cfg.conv_channels[2], self.encoder.cfg.window_size)
        
        d = self.decoder_conv(d)  # (B, C, T)
        return z, d.transpose(1, 2)  # (B, T, C)


@torch.inference_mode()
def encode_batch(
    model: CNNEncoder | LegacyCNNEncoder | MultiScaleCNNEncoder | TemporalTransformerEncoder,
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
