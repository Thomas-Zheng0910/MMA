"""
Transformer Late Fusion
========================
A standard Transformer-encoder baseline for multimodal / multi-stream HAR.

Architecture
------------
For each modality m:
    x_m  [B, T_m, D_m]
    → Linear projection  → [B, T_m, d_model]
    → Sinusoidal positional encoding
    → N × TransformerEncoderLayer (MHSA + FFN)
    → Temporal mean pool           → f_m  [B, d_model]

All modality features are concatenated and classified.

References
----------
Transformer encoder:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    https://arxiv.org/abs/1706.03762

Application to sensor-based HAR:
    Li et al., "Two-Stream Convolution Augmented Transformer for Human
    Activity Recognition", AAAI 2021.
    https://ojs.aaai.org/index.php/AAAI/article/view/16177
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal Positional Encoding ──────────────────────────────

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


# ── Per-Modality Transformer Encoder ────────────────────────────

class ModalityTransformerEncoder(nn.Module):
    """Linear projection → positional encoding → TransformerEncoder → pool."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe   = SinusoidalPE(d_model, dropout=dropout)
        layer     = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for more stable training
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             norm=nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D_in] → [B, d_model]"""
        if x.dim() == 2:
            x = x.unsqueeze(1)          # [B, D] → [B, 1, D]
        x = self.proj(x)                # [B, T, d_model]
        x = self.pe(x)
        x = self.encoder(x)             # [B, T, d_model]
        return x.mean(dim=1)            # temporal mean pool → [B, d_model]


# ── Transformer Late Fusion Model ────────────────────────────────

class TransFusion(nn.Module):
    """Transformer encoder late-fusion baseline for multimodal HAR.

    Each modality is independently encoded by a Transformer encoder; the
    resulting feature vectors are concatenated and classified.

    Args:
        num_modalities:  number of input modalities.
        d_model:         Transformer model dimension.
        num_activities:  number of target classes.
        input_dim:       int or list[int] – per-modality input feature size.
        n_heads:         number of attention heads (must divide d_model).
        n_layers:        number of Transformer encoder layers.
        dim_feedforward: FFN inner dimension.
        dropout:         dropout rate.
    """

    def __init__(
        self,
        num_modalities: int = 1,
        d_model: int = 128,
        num_activities: int = 27,
        input_dim=6,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if isinstance(input_dim, (list, tuple)):
            input_dims = list(input_dim)
        else:
            input_dims = [input_dim] * num_modalities

        self.encoders = nn.ModuleList([
            ModalityTransformerEncoder(
                input_dim=input_dims[i],
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for i in range(num_modalities)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * num_modalities),
            nn.Linear(d_model * num_modalities, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_activities),
        )

    def forward(self, x_list):
        """
        x_list: list of [B, T_m, D_m] per modality.
        Returns: logits [B, num_activities]
        """
        feats = [enc(x) for enc, x in zip(self.encoders, x_list)]
        fused = torch.cat(feats, dim=-1)  # [B, M * d_model]
        return self.classifier(fused)
