"""
GRU Late Fusion
===============
A standard recurrent late-fusion baseline for multimodal / multi-stream HAR.

Architecture
------------
For each modality m:
    x_m  [B, T_m, D_m]
    → Bi-GRU  (hidden_dim per direction)
    → Temporal self-attention pooling  → f_m  [B, feature_dim]

All modality features are concatenated and passed to a linear classifier.

This architecture is widely used as a strong recurrent baseline in HAR
comparisons, e.g. as a point of comparison in:
    Islam & Iqbal, "Cooperative Multitask Learning for Guided Multimodal
    Fusion", AAAI 2022.  https://doi.org/10.1609/aaai.v36i1.19988

The GRU cell is from:
    Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation", EMNLP 2014.
    https://arxiv.org/abs/1406.1078
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Temporal Self-Attention Pool ────────────────────────────────

class TemporalSelfAttention(nn.Module):
    """Additive self-attention over GRU hidden states → context vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, h: torch.Tensor):
        """
        h: [B, T, H]
        Returns: context [B, H], weights [B, T]
        """
        scores = self.attn(h).squeeze(-1)          # [B, T]
        alpha  = F.softmax(scores, dim=1)           # [B, T]
        ctx    = (alpha.unsqueeze(-1) * h).sum(1)  # [B, H]
        return ctx, alpha


# ── Per-Modality Bi-GRU Encoder ─────────────────────────────────

class ModalityGRUEncoder(nn.Module):
    """Bi-GRU → temporal self-attention → projection for one modality."""

    def __init__(self, input_dim: int, feature_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out = hidden_dim * 2  # bidirectional
        self.attn = TemporalSelfAttention(gru_out)
        self.proj = nn.Sequential(
            nn.LayerNorm(gru_out),
            nn.Linear(gru_out, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        """x: [B, T, D] → [B, feature_dim]"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h, _ = self.gru(x)          # [B, T, 2*hidden]
        c, _ = self.attn(h)         # [B, 2*hidden]
        return self.proj(c)         # [B, feature_dim]


# ── GRU Late Fusion Model ────────────────────────────────────────

class GRUFusion(nn.Module):
    """BiGRU late-fusion baseline for multimodal HAR.

    Each modality is processed by an independent Bi-GRU encoder; the
    resulting feature vectors are concatenated and classified.

    Args:
        num_modalities:  number of input modalities.
        feature_dim:     per-modality feature size after encoding.
        num_activities:  number of target classes.
        input_dim:       int or list[int] – per-modality input feature size.
        hidden_dim:      GRU hidden size (per direction).
        num_gru_layers:  number of stacked GRU layers.
        dropout:         dropout rate.
    """

    def __init__(
        self,
        num_modalities: int = 1,
        feature_dim: int = 128,
        num_activities: int = 27,
        input_dim=6,
        hidden_dim: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if isinstance(input_dim, (list, tuple)):
            input_dims = list(input_dim)
        else:
            input_dims = [input_dim] * num_modalities

        self.encoders = nn.ModuleList([
            ModalityGRUEncoder(
                input_dim=input_dims[i],
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_gru_layers,
                dropout=dropout,
            )
            for i in range(num_modalities)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * num_modalities),
            nn.Linear(feature_dim * num_modalities, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_activities),
        )

    def forward(self, x_list):
        """
        x_list: list of [B, T_m, D_m] per modality.
        Returns: logits [B, num_activities]
        """
        feats = [enc(x) for enc, x in zip(self.encoders, x_list)]
        fused = torch.cat(feats, dim=-1)  # [B, M * feature_dim]
        return self.classifier(fused)
