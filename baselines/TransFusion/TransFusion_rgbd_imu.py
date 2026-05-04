"""
TransFusion – Multimodal Cross-Attention Transformer for RGBD + IMU
=====================================================================
A cross-attention Transformer baseline for multimodal HAR.

Architecture
------------
RGBD branch:
    (B,T,4,H,W) → ResNet18 per-frame → (B,T,D_cnn)
    → Linear project → sinusoidal PE → [CLS] + seq → TransformerEncoder
    → CLS token  → f_rgbd  [B, d_model]

IMU branch:
    (B,T_imu,6)
    → Linear project → sinusoidal PE → [CLS] + seq → TransformerEncoder
    → CLS token  → f_imu  [B, d_model]

Cross-attention fusion:
    RGBD CLS queries IMU sequence  → g_rgbd  [B, d_model]
    IMU  CLS queries RGBD sequence → g_imu   [B, d_model]
    concat(g_rgbd, g_imu)          → Classifier → logits

Supports None inputs for missing-modality evaluation (missing branch is
replaced with zero tokens so the cross-attention still runs).

References
----------
Transformer encoder & cross-attention:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    https://arxiv.org/abs/1706.03762

Multimodal cross-attention fusion pattern:
    Nagrani et al., "Attention Bottlenecks for Multimodal Fusion",
    NeurIPS 2021.  https://arxiv.org/abs/2107.00135
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.pretrained_cnn import PretrainedCNN


# ── Sinusoidal Positional Encoding ──────────────────────────────

class SinusoidalPE(nn.Module):
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
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


# ── Single-Modality Transformer Branch ──────────────────────────

class ModalityBranch(nn.Module):
    """Project → PE → [CLS] + seq → TransformerEncoder.

    Returns the full sequence (including CLS at position 0) for
    cross-attention, and the CLS feature for unimodal classification.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pe   = SinusoidalPE(d_model, dropout=dropout)
        self.cls  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D_in]
        Returns:
            seq: [B, 1+T, d_model]   (CLS at position 0)
            cls: [B, d_model]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x   = self.proj(x)                                     # [B, T, d_model]
        x   = self.pe(x)
        cls = self.cls.expand(x.size(0), -1, -1)               # [B, 1, d_model]
        seq = torch.cat([cls, x], dim=1)                       # [B, 1+T, d_model]
        seq = self.encoder(seq)
        return seq, seq[:, 0]                                  # seq, cls_feat


# ── Cross-Attention Module ───────────────────────────────────────

class CrossAttention(nn.Module):
    """Single CLS query attending over a sequence (key/value)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query:   [B, 1, d_model]  (CLS from one branch)
        context: [B, T, d_model]  (full sequence from other branch)
        Returns: [B, d_model]
        """
        out, _ = self.attn(query, context, context)
        out = self.norm(out.squeeze(1) + query.squeeze(1))  # residual
        return out


# ── TransFusion RGBD+IMU Model ──────────────────────────────────

class TransFusionRGBDIMU(nn.Module):
    """Cross-attention Transformer baseline for RGBD + IMU action recognition.

    Args:
        num_activities:  number of target classes.
        d_model:         Transformer model dimension.
        n_heads:         number of attention heads (must divide d_model).
        n_layers:        Transformer encoder depth per branch.
        dim_feedforward: FFN inner dimension.
        dropout:         dropout rate.
        cnn_d_model:     output dim of the ResNet18 CNN frontend.
        freeze:          ResNet18 freeze mode ("all" | "partial" | "none").
        in_channels:     input channels for the CNN (4 for RGBD).
    """

    def __init__(
        self,
        num_activities: int = 27,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        cnn_d_model: int = 128,
        freeze: str = "partial",
        in_channels: int = 4,
    ):
        super().__init__()
        self.cnn_d_model = cnn_d_model
        self.d_model = d_model

        # RGBD per-frame CNN frontend
        self.rgbd_cnn = PretrainedCNN(
            in_channels=in_channels,
            d_model=cnn_d_model,
            freeze=freeze,
        )

        # Per-modality Transformer branches
        self.rgbd_branch = ModalityBranch(
            input_dim=cnn_d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.imu_branch = ModalityBranch(
            input_dim=6,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Cross-attention: each branch's CLS queries the other's sequence
        self.cross_rgbd = CrossAttention(d_model, n_heads, dropout)
        self.cross_imu  = CrossAttention(d_model, n_heads, dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_activities),
        )

    def forward(self, rgbd, imu):
        """
        rgbd: (B, T, 4, H, W) or None
        imu:  (B, T_imu, 6)   or None
        Returns: logits (B, num_activities)
        """
        # ── RGBD encoding ──────────────────────────────────────
        if rgbd is not None:
            B, T, C, H, W = rgbd.shape
            rgbd_feat = self.rgbd_cnn(rgbd.reshape(B * T, C, H, W))
            rgbd_feat = rgbd_feat.reshape(B, T, -1)
        else:
            B = imu.shape[0]
            rgbd_feat = torch.zeros(
                B, 1, self.cnn_d_model, device=imu.device, dtype=imu.dtype
            )

        # ── IMU encoding ───────────────────────────────────────
        if imu is None:
            B = rgbd.shape[0]
            imu = torch.zeros(B, 1, 6, device=rgbd.device, dtype=rgbd.dtype)

        # Branch forward passes
        rgbd_seq, rgbd_cls = self.rgbd_branch(rgbd_feat)  # [B,1+T_v,D], [B,D]
        imu_seq,  imu_cls  = self.imu_branch(imu)         # [B,1+T_i,D], [B,D]

        # Cross-attention: each CLS attends to the other modality's sequence
        # Use full sequence (excluding CLS) as context to avoid self-loop
        g_rgbd = self.cross_rgbd(
            rgbd_cls.unsqueeze(1),  # query: [B, 1, D]
            imu_seq[:, 1:],         # context: IMU tokens [B, T_i, D]
        )
        g_imu = self.cross_imu(
            imu_cls.unsqueeze(1),   # query: [B, 1, D]
            rgbd_seq[:, 1:],        # context: RGBD tokens [B, T_v, D]
        )

        fused = torch.cat([g_rgbd, g_imu], dim=-1)  # [B, 2*d_model]
        return self.classifier(fused)
