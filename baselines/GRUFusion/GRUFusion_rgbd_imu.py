"""
GRU Late Fusion – RGBD + IMU variant
=====================================
Wraps GRUFusion with a ResNet18 frontend for per-frame RGBD feature
extraction, mirroring the MuMuRGBDIMU interface.

Architecture:
  RGBD (B,T,4,H,W) → ResNet18 → (B,T,D_cnn) --+
                                                 +--> GRUFusion(2 modalities)
  IMU  (B,T_imu,6) ----------------------------+

Supports None inputs for missing-modality evaluation.
"""

import torch
import torch.nn as nn
from model.backbones.pretrained_cnn import PretrainedCNN
from .GRUFusion import GRUFusion


class GRUFusionRGBDIMU(nn.Module):
    """BiGRU late-fusion baseline for RGBD + IMU action recognition."""

    def __init__(
        self,
        num_activities: int = 27,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
        cnn_d_model: int = 128,
        freeze: str = "partial",
        in_channels: int = 4,
    ):
        super().__init__()
        self.cnn_d_model = cnn_d_model

        self.rgbd_cnn = PretrainedCNN(
            in_channels=in_channels,
            d_model=cnn_d_model,
            freeze=freeze,
        )

        self.gru_fusion = GRUFusion(
            num_modalities=2,
            feature_dim=feature_dim,
            num_activities=num_activities,
            input_dim=[cnn_d_model, 6],
            hidden_dim=hidden_dim,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
        )

    def forward(self, rgbd, imu):
        """
        rgbd: (B, T, 4, H, W) or None
        imu:  (B, T_imu, 6)   or None
        Returns: logits (B, num_activities)
        """
        if rgbd is not None:
            B, T, C, H, W = rgbd.shape
            rgbd_feat = self.rgbd_cnn(rgbd.reshape(B * T, C, H, W))
            rgbd_feat = rgbd_feat.reshape(B, T, -1)
        else:
            B = imu.shape[0]
            rgbd_feat = torch.zeros(
                B, 1, self.cnn_d_model, device=imu.device, dtype=imu.dtype
            )

        if imu is None:
            B = rgbd.shape[0]
            imu = torch.zeros(B, 1, 6, device=rgbd.device, dtype=rgbd.dtype)

        return self.gru_fusion([rgbd_feat, imu])
