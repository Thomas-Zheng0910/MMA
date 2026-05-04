# Baselines

This document describes every baseline implemented under `baselines/`, covering its origin, architecture, and known published performance.

---

## 1. MuMu — Cooperative Multitask Learning for Guided Multimodal Fusion

### Origin

> S. M. Islam and A. Iqbal, "Cooperative Multitask Learning for Guided Multimodal Fusion,"
> *AAAI 2022*, pp. 6949–6957.
> <https://doi.org/10.1609/aaai.v36i1.19988>

### Files

| File | Scope |
|------|-------|
| `MuMu/MuMu.py` | IMU-only (multi-stream) |
| `MuMu/MuMu_rgbd_imu.py` | RGBD + IMU wrapper |

### Architecture

```
For each modality m:
  x_m [B, T_m, D_m]
  → UnimodalFeatureEncoder
      Bi-LSTM (2 layers)
      → Temporal Self-Attention (additive)    → f_m  [B, D]

Fusion:
  SMFusion (spectral/shared attention mask across modalities)
  → GMFusion (gated mixing with attention weights)
  → Auxiliary head on individual branches
  → Target head on fused representation

Outputs: (y_aux [B, C], y_target [B, C], alpha, attn_weights)
```

Key design: cooperative multi-task loss—auxiliary unimodal heads supervise each branch while the target head supervises the fused output, encouraging modality-specific feature learning.

### Published Performance on UTD-MHAD

| Modality | Acc (paper) | Acc (this repo) |
|----------|-------------|-----------------|
| RGBD + IMU | 94.62% | 61.16% |

> **Note:** The repo accuracy of 61.16% is significantly below the paper's 94.62%. This gap is expected — the paper uses skeleton + inertial modalities (not RGBD + inertial), a different train/test split, and more training epochs. The RGBD+IMU variant in this repo is a re-purposing rather than a direct replication.

---

## 2. GRUFusion — Bi-GRU Late Fusion

### Origin

The GRU cell is introduced in:

> K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, Y. Bengio,
> "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,"
> *EMNLP 2014*, pp. 1724–1734.
> <https://arxiv.org/abs/1406.1078>

Bidirectional GRUs with attention pooling are a canonical strong recurrent baseline in HAR, used as a comparison point throughout the MuMu paper above and many subsequent works (e.g., DeepConvLSTM — Ordóñez & Roggen, *Sensors* 2016).

### Files

| File | Scope |
|------|-------|
| `GRUFusion/GRUFusion.py` | IMU-only or arbitrary multi-stream |
| `GRUFusion/GRUFusion_rgbd_imu.py` | RGBD + IMU wrapper (ResNet18 CNN frontend) |

### Architecture

```
For each modality m:
  x_m [B, T_m, D_m]
  → Bi-GRU (num_gru_layers=2, hidden_dim=128)   hidden: [B, T_m, 2·H]
  → Additive self-attention pool                  → f_m  [B, 2·H]
  → Linear(2·H, feature_dim)                     → f_m  [B, feature_dim]

Fusion (late):
  concat(f_1, ..., f_M)  [B, M·feature_dim]
  → LayerNorm → Linear → ReLU → Dropout → Linear → logits [B, C]
```

For the RGBD+IMU variant, RGBD frames are first passed through a pretrained ResNet18 (4-channel, partial freeze) per frame before the GRU encoder.

**Default hyperparameters:**

| Param | Value |
|-------|-------|
| `hidden_dim` | 128 |
| `num_gru_layers` | 2 |
| `feature_dim` | 128 |
| `dropout` | 0.3 |

### Known Performance

There is no single canonical Bi-GRU late fusion paper targeting UTD-MHAD RGBD+IMU. Representative figures from the inertial-only HAR literature (IMU, different datasets):

| Setting | Dataset | Acc |
|---------|---------|-----|
| Bi-GRU (Ordóñez & Roggen 2016, LSTM variant) | OPPORTUNITY | ~92% |
| Bi-GRU + attention (Islam & Iqbal 2022, comparison) | UTD-MHAD (skel+imu) | ~87–90% |

Performance on this repo's RGBD+IMU UTD-MHAD setup is **not yet measured** — see `experiment_report.md` for methodology.

---

## 3. TransFusion — Transformer Encoder Late Fusion

### Origin

> A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin,
> "Attention Is All You Need," *NeurIPS 2017*.
> <https://arxiv.org/abs/1706.03762>

Application to sensor-based HAR:

> D. Li, J. Yao, Q. Nie, "Two-Stream Convolution Augmented Transformer for Human Activity Recognition,"
> *AAAI 2021*, pp. 286–293.
> <https://ojs.aaai.org/index.php/AAAI/article/view/16177>

(THAT — Two-stream HHAR Transformer — uses Transformer encoders on accelerometer+gyroscope, reaching 97.36% on WISDM. The architecture here follows the same per-modality encoder + pooling + concat pattern, adapted to UTD-MHAD.)

### Files

| File | Scope |
|------|-------|
| `TransFusion/TransFusion.py` | IMU-only or arbitrary multi-stream |
| `TransFusion/TransFusion_rgbd_imu.py` | RGBD + IMU with cross-attention fusion |

### Architecture

**`TransFusion.py` (late fusion):**

```
For each modality m:
  x_m [B, T_m, D_m]
  → Linear(D_m, d_model)
  → Sinusoidal positional encoding
  → N × TransformerEncoderLayer (Pre-LN, MHSA + FFN, batch_first)
  → Temporal mean pool                           → f_m  [B, d_model]

Fusion (late):
  concat(f_1, ..., f_M)  [B, M·d_model]
  → LayerNorm → Linear → GELU → Dropout → Linear → logits [B, C]
```

**`TransFusion_rgbd_imu.py` (cross-attention fusion):**

```
RGBD branch:
  (B,T,4,H,W) → ResNet18 per-frame [B,T,D_cnn]
  → Linear + Sin-PE → prepend [CLS] → TransformerEncoder
  → (seq [B,1+T,d], cls [B,d])

IMU branch:
  (B,T_i,6)
  → Linear + Sin-PE → prepend [CLS] → TransformerEncoder
  → (seq [B,1+T_i,d], cls [B,d])

Cross-attention (Nagrani et al., 2021):
  g_rgbd = CrossAttn(query=cls_rgbd, context=imu_tokens)   [B,d]
  g_imu  = CrossAttn(query=cls_imu,  context=rgbd_tokens)  [B,d]

Classifier:
  concat(g_rgbd, g_imu) [B,2d]
  → LayerNorm → Linear → GELU → Dropout → Linear → logits [B,C]
```

The cross-attention design is inspired by the Attention Bottleneck paper:

> A. Nagrani, S. Yang, A. Zisserman, A. Jha, C. Hartmann, C. Sun, D. Ross, C. Schmid,
> "Attention Bottlenecks for Multimodal Fusion," *NeurIPS 2021*.
> <https://arxiv.org/abs/2107.00135>

**Default hyperparameters:**

| Param | Value |
|-------|-------|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `dim_feedforward` | 256 |
| `dropout` | 0.1 |

### Known Performance

| Setting | Dataset | Acc |
|---------|---------|-----|
| THAT (Li et al., AAAI 2021) — accel+gyro | WISDM | 97.36% |
| THAT (Li et al., AAAI 2021) — accel+gyro | UCI-HAR | 96.53% |
| Attention Bottleneck (Nagrani et al., 2021) — audio+video | VGGSound | 67.0% |
| Attention Bottleneck (Nagrani et al., 2021) — audio+video | AudioSet | 47.4% |

No direct Transformer-based RGBD+IMU result on UTD-MHAD is reported in published literature. Performance on this repo's setup is **not yet measured**.

---

## Summary Table

| Model | File(s) | Fusion type | Backbone | Best published Acc (dataset) |
|-------|---------|-------------|----------|------------------------------|
| MuMu | `MuMu/` | SMFusion + GMFusion | Bi-LSTM | 94.62% (UTD-MHAD, skel+imu) |
| GRUFusion | `GRUFusion/` | Late concat | Bi-GRU + attn | ~92% (OPPORTUNITY, imu) |
| TransFusion (late) | `TransFusion/TransFusion.py` | Late concat | Transformer encoder | 97.36% (WISDM, accel+gyro) |
| TransFusion (cross-attn) | `TransFusion/TransFusion_rgbd_imu.py` | Cross-attention | ResNet18 + Transformer | 67.0% (VGGSound, audio+video) |

---

## References

1. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder," EMNLP 2014. <https://arxiv.org/abs/1406.1078>
2. Vaswani et al., "Attention Is All You Need," NeurIPS 2017. <https://arxiv.org/abs/1706.03762>
3. Islam & Iqbal, "Cooperative Multitask Learning for Guided Multimodal Fusion," AAAI 2022. <https://doi.org/10.1609/aaai.v36i1.19988>
4. Li et al., "Two-Stream Convolution Augmented Transformer for Human Activity Recognition," AAAI 2021. <https://ojs.aaai.org/index.php/AAAI/article/view/16177>
5. Nagrani et al., "Attention Bottlenecks for Multimodal Fusion," NeurIPS 2021. <https://arxiv.org/abs/2107.00135>
6. Chen et al., "UTD-MHAD: A Multimodal Dataset for Wearable and Depth Sensors Based Human Action Recognition," ICIP 2015.
7. Ordóñez & Roggen, "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition," *Sensors* 2016.
