# models/cnn_gru_attn.py
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Lightweight CBAM (Channel + Spatial) ----
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b,c,h,w = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # concat of avg-channel and max-channel -> 2-ch
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)  # [B,2,H,W]
        w = torch.sigmoid(self.conv(s))  # [B,1,H,W]
        return x * w

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))

# ---- Attention Pooling on GRU outputs ----
class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention for sequence pooling.
       Given H=[h1..hL] (B,L,D), returns context vector (B,D) and attention weights (B,L).
    """
    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model, bias=True)
        self.v   = nn.Linear(d_model, 1, bias=False)
        self.tau = nn.Parameter(torch.tensor(float(temperature)), requires_grad=True)

    def forward(self, H):
        # H: [B, L, D]
        e = torch.tanh(self.lin(H))            # [B,L,D]
        scores = self.v(e).squeeze(-1)         # [B,L]
        a = torch.softmax(scores / self.tau.clamp_min(1e-3), dim=1)  # [B,L]
        ctx = torch.bmm(a.unsqueeze(1), H).squeeze(1)  # [B,D]
        return ctx, a

class ScaledDotAttention(nn.Module):
    """Scaled dot-product attention with a learnable global query for pooling."""
    def __init__(self, d_model: int, temperature: float = None):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # [1,1,D]
        self.scale = 1.0 / math.sqrt(d_model) if temperature is None else 1.0/temperature

    def forward(self, H):
        # H: [B,L,D]
        B,L,D = H.shape
        q = self.q.expand(B, -1, -1)            # [B,1,D]
        scores = torch.bmm(q, H.transpose(1,2)).squeeze(1) * self.scale  # [B,L]
        a = torch.softmax(scores, dim=1)
        ctx = torch.bmm(a.unsqueeze(1), H).squeeze(1)  # [B,D]
        return ctx, a

def _get_4x4_indices(order: str):
    order = (order or "row").lower()
    if order in ("z", "zorder", "z-order"):
        return [0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15]
    elif order in ("hilbert",):
        # a simple 4x4 hilbert-like order
        return [0,1,5,4, 2,3,7,6, 10,11,15,14, 8,9,13,12]
    else:  # row-major
        return list(range(16))

class CNNGRUAttn(nn.Module):
    """
    Hybrid Attention:
      - Optional CNN-level CBAM (channel + spatial) to enhance salient patches.
      - GRU over 4x4 tokens (sequence length 16).
      - Attention pooling (additive or scaled-dot) over GRU outputs.

    Args:
      num_classes: 2 for binary
      cnn_channels: (c1,c2) feature channels
      use_cbam: bool, enable CBAM after CNN
      cbam_reduction: int, channel reduction ratio
      gru_hidden: hidden dim
      gru_layers: number of layers
      bidirectional: GRU bidirection
      attn_type: "add" or "dot"
      dropout: classifier dropout
      use_batchnorm: conv bn toggle
      sequence_order: "row" | "z" | "hilbert"
    """
    def __init__(self,
                 num_classes: int = 2,
                 cnn_channels=(32, 64),
                 use_cbam: bool = True,
                 cbam_reduction: int = 8,
                 gru_hidden: int = 128,
                 gru_layers: int = 1,
                 bidirectional: bool = False,
                 attn_type: str = "add",
                 dropout: float = 0.5,
                 use_batchnorm: bool = True,
                 sequence_order: str = "row",
                 temperature: float = 1.0):
        super().__init__()
        c1, c2 = int(cnn_channels[0]), int(cnn_channels[1])
        BN = (lambda c: nn.BatchNorm2d(c)) if use_batchnorm else (lambda c: nn.Identity())
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            BN(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),                    # 8x8 -> 4x4
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            BN(c2), nn.ReLU(inplace=True),
        )
        self.use_cbam = bool(use_cbam)
        if self.use_cbam:
            self.cbam = CBAM(c2, reduction=cbam_reduction, spatial_kernel=7)

        self.sequence_order = sequence_order
        self.gru = nn.GRU(input_size=c2,
                          hidden_size=gru_hidden,
                          num_layers=gru_layers,
                          batch_first=True,
                          bidirectional=bidirectional,
                          dropout=0.0 if gru_layers==1 else 0.2)
        D = gru_hidden * (2 if bidirectional else 1)
        if attn_type.lower() in ("add", "additive"):
            self.pool = AdditiveAttention(D, temperature=temperature)
        elif attn_type.lower() in ("dot", "scaled_dot"):
            self.pool = ScaledDotAttention(D, temperature=None if temperature is None else (D**0.5/temperature))
        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}")

        self.classifier = nn.Sequential(
            nn.Linear(D, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _to_sequence(self, feat):
        # feat: [B,C,4,4] -> [B,16,C] with desired order
        B,C,H,W = feat.shape
        L = H*W
        idx = list(range(L))
        if (H,W) == (4,4):
            idx = _get_4x4_indices(self.sequence_order)
        seq = feat.view(B, C, L)[:, :, idx].permute(0, 2, 1)  # [B,L,C]
        return seq

    def forward(self, x, return_attn: bool = False):
        # x: [B,1,8,8]
        feat = self.cnn(x)  # [B,C,4,4]
        if self.use_cbam:
            feat = self.cbam(feat)
        seq = self._to_sequence(feat)        # [B,16,C]
        H, _ = self.gru(seq)                  # [B,16,D]
        ctx, a = self.pool(H)                 # ctx [B,D], a [B,16]
        logits = self.classifier(ctx)         # [B,num_classes]
        if return_attn:
            return logits, a
        return logits
