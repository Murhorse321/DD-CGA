# models/cnn_gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_4x4_indices(order: str):
    """
    返回 4x4 网格在行主序展平下的访问顺序索引列表（长度16）。
    - 'row'     : 行优先（0..15）
    - 'z'       : Z-order/Morton 顺序
    - 'hilbert' : 二阶 Hilbert 曲线顺序
    其他输入一律回退到 'row'。
    """
    order = (order or "row").lower()
    if order == "z":
        # Z-order (Morton) for 4x4
        return [0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15]
    elif order == "hilbert":
        # 4x4 的二阶 Hilbert（(x,y)->y*4+x）
        return [0,4,5,1, 2,6,7,3, 11,10,14,15, 13,9,8,12]
    else:
        return list(range(16))  # row-major

class CNNGRU(nn.Module):
    """
    输入 [B,1,8,8] → CNN 到 [B,C,4,4] → 重排为序列 [B,16,C] → GRU → 池化/聚合 → 分类
    """
    def __init__(
        self,
        num_classes: int = 2,
        cnn_channels=(32, 64),
        gru_hidden: int = 128,
        gru_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
        pooling: str = "mean",      # "mean" 或 "last"
        sequence_order: str = "row" # "row" / "z" / "hilbert"
    ):
        super().__init__()
        c1, c2 = cnn_channels
        def BN(c): return nn.BatchNorm2d(c) if use_batchnorm else nn.Identity()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),  # [B,c1,8,8]
            BN(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # [B,c1,4,4]
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), # [B,c2,4,4]
            BN(c2), nn.ReLU(inplace=True),
        )

        self.token_dim = c2
        self.seq_len   = 4 * 4
        self.sequence_order = sequence_order

        self.gru = nn.GRU(
            input_size=self.token_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if gru_layers == 1 else 0.2,
        )
        self.bidirectional = bidirectional
        self.pooling = pooling
        gru_out_dim = gru_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.cnn(x)                         # [B,C,4,4]
        B, C, H, W = feat.shape

        # —— 按 sequence_order 重排 4x4 token —— #
        if H == 4 and W == 4:
            idx = _get_4x4_indices(self.sequence_order)  # 长度16的索引
            seq = feat.view(B, C, H*W)[:, :, idx].permute(0, 2, 1)  # [B,16,C]
        else:
            # 兜底：非 4x4 时，回退为行优先
            seq = feat.view(B, C, H*W).permute(0, 2, 1)             # [B,H*W,C]

        out, _ = self.gru(seq)                   # [B,L,D]
        if self.pooling == "last":
            pooled = out[:, -1, :]               # [B,D]
        else:
            pooled = out.mean(dim=1)             # [B,D]

        logits = self.classifier(pooled)
        return logits
