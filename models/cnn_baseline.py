# models/cnn_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    """
    适配输入 [B, 1, 8, 8]
    两层卷积 + 两次 2x2 池化后，空间尺寸从 8x8 -> 4x4 -> 2x2
    最终特征维度 = 64 * 2 * 2 = 256
    """
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B,32,8,8]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # [B,32,4,4]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B,64,4,4]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # [B,64,2,2]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # [B,256]
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
