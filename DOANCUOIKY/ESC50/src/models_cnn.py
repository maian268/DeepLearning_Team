"""src/models_cnn.py - CNN theo Phần C5 (Tấn).

Kiến trúc:
    Input [B, 1, n_mels, 216]
    4 khối: [Conv3x3(pad=1, bias=False) -> BN -> ReLU] x2 -> MaxPool2x2
    kênh: 1->32, 32->64, 64->128, 128->256
    Global Average Pooling -> Dropout(p) -> Linear(256 -> n_classes)

Số tham số ~1.19 triệu, không đổi theo n_mels.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Hai lớp Conv3x3-BN-ReLU rồi MaxPool 2x2.

    Thanh dùng lại lớp này cho 3 khối đầu của CRNN (kênh 32, 64, 128).
    """

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, n_mels: int = 128, n_classes: int = 50, dropout: float = 0.3):
        super().__init__()
        self.n_mels = n_mels
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # [B, 256, n_mels/16, 13]
        x = self.gap(x).flatten(1)    # [B, 256]
        x = self.drop(x)
        return self.fc(x)             # [B, n_classes]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


