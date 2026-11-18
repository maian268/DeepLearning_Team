"""
Mô hình CRNN (Convolutional Recurrent Neural Network) cho phân loại âm thanh ESC-50.

Theo Phần C6 của tài liệu kế hoạch:
  - 3 khối đầu giống hệt CNN (kênh 32, 64, 128), dùng lại ConvBlock từ models_cnn
  - thay khối 4 và GAP bằng nhánh hồi quy (Linear -> BiGRU -> trung bình theo thời gian)
Số tham số: ~1,06 triệu với 128 bins (1.056.594), ~0,93 triệu với 64 bins (925.522).
"""

import torch
import torch.nn as nn

# Dùng chung ConvBlock và count_params với CNN để hai mô hình thống nhất.
# Thử import theo package (src.models_crnn) trước, rồi mới import cùng thư mục.
try:
    from .models_cnn import ConvBlock, count_params  # noqa: F401
except ImportError:
    from models_cnn import ConvBlock, count_params  # noqa: F401


class CRNN(nn.Module):
    """
    Kiến trúc CRNN:
      1. CNN Backbone: 3 khối ConvBlock với kênh 32 -> 64 -> 128
      2. Projection: Linear đưa đặc trưng không gian về 128 chiều
      3. BiGRU: 2 layer GRU 2 chiều (hidden_size=128), ra 256 chiều
      4. Temporal Pooling: Lấy trung bình theo trục thời gian (mean)
      5. Classifier: Dropout(0.3) -> Linear(256, 50)
    """
    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 50,
        dropout: float = 0.3,   # cùng tên tham số với CNN (và khóa "dropout" trong CFG)
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_classes = n_classes

        # 1. 3 khối tích chập CNN trích xuất không gian - tần số
        self.cnn_block1 = ConvBlock(1, 32)
        self.cnn_block2 = ConvBlock(32, 64)
        self.cnn_block3 = ConvBlock(64, 128)

        # Trục tần số n_mels bị giảm 8 lần qua 3 tầng MaxPool(2)
        reduced_freq_bins = n_mels // 8
        in_features_proj = 128 * reduced_freq_bins

        # 2. Linear projection đưa về 128 chiều trước khi vào RNN
        self.proj = nn.Sequential(
            nn.Linear(in_features_proj, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 3. BiGRU học phụ thuộc ngữ cảnh chuỗi theo trục thời gian
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # 4. Tầng phân loại đầu ra
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),  # 256 = 128 * 2 do Bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 1, n_mels, 216]
        Output: [B, n_classes]  (logit, chưa softmax)
        """
        # --- 1. CNN Feature Extraction ---
        x = self.cnn_block1(x)  # -> [B, 32, n_mels/2, 108]
        x = self.cnn_block2(x)  # -> [B, 64, n_mels/4, 54]
        x = self.cnn_block3(x)  # -> [B, 128, n_mels/8, 27]

        # --- 2. Permute và Reshape sang chuỗi thời gian ---
        # Đưa trục thời gian (27 frames) lên thứ hai: [B, 27, 128, n_mels/8]
        x = x.permute(0, 3, 1, 2)
        b, t, c, f = x.shape
        x = x.reshape(b, t, c * f)  # -> [B, 27, 128 * (n_mels/8)]

        # --- 3. Linear Projection ---
        x = self.proj(x)  # -> [B, 27, 128]

        # --- 4. BiGRU Layer ---
        gru_out, _ = self.gru(x)  # -> [B, 27, 256]

        # --- 5. Temporal Global Average Pooling ---
        pooled = gru_out.mean(dim=1)  # -> [B, 256]

        # --- 6. Classification Logits ---
        return self.classifier(pooled)  # -> [B, 50]
