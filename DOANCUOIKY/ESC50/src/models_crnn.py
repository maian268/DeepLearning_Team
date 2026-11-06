"""
Mô hình CRNN (Convolutional Recurrent Neural Network) cho phân loại âm thanh ESC-50.

"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Khối tích chập kép:
    [Conv2d (3x3) -> BatchNorm2d -> ReLU] x 2 -> MaxPool2d (2x2)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False  # bias=False vì đã có BatchNorm ngay phía sau
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


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
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_classes = n_classes

        # 1. 3 Khối tích chập CNN trích xuất không gian - tần số
        self.cnn_block1 = ConvBlock(in_channels=1, out_channels=32)
        self.cnn_block2 = ConvBlock(in_channels=32, out_channels=64)
        self.cnn_block3 = ConvBlock(in_channels=64, out_channels=128)

        # Trục tần số n_mels bị giảm 8 lần qua 3 tầng MaxPool(2)
        reduced_freq_bins = n_mels // 8
        in_features_proj = 128 * reduced_freq_bins

        # 2. Linear projection đưa về 128 chiều trước khi vào RNN
        self.proj = nn.Sequential(
            nn.Linear(in_features_proj, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # 3. BiGRU học phụ thuộc ngữ cảnh chuỗi theo trục thời gian
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # 4. Tầng phân loại đầu ra
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)  # 256 = 128 * 2 do Bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, 1, n_mels, 216]
        Output: [B, n_classes]
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
        logits = self.classifier(pooled)  # -> [B, 50]
        return logits


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Trả về (tổng số tham số, số tham số cần train)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM THỬ MÔ HÌNH CRNN TRÊN DỮ LIỆU GIẢ (SMOKE TEST H-01)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy kiểm thử trên: {device}\n")

    batch_size = 4
    time_steps = 216

    # Test cấu hình n_mels = 64
    print("[1] Kiểm tra với n_mels = 64:")
    model_64 = CRNN(n_mels=64, n_classes=50).to(device)
    dummy_input_64 = torch.randn(batch_size, 1, 64, time_steps).to(device)
    out_64 = model_64(dummy_input_64)
    total_64, train_64 = count_parameters(model_64)
    print(f"  + Input shape:  {list(dummy_input_64.shape)}")
    print(f"  + Output shape: {list(out_64.shape)}")
    print(f"  + Số tham số:   {train_64:,} tham số")
    assert out_64.shape == (batch_size, 50), f"Lỗi shape: {out_64.shape}"
    print("  -> Đạt yêu cầu cho 64 bins!\n")

    # Test cấu hình n_mels = 128 (Cấu hình chính cho EXP5)
    print("[2] Kiểm tra với n_mels = 128:")
    model_128 = CRNN(n_mels=128, n_classes=50).to(device)
    dummy_input_128 = torch.randn(batch_size, 1, 128, time_steps).to(device)
    out_128 = model_128(dummy_input_128)
    total_128, train_128 = count_parameters(model_128)
    print(f"  + Input shape:  {list(dummy_input_128.shape)}")
    print(f"  + Output shape: {list(out_128.shape)}")
    print(f"  + Số tham số:   {train_128:,} tham số (~1.06 triệu)")
    assert out_128.shape == (batch_size, 50), f"Lỗi shape: {out_128.shape}"
    print("  -> Đạt yêu cầu cho 128 bins!\n")

    print("=" * 60)
    print("TẤT CẢ KIỂM THỬ THÀNH CÔNG! SẴN SÀNG MERGE VÀO REGISTRY.")
    print("=" * 60)