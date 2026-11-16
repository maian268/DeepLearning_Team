"""
src/models/__init__.py - Bộ điều phối khởi tạo kiến trúc mô hình (H-02).
"""

import sys
from pathlib import Path
import torch.nn as nn

# Trỏ đường dẫn về thư mục cha src/ nơi chứa models_cnn.py và models_crnn.py
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models_cnn import CNN, count_params
from models_crnn import CRNN

MODEL_REGISTRY = {
    "cnn": CNN,
    "crnn": CRNN,
}


def build_model(
    arch: str,
    n_mels: int = 128,
    n_classes: int = 50,
    dropout: float = 0.3,
) -> nn.Module:
    key = arch.strip().lower()
    if key not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Kiến trúc '{arch}' không được hỗ trợ. Chỉ hỗ trợ: {supported}.")

    return MODEL_REGISTRY[key](n_mels=n_mels, n_classes=n_classes, dropout=dropout)


__all__ = ["CNN", "CRNN", "MODEL_REGISTRY", "build_model", "count_params"]