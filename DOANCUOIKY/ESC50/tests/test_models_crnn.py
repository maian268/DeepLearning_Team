"""tests/test_models_crnn.py - Unit test kiểm tra kiến trúc CRNN (Phần C6).

Chạy:
    python tests/test_models_crnn.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "src"), os.path.join(HERE, "..", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

from models_crnn import CRNN, count_params
import train_lib
from train_lib import CFG, fit, set_seed

EXPECTED_PARAMS_128 = 1_056_594  # ~1,06 triệu với 128 bins
EXPECTED_PARAMS_64 = 925_522     # ~0,93 triệu với 64 bins


def test_crnn_output_shape():
    """Input [B, 1, n_mels, 216] -> output [B, 50] cho cả 64 và 128 bins."""
    for n in (64, 128):
        m = CRNN(n_mels=n, n_classes=50).eval()
        with torch.no_grad():
            out = m(torch.randn(4, 1, n, 216))
        assert out.shape == (4, 50), (n, out.shape)


def test_crnn_param_count():
    """Số tham số khớp thiết kế C6: 1.056.594 (128 bins) và 925.522 (64 bins)."""
    assert count_params(CRNN(n_mels=128)) == EXPECTED_PARAMS_128
    assert count_params(CRNN(n_mels=64)) == EXPECTED_PARAMS_64


def test_crnn_conv_and_gru_config():
    """Kiểm tra 3 khối CNN (6 lớp Conv2d không bias) và tầng BiGRU 2 lớp."""
    m = CRNN(n_mels=128, dropout=0.3)
    convs = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    assert len(convs) == 6  # 3 khối đầu x 2 conv
    for c in convs:
        assert c.bias is None

    assert isinstance(m.gru, nn.GRU)
    assert m.gru.bidirectional is True
    assert m.gru.num_layers == 2
    assert m.gru.hidden_size == 128


def test_crnn_dropout_modes():
    """Dropout tắt khi eval (kết quả cố định) và bật khi train."""
    m = CRNN(n_mels=128, dropout=0.3)
    x = torch.randn(4, 1, 128, 216)

    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))

    m.train()
    with torch.no_grad():
        assert not torch.allclose(m(x), m(x))


def test_crnn_fit_smoke():
    """Kiểm tra CRNN chạy tương thích hoàn toàn với hàm fit() của train_lib.py."""
    set_seed(42)
    X = torch.randn(64, 1, 64, 216)
    y = torch.arange(64) % 50
    cfg = dict(CFG, max_epochs=2)

    out = fit(CRNN(n_mels=64, dropout=cfg["dropout"]), X, y, X, y, cfg, verbose=False)
    assert len(out["history"]) == 2
    assert out["best_state"] is not None


if __name__ == "__main__":
    tests = [
        test_crnn_output_shape,
        test_crnn_param_count,
        test_crnn_conv_and_gru_config,
        test_crnn_dropout_modes,
        test_crnn_fit_smoke,
    ]
    for i, t in enumerate(tests, 1):
        t()
        print(f"{i}/{len(tests)} {t.__name__} OK")
    print("ALL CRNN TESTS OK")