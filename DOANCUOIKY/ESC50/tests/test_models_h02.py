"""
tests/test_models_h02.py - Unit test cho hạng mục H-02 (CRNN & build_model).

"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn as nn

from models import CNN, CRNN, build_model, count_params


def test_build_model_factory():
    """Kiểm tra build_model trả về đúng class và báo lỗi khi sai tên kiến trúc."""
    m_cnn = build_model("CNN", n_mels=128, n_classes=50, dropout=0.3)
    m_crnn = build_model("crnn", n_mels=64, n_classes=50, dropout=0.3)

    assert isinstance(m_cnn, CNN)
    assert isinstance(m_crnn, CRNN)

    try:
        build_model("resnet50")
        assert False, "Phải báo ValueError khi truyền tên kiến trúc không hợp lệ"
    except ValueError:
        pass


def test_crnn_shapes_and_params():
    """Kiểm tra kích thước đầu ra [B, 50] và số tham số chuẩn cho 64 & 128 bins."""
    expected_params = {
        128: 1_056_594,
        64: 925_522,
    }
    for n_mels, exp_p in expected_params.items():
        model = build_model("crnn", n_mels=n_mels, n_classes=50).eval()
        assert count_params(model) == exp_p, f"Sai số tham số ở {n_mels} bins"

        x = torch.randn(4, 1, n_mels, 216)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 50)


def test_crnn_architecture_details():
    """Kiểm tra 3 khối ConvBlock không bias và 2 tầng BiGRU."""
    model = build_model("crnn", n_mels=128, dropout=0.3)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 6
    for c in convs:
        assert c.bias is None

    assert model.gru.bidirectional is True
    assert model.gru.num_layers == 2
    assert model.gru.hidden_size == 128


def test_crnn_backward_step():
    """Kiểm tra tính đạo hàm và cập nhật trọng số (1 bước train giả lập)."""
    model = build_model("crnn", n_mels=64, n_classes=50).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(2, 1, 64, 216)
    y = torch.tensor([3, 42])

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


if __name__ == "__main__":
    tests = [
        test_build_model_factory,
        test_crnn_shapes_and_params,
        test_crnn_architecture_details,
        test_crnn_backward_step,
    ]
    for idx, fn in enumerate(tests, 1):
        fn()
        print(f"{idx}/{len(tests)} {fn.__name__} OK")
    print("ALL H-02 TESTS OK")