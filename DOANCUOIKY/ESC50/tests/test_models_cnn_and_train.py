"""tests/test_models_cnn_and_train.py - test cho T-01 và T-02 (Tấn).

Gộp trong một file, đều dùng dữ liệu giả và chạy được trên CPU:

  A. src/models_cnn.py (Phần C5)
     1. shape đầu ra
     2. số tham số không đổi theo n_mels
     3. shape sau từng khối
     4. conv không bias
     5. dropout 0,3 chỉ bật khi train

  B. src/train_lib.py (Phần C7)
     6. smoke test fit()
     7. early stopping
     8. hook SpecAugment

Chạy:
    python test_models_and_train.py        (chạy trực tiếp)
    pytest test_models_and_train.py -v     (nếu đã cài pytest)

File giả định models_cnn.py và train_lib.py nằm cùng thư mục, hoặc trong
src/ cạnh thư mục tests/. Nếu repo tổ chức khác thì sửa đoạn import dưới đây.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "src"), os.path.join(HERE, "..", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

import train_lib
from models_cnn import CNN, count_params
from train_lib import CFG, compute_norm_stats, fit, normalize, set_seed

N_MELS = 64
N_CLASSES = 50
EXPECTED_PARAMS = 1_185_490   # ~1,19 triệu theo C5, không đổi theo n_mels


def _fake_data(n, label_shift=0.0):
    """Dữ liệu giả [n,1,N_MELS,216] và nhãn xoay vòng 0..49."""
    y = torch.arange(n) % N_CLASSES
    X = torch.randn(n, 1, N_MELS, 216) + y.view(-1, 1, 1, 1) * label_shift
    return X, y


# ===========================================================================
# A. models_cnn.py
# ===========================================================================
def test_output_shape():
    """Input [B,1,n_mels,216] -> output [B,50] cho cả 64 và 128 bins."""
    for n in (64, 128):
        m = CNN(n_mels=n).eval()
        with torch.no_grad():
            out = m(torch.randn(4, 1, n, 216))
        assert out.shape == (4, 50), (n, out.shape)


def test_param_count_independent_of_n_mels():
    """Số tham số = 1.185.490 và không đổi theo n_mels (C5)."""
    assert count_params(CNN(n_mels=64)) == EXPECTED_PARAMS
    assert count_params(CNN(n_mels=128)) == EXPECTED_PARAMS


def test_feature_shapes_per_block():
    """Kích thước sau từng khối khớp bảng C5 (cột 64 bins và 128 bins)."""
    expected = {
        64: [(32, 32, 108), (64, 16, 54), (128, 8, 27), (256, 4, 13)],
        128: [(32, 64, 108), (64, 32, 54), (128, 16, 27), (256, 8, 13)],
    }
    for n, shapes in expected.items():
        m = CNN(n_mels=n).eval()
        h = torch.randn(2, 1, n, 216)
        got = []
        with torch.no_grad():
            for blk in m.features:
                h = blk(h)
                got.append(tuple(h.shape[1:]))
        assert got == shapes, (n, got)


def test_conv_has_no_bias():
    """C5: Conv3x3 với padding 1 và bias=False."""
    convs = [m for m in CNN().modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 8                      # 4 khối x 2 conv
    for c in convs:
        assert c.bias is None
        assert c.kernel_size == (3, 3) and c.padding == (1, 1)


def test_dropout_default_and_modes():
    """Dropout p=0,3; chỉ có tác dụng khi train, tắt khi eval."""
    m = CNN()
    assert m.drop.p == 0.3
    x = torch.randn(8, 1, 64, 216)

    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))       # eval: kết quả lặp lại được

    m.train()
    with torch.no_grad():
        assert not torch.allclose(m(x), m(x))   # train: dropout làm kết quả khác nhau


# ===========================================================================
# B. train_lib.py
# ===========================================================================
def test_fit_smoke():
    """fit() chạy hết, history/best_state đúng định dạng."""
    set_seed(42)
    X_tr, y_tr = _fake_data(200, label_shift=0.05)
    X_va, y_va = _fake_data(100, label_shift=0.05)

    mean, std = compute_norm_stats(X_tr)
    X_tr, X_va = normalize(X_tr, mean, std), normalize(X_va, mean, std)
    # Sau chuẩn hóa, train phải có mean ~ 0 và std ~ 1
    assert abs(X_tr.mean().item()) < 1e-3
    assert abs(X_tr.std().item() - 1.0) < 1e-3

    cfg = dict(CFG, max_epochs=3)
    out = fit(CNN(n_mels=N_MELS, dropout=cfg["dropout"]),
              X_tr, y_tr, X_va, y_va, cfg, verbose=False)

    assert len(out["history"]) == 3
    assert set(out["history"][0]) == {"epoch", "train_loss", "val_loss",
                                      "train_acc", "val_acc", "lr"}
    assert out["best_state"] is not None
    assert 1 <= out["best_epoch"] <= 3
    # Trọng số tốt nhất phải nằm trên CPU để lưu checkpoint
    assert all(v.device.type == "cpu" for v in out["best_state"].values())


def test_early_stopping():
    """Dừng đúng khi val_loss không cải thiện."""
    set_seed(42)
    X, y = _fake_data(64)

    # Ép val_loss luôn bằng 1.0: epoch 1 là tốt nhất, sau đó không cải thiện
    original_evaluate = train_lib.evaluate
    train_lib.evaluate = lambda *a, **k: (1.0, 0.5, None, None)
    try:
        cfg = dict(CFG, max_epochs=10, early_stop_patience=2)
        out = fit(CNN(n_mels=N_MELS), X, y, X, y, cfg, verbose=False)
    finally:
        train_lib.evaluate = original_evaluate   # luôn khôi phục

    assert len(out["history"]) == 3, len(out["history"])  # epoch 1, 2, 3 rồi dừng
    assert out["best_epoch"] == 1


def test_specaug_hook():
    """specaug_fn chỉ được gọi lúc train, đúng số lần."""
    set_seed(42)
    X, y = _fake_data(64)
    calls = []

    def fake_specaug(x):
        calls.append(tuple(x.shape))
        return x

    cfg = dict(CFG, max_epochs=1)
    fit(CNN(n_mels=N_MELS), X, y, X, y, cfg, specaug_fn=fake_specaug,
        verbose=False)

    # 64 mẫu / batch 32 = 2 lần, chỉ gọi khi train (không gọi khi val)
    assert calls == [(32, 1, N_MELS, 216), (32, 1, N_MELS, 216)], calls


# ===========================================================================
# Chạy trực tiếp
# ===========================================================================
if __name__ == "__main__":
    tests = [
        test_output_shape,
        test_param_count_independent_of_n_mels,
        test_feature_shapes_per_block,
        test_conv_has_no_bias,
        test_dropout_default_and_modes,
        test_fit_smoke,
        test_early_stopping,
        test_specaug_hook,
    ]
    for i, t in enumerate(tests, 1):
        t()
        print(f"{i}/{len(tests)} {t.__name__} OK")
    print("ALL OK")
