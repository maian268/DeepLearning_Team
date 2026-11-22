"""src/train_lib.py - huấn luyện và chạy thí nghiệm (T-02 và T-03, Tấn).

Theo Phần C7, C9 của tài liệu kế hoạch. CFG chép nguyên văn, không tự đổi.

T-02: set_seed, chuẩn hóa mean/std, train 1 epoch, đánh giá, fit() có
      ReduceLROnPlateau + early stopping + lưu trọng số tốt nhất theo val_loss.
T-03: run_experiment(): chia fold (C3), chuẩn hóa, fit, nạp checkpoint tốt nhất,
      test đúng 1 lần, ghi results/exp{id}_fold{k}.json và checkpoints/*.pt.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# C7: cấu hình chung cho mọi EXP (không tự đổi, muốn đổi phải báo nhóm trưởng)
# ----------------------------------------------------------------------------
CFG = dict(
    seed=42, batch_size=32, max_epochs=60,
    optimizer="Adam", lr=1e-3, weight_decay=1e-4,
    scheduler="ReduceLROnPlateau", sched_factor=0.5, sched_patience=4, min_lr=1e-5,
    early_stop_patience=10, monitor="val_loss",
    grad_clip=5.0, dropout=0.3, label_smoothing=0.0,
    amp=True, num_workers=0,
)


def load_config(*yaml_paths) -> dict:
    """Đọc cấu hình: bắt đầu từ CFG (C7), rồi gộp lần lượt các file YAML.

    Ví dụ: load_config("configs/base.yaml", "configs/exp2.yaml"). File sau ghi đè
    file trước. Trả về dict phẳng; các khóa của exp (arch, n_mels, specaug...) cũng
    nằm trong đó. Giả định file YAML dùng đúng tên khóa như CFG; nếu An dùng cấu
    trúc lồng nhau (ví dụ train: {...}) thì chỉnh lại hàm này cho khớp.
    """
    import yaml                                  # import muộn: chỉ cần khi đọc YAML
    cfg = dict(CFG)
    for p in yaml_paths:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg.update(data)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Cố định mọi nguồn ngẫu nhiên (C7)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Chuẩn hóa (C2): hai số vô hướng tính trên log-mel của tập TRAIN của fold đó
# ----------------------------------------------------------------------------
def compute_norm_stats(X_train: torch.Tensor):
    """Trả về (mean, std) là hai số float, tính trên toàn bộ X_train."""
    mean = X_train.float().mean().item()
    std = X_train.float().std().item()
    return mean, std


def normalize(X: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """(x - mean) / (std + 1e-8). Dùng mean/std của train cho cả val và test."""
    return (X - mean) / (std + 1e-8)


# ----------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------
def train_one_epoch(model, X, y, optimizer, criterion, scaler, cfg, device,
                    use_amp, specaug_fn=None):
    """Train 1 epoch. X, y đã nằm trên device và X đã được chuẩn hóa.

    Trả về (train_loss, train_acc) trung bình theo mẫu.
    """
    model.train()
    n = X.shape[0]
    perm = torch.randperm(n, device=device)      # xáo trộn mỗi epoch (C7)
    bs = cfg["batch_size"]
    loss_sum, correct = 0.0, 0

    for i in range(0, n, bs):
        idx = perm[i:i + bs]
        xb, yb = X[idx], y[idx]
        if specaug_fn is not None:               # chỉ khi train, trên batch đã chuẩn hóa
            xb = specaug_fn(xb)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)               # cần trước khi clip khi dùng AMP
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * xb.shape[0]
        correct += (logits.argmax(1) == yb).sum().item()

    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, X, y, criterion, cfg, device, use_amp):
    """Đánh giá (val hoặc test). Trả về (loss, acc, preds, probs).

    probs có shape [N, 50] (softmax), dùng khi ghi JSON ở run_experiment.
    """
    model.eval()
    n = X.shape[0]
    bs = cfg["batch_size"]
    loss_sum, correct = 0.0, 0
    all_logits = []

    for i in range(0, n, bs):
        xb, yb = X[i:i + bs], y[i:i + bs]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.shape[0]
        correct += (logits.argmax(1) == yb).sum().item()
        all_logits.append(logits.float())

    logits = torch.cat(all_logits)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(1)
    return loss_sum / n, correct / n, preds, probs


def fit(model, X_tr, y_tr, X_va, y_va, cfg=None, specaug_fn=None, device=None,
        verbose=True):
    """Huấn luyện có early stopping, lưu trọng số tốt nhất theo val_loss.

    Yêu cầu: X_tr, X_va ĐÃ chuẩn hóa (dùng compute_norm_stats + normalize).
    specaug_fn: hàm nhận batch [B,1,n_mels,216] đã chuẩn hóa, trả về batch đã
    augment. Truyền None cho EXP không dùng SpecAugment.

    Trả về dict: best_state (state_dict trên CPU), history, best_epoch,
    best_val_loss, val_acc_best, train_time_s.
    """
    cfg = cfg or CFG
    device = device or get_device()
    use_amp = bool(cfg["amp"]) and device.type == "cuda"   # AMP chỉ có ý nghĩa trên GPU

    model = model.to(device)
    # Đưa toàn bộ dữ liệu của fold lên device một lần (C7)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_va, y_va = X_va.to(device), y_va.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["sched_factor"],
        patience=cfg["sched_patience"], min_lr=cfg["min_lr"])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = []
    best_val_loss = float("inf")
    best_epoch, best_val_acc, bad_epochs = 0, 0.0, 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, cfg["max_epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]     # lr dùng trong epoch này
        tr_loss, tr_acc = train_one_epoch(model, X_tr, y_tr, optimizer, criterion,
                                          scaler, cfg, device, use_amp, specaug_fn)
        va_loss, va_acc, _, _ = evaluate(model, X_va, y_va, criterion, cfg, device,
                                         use_amp)
        scheduler.step(va_loss)

        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                            train_acc=tr_acc, val_acc=va_acc, lr=lr))
        if verbose:
            print(f"ep {epoch:02d} | lr {lr:.1e} | train {tr_loss:.3f}/{tr_acc:.3f} "
                  f"| val {va_loss:.3f}/{va_acc:.3f}")

        if va_loss < best_val_loss:              # cải thiện theo val_loss
            best_val_loss, best_epoch, best_val_acc = va_loss, epoch, va_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stop_patience"]:
                if verbose:
                    print(f"Early stop sau epoch {epoch} (best epoch {best_epoch})")
                break

    return dict(best_state=best_state, history=history, best_epoch=best_epoch,
                best_val_loss=best_val_loss, val_acc_best=best_val_acc,
                train_time_s=time.time() - t0)


# ----------------------------------------------------------------------------
# T-03: chạy một thí nghiệm trên một fold
# ----------------------------------------------------------------------------
def split_indices(fold, test_fold: int):
    """Chia fold theo C3: test = k, val = (k mod 5) + 1, train = 3 fold còn lại.

    fold: tensor int64 [N] giá trị 1..5 (trường `fold` của cache).
    Trả về (train_idx, val_idx, test_idx) là các tensor chỉ số tăng dần.
    (Chỗ này tự tính theo C3 để không phụ thuộc get_splits của Quyên; nếu cần
    dùng get_splits thật thì thay hàm này, kết quả phải giống nhau.)
    """
    val_fold = test_fold % 5 + 1
    test_idx = (fold == test_fold).nonzero(as_tuple=True)[0]
    val_idx = (fold == val_fold).nonzero(as_tuple=True)[0]
    train_idx = ((fold != test_fold) & (fold != val_fold)).nonzero(as_tuple=True)[0]
    return train_idx, val_idx, test_idx


def _build_model(arch: str, n_mels: int, dropout: float) -> nn.Module:
    """Tạo CNN hoặc CRNN. (Khi Thanh xong build_model ở H-02 thì thay hàm này.)"""
    name = arch.lower()
    if name == "cnn":
        try:
            from .models_cnn import CNN
        except ImportError:
            from models_cnn import CNN
        return CNN(n_mels=n_mels, dropout=dropout)
    if name == "crnn":
        try:
            from .models_crnn import CRNN
        except ImportError:
            from models_crnn import CRNN
        return CRNN(n_mels=n_mels, dropout=dropout)
    raise ValueError(f"arch không hợp lệ: {arch!r} (chỉ có 'CNN' hoặc 'CRNN')")


def run_experiment(exp_id, arch, n_mels, specaug, test_fold, cfg=None,
                   cache_dir="cache", results_dir="results", ckpt_dir="checkpoints",
                   specaug_fn=None, device=None, verbose=True):
    """Chạy 1 EXP trên 1 fold test, ghi JSON (C9) và checkpoint.

    Bỏ qua nếu results/exp{exp_id}_fold{test_fold}.json đã tồn tại (trả về nội dung
    file đó). specaug_fn là hàm batch -> batch (bọc specaug_batch của Quyên); bắt
    buộc khi specaug=True. Quy trình test: nạp lại checkpoint tốt nhất rồi chạy
    test đúng 1 lần, không dùng kết quả test để chỉnh gì cả.
    """
    cfg = cfg or CFG
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    stem = f"exp{exp_id}_fold{test_fold}"
    json_path = os.path.join(results_dir, stem + ".json")
    ckpt_path = os.path.join(ckpt_dir, stem + ".pt")

    if os.path.exists(json_path):                # resume: bỏ qua fold đã xong
        if verbose:
            print(f"Bỏ qua {stem}: đã có {json_path}")
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    if specaug and specaug_fn is None:
        raise ValueError("specaug=True nhưng chưa truyền specaug_fn "
                         "(hàm batch -> batch, bọc specaug_batch của Quyên)")
    if not specaug:
        specaug_fn = None                        # EXP không SpecAugment thì tắt hẳn

    # --- Dữ liệu ---
    cache = torch.load(os.path.join(cache_dir, f"logmel{n_mels}.pt"),
                       map_location="cpu", weights_only=False)
    X, y, fold = cache["X"], cache["y"], cache["fold"]
    train_idx, val_idx, test_idx = split_indices(fold, test_fold)
    val_fold = test_fold % 5 + 1
    if len(fold) == 2000:                        # cache thật: 1200/400/400 (C3)
        assert (len(train_idx), len(val_idx), len(test_idx)) == (1200, 400, 400)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    # --- Chuẩn hóa bằng mean/std của train (C2) ---
    mean, std = compute_norm_stats(X_tr)
    X_tr, X_va, X_te = (normalize(a, mean, std) for a in (X_tr, X_va, X_te))

    # --- Mô hình (set_seed ngay trước khi khởi tạo để lặp lại được) ---
    set_seed(cfg["seed"])
    model = _build_model(arch, n_mels, cfg["dropout"])
    params = sum(p.numel() for p in model.parameters())

    # --- Huấn luyện ---
    device = device or get_device()
    out = fit(model, X_tr, y_tr, X_va, y_va, cfg, specaug_fn=specaug_fn,
              device=device, verbose=verbose)

    # --- Test đúng 1 lần, bằng checkpoint tốt nhất theo val_loss ---
    model.load_state_dict(out["best_state"])
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    _, acc, preds, probs = evaluate(model, X_te.to(device), y_te.to(device),
                                    criterion, cfg, device, use_amp)

    from sklearn.metrics import f1_score          # import muộn: chỉ cần khi chạy thật
    labels = y_te.tolist()
    preds_l = preds.cpu().tolist()
    macro_f1 = float(f1_score(labels, preds_l, average="macro",
                              labels=list(range(50)), zero_division=0))

    # --- Ghi checkpoint (C9) rồi ghi JSON sau cùng (JSON xuất hiện = fold đã xong) ---
    torch.save(dict(state_dict=out["best_state"], arch=arch, n_mels=n_mels,
                    mean=mean, std=std, classes=cache["classes"], cfg=cfg),
               ckpt_path)

    result = dict(
        exp_id=exp_id, arch=arch, n_mels=n_mels, specaug=bool(specaug),
        test_fold=test_fold, val_fold=val_fold, seed=cfg["seed"], params=params,
        best_epoch=out["best_epoch"], train_time_s=out["train_time_s"],
        val_acc_best=out["val_acc_best"], acc=float(acc), macro_f1=macro_f1,
        filenames=[cache["filenames"][i] for i in test_idx.tolist()],
        labels=labels, preds=preds_l, probs=probs.cpu().tolist(),
        history=out["history"],
    )
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(tmp, json_path)                   # ghi nguyên tử, tránh file dở dang
    if verbose:
        print(f"{stem}: acc={acc:.3f}, macro_f1={macro_f1:.3f}, "
              f"best_epoch={out['best_epoch']}, params={params:,}")
    return result
