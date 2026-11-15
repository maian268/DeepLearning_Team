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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Chuẩn hóa (C2): 
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

    probs có shape [N, 50] (softmax), dùng khi ghi JSON ở T-03.
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
