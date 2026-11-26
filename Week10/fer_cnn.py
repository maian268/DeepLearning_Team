"""
BTL02 - Nhan dien cam xuc khuon mat (FER2013) bang CNN - PyTorch.

MOT FILE DUY NHAT cho ca nhom. Moi nguoi chi doi tham so dong lenh, khong sua code.

Cac lenh:
  train    : huan luyen 1 thi nghiem, luu ket qua vao runs/<name>/
  eval     : danh gia tren tap test (confusion matrix, anh dung/sai, bao cao tung lop)
  filters  : ve filter cua lop Conv (mac dinh lop cuoi) va feature map
  summary  : gop ket qua tat ca thi nghiem thanh bang (nhat ky thuc nghiem)

Du lieu (chon 1 trong 2 cach), dat trong --data_dir:
  (a) fer2013.csv  (file Kaggle: cot emotion, pixels, Usage)
  (b) thu muc Training/<lop>/*.png, PublicTest/<lop>/*.png, PrivateTest/<lop>/*.png

Vi du:
  python fer_cnn.py train --name base --data_dir data
  python fer_cnn.py train --name big --blocks 4 --convs 3 --base 64 --fc 2048
  python fer_cnn.py train --name sgd --opt sgd --lr 0.01 --aug 0
  python fer_cnn.py eval --name base
  python fer_cnn.py filters --name base
  python fer_cnn.py summary
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SPLITS = ["Training", "PublicTest", "PrivateTest"]  # train / validation / test
CLASSES_CSV = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# ----------------------------------------------------------------------------
# Du lieu
# ----------------------------------------------------------------------------
def load_data(data_dir):
    """Tra ve ({split: (x uint8 NxHxW, y int N)}, danh sach ten lop)."""
    cache = os.path.join(data_dir, "fer_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {s: (z[s + "_x"], z[s + "_y"]) for s in SPLITS}, [str(c) for c in z["classes"]]

    csv_path = os.path.join(data_dir, "fer2013.csv")
    xs = {s: [] for s in SPLITS}
    ys = {s: [] for s in SPLITS}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                s = row["Usage"]
                xs[s].append(np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48))
                ys[s].append(int(row["emotion"]))
        classes = CLASSES_CSV
    else:
        train_dir = os.path.join(data_dir, "Training")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                "Khong thay fer2013.csv hoac thu muc Training/ trong " + data_dir)
        classes = sorted(d for d in os.listdir(train_dir)
                         if os.path.isdir(os.path.join(train_dir, d)))
        for s in SPLITS:
            for ci, c in enumerate(classes):
                for p in sorted(glob.glob(os.path.join(data_dir, s, c, "*.png"))):
                    img = Image.open(p).convert("L").resize((48, 48))
                    xs[s].append(np.array(img, dtype=np.uint8))
                    ys[s].append(ci)

    data = {s: (np.stack(xs[s]), np.array(ys[s], dtype=np.int64)) for s in SPLITS}
    try:  # luu cache de cac lan chay sau nhanh hon
        np.savez(cache, classes=np.array(classes),
                 **{s + "_x": data[s][0] for s in SPLITS},
                 **{s + "_y": data[s][1] for s in SPLITS})
    except OSError:
        pass
    return data, classes


class FER(Dataset):
    def __init__(self, x, y, train=False, aug=True):
        self.x, self.y = x, y
        tf = []
        if train and aug:
            tf += [T.RandomHorizontalFlip(),
                   T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))]
        tf += [T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        self.tf = T.Compose(tf)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.x[i])), int(self.y[i])


def make_loaders(data, a):
    tr = FER(*data["Training"], train=True, aug=bool(a.aug))
    va = FER(*data["PublicTest"])
    te = FER(*data["PrivateTest"])
    kw = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(tr, shuffle=True, **kw),
            DataLoader(va, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw))


# ----------------------------------------------------------------------------
# Mo hinh: `blocks` khoi, moi khoi gom `convs` lop Conv3x3 + MaxPool + Dropout
# so kenh: base, 2*base, 4*base, ... (toi da 512)
# ----------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, n_classes, blocks=3, convs=2, base=64, fc=1024, dropout=0.2, bn=True):
        super().__init__()
        if not 1 <= blocks <= 5:
            raise ValueError("--blocks phai tu 1 den 5 (anh 48x48 chi pool duoc toi da 5 lan)")
        layers, cin = [], 1
        for b in range(blocks):
            cout = min(base * (2 ** b), 512)
            for _ in range(convs):
                layers.append(nn.Conv2d(cin, cout, 3, padding=1))
                if bn:
                    layers.append(nn.BatchNorm2d(cout))
                layers.append(nn.ReLU(inplace=True))
                cin = cout
            layers += [nn.MaxPool2d(2), nn.Dropout(dropout)]
        self.features = nn.Sequential(*layers)
        size = 48 // (2 ** blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(cin * size * size, fc), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(fc, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg, n_classes):
    return CNN(n_classes, cfg["blocks"], cfg["convs"], cfg["base"], cfg["fc"],
               cfg["dropout"], bool(cfg["bn"]))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_eval(model, loader, dev, crit=None):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        out = model(xb)
        if crit is not None:
            total_loss += crit(out, yb).item() * len(yb)
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(yb.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return total_loss / len(labels), float((preds == labels).mean()), preds, labels


def load_run(a):
    """Doc checkpoint tot nhat cua thi nghiem --name."""
    ck = torch.load(os.path.join(a.runs, a.name, "best.pt"), map_location="cpu")
    model = build_model(ck["args"], len(ck["classes"]))
    model.load_state_dict(ck["model"])
    return model.to(get_device()).eval(), ck


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def cmd_train(a):
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = get_device()
    out_dir = os.path.join(a.runs, a.name)
    os.makedirs(out_dir, exist_ok=True)

    data, classes = load_data(a.data_dir)
    tr, va, te = make_loaders(data, a)
    print("Train/Val/Test:", len(tr.dataset), len(va.dataset), len(te.dataset),
          "| lop:", classes, "| thiet bi:", dev)

    cfg = vars(a).copy()
    model = build_model(cfg, len(classes)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print("So tham so:", n_params)

    weight = None
    if a.class_weight:
        counts = np.bincount(data["Training"][1], minlength=len(classes)).astype(np.float32)
        weight = torch.tensor(counts.sum() / (len(counts) * counts)).to(dev)
    crit = nn.CrossEntropyLoss(weight=weight)

    if a.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=3)
             if a.sched else None)

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_epoch, bad, t0 = float("inf"), 0, 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tl, tc, n = 0.0, 0, 0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tc += (out.argmax(1) == yb).sum().item()
            n += len(yb)
        vl, vacc, _, _ = run_eval(model, va, dev, crit)
        if sched:
            sched.step(vl)
        for k, v in zip(hist, [ep, tl / n, tc / n, vl, vacc]):
            hist[k].append(v)
        print(f"Epoch {ep:3d} | train loss {tl/n:.4f} acc {tc/n:.4f} | "
              f"val loss {vl:.4f} acc {vacc:.4f} | lr {opt.param_groups[0]['lr']:.1e}")

        if vl < best_loss:
            best_loss, best_epoch, bad = vl, ep, 0
            torch.save({"model": model.state_dict(), "args": cfg, "classes": classes},
                       os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"EarlyStopping: {a.patience} epoch khong cai thien val_loss.")
                break

    # log + duong cong
    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(hist))
        w.writerows(zip(*hist.values()))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["epoch"], hist["train_acc"], label="train")
    ax[0].plot(hist["epoch"], hist["val_acc"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[1].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(f"{a.name} (best epoch {best_epoch})")
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # danh gia model tot nhat tren val + test (test chi dung 1 lan, sau khi train xong)
    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location=dev)
    model.load_state_dict(ck["model"])
    vl, vacc, _, _ = run_eval(model, va, dev, crit)
    _, tacc, _, _ = run_eval(model, te, dev, crit)
    result = {"name": a.name, "best_epoch": best_epoch, "epochs_run": len(hist["epoch"]),
              "val_loss": round(vl, 4), "val_acc": round(vacc, 4), "test_acc": round(tacc, 4),
              "params": n_params, "minutes": round((time.time() - t0) / 60, 1),
              "config": {k: cfg[k] for k in ["blocks", "convs", "base", "fc", "dropout", "bn",
                                              "opt", "lr", "wd", "sched", "aug",
                                              "class_weight", "batch", "seed"]}}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("KET QUA:", json.dumps({k: v for k, v in result.items() if k != "config"}))


# ----------------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------------
def save_grid(imgs, titles, path, title, rows=4, cols=8):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < len(imgs):
            ax.imshow(imgs[i], cmap="gray")
            ax.set_title(titles[i], fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def cmd_eval(a):
    dev = get_device()
    model, ck = load_run(a)
    classes = ck["classes"]
    out_dir = os.path.join(a.runs, a.name)
    data, _ = load_data(a.data_dir)
    x, y = data["PrivateTest"]
    _, acc, pred, lab = run_eval(model, DataLoader(FER(x, y), batch_size=256, num_workers=0), dev)

    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(lab, pred):
        cm[t, p] += 1
    pct = cm / cm.sum(1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(k)); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(classes)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{pct[i, j]:.0f}%", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black", fontsize=8)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix - test accuracy {acc * 100:.2f}%")
    fig.colorbar(im)
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    lines = [f"Test accuracy: {acc * 100:.2f}%", "", f"{'class':10s} precision recall   f1   support"]
    for c in range(k):
        tp = cm[c, c]
        prec = tp / max(cm[:, c].sum(), 1)
        rec = tp / max(cm[c].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        lines.append(f"{classes[c]:10s} {prec:9.3f} {rec:6.3f} {f1:6.3f} {cm[c].sum():7d}")
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    rng = np.random.RandomState(0)
    for tag, idx in [("wrong", np.where(pred != lab)[0]), ("right", np.where(pred == lab)[0])]:
        idx = rng.choice(idx, size=min(32, len(idx)), replace=False)
        save_grid([x[i] for i in idx],
                  [f"{classes[pred[i]]} ({classes[lab[i]]})" for i in idx],
                  os.path.join(out_dir, f"{tag}_cases.png"),
                  f"{tag.capitalize()} predictions: Predict (True)")
    print("Da luu confusion_matrix.png, report.txt, wrong_cases.png, right_cases.png vao", out_dir)


# ----------------------------------------------------------------------------
# filters: ve trong so cua lop Conv + feature map cua 1 anh test
# (voi lop co nhieu kenh vao, trung binh theo kenh vao de ve duoc thanh anh 3x3)
# ----------------------------------------------------------------------------
def cmd_filters(a):
    dev = get_device()
    model, ck = load_run(a)
    out_dir = os.path.join(a.runs, a.name)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    conv = convs[-1] if a.layer == "last" else convs[0]

    w = conv.weight.detach().cpu().numpy().mean(axis=1)  # (cout, 3, 3)
    n, cols = min(len(w), 64), 8
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(w[i], cmap="gray", interpolation="nearest")
    fig.suptitle(f"Filters cua lop Conv {a.layer} ({len(w)} filter, hien {n})")
    fig.savefig(os.path.join(out_dir, f"filters_{a.layer}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    data, _ = load_data(a.data_dir)
    x, y = data["PrivateTest"]
    img = x[a.index]
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    feats = {}
    h = conv.register_forward_hook(lambda m, i, o: feats.update(out=o.detach().cpu()))
    with torch.no_grad():
        logits = model(tf(Image.fromarray(img)).unsqueeze(0).to(dev))
    h.remove()
    fm = feats["out"][0].numpy()
    n = min(len(fm), 32)
    fig, axes = plt.subplots(4, 9, figsize=(13, 6.5))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"{ck['classes'][y[a.index]]} (that)\n{ck['classes'][logits.argmax(1).item()]} (doan)",
                      fontsize=7)
    for i in range(min(n, 35)):
        axes[i + 1].imshow(fm[i], cmap="viridis")
    fig.suptitle(f"Feature map cua lop Conv {a.layer}")
    fig.savefig(os.path.join(out_dir, f"featuremaps_{a.layer}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Da luu filters_*.png va featuremaps_*.png vao", out_dir)


# ----------------------------------------------------------------------------
# summary: nhat ky thuc nghiem tu dong
# ----------------------------------------------------------------------------
def cmd_summary(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(a.runs, "*", "result.json"))):
        with open(p) as f:
            r = json.load(f)
        rows.append({**{k: v for k, v in r.items() if k != "config"}, **r["config"]})
    if not rows:
        print("Chua co thi nghiem nao trong", a.runs)
        return
    rows.sort(key=lambda r: -r["val_acc"])
    keys = list(rows[0])
    with open(os.path.join(a.runs, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    show = ["name", "blocks", "convs", "base", "fc", "dropout", "opt", "lr", "aug",
            "best_epoch", "val_loss", "val_acc", "test_acc"]
    print("  ".join(f"{k:>9s}" for k in show))
    for r in rows:
        print("  ".join(f"{str(r[k]):>9s}" for k in show))
    print("\nDa luu", os.path.join(a.runs, "results.csv"))


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="BTL02 - FER2013 CNN (PyTorch)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data_dir", default="data")
        sp.add_argument("--runs", default="runs")
        sp.add_argument("--workers", type=int, default=2)

    t = sub.add_parser("train", help="huan luyen 1 thi nghiem")
    common(t)
    t.add_argument("--name", required=True, help="ten thi nghiem (thu muc runs/<name>)")
    t.add_argument("--blocks", type=int, default=3, help="so khoi Conv (1-5)")
    t.add_argument("--convs", type=int, default=2, help="so lop Conv moi khoi")
    t.add_argument("--base", type=int, default=64, help="so kenh khoi dau")
    t.add_argument("--fc", type=int, default=1024, help="so no-ron lop Dense")
    t.add_argument("--dropout", type=float, default=0.2)
    t.add_argument("--bn", type=int, default=1, help="1=dung BatchNorm")
    t.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.0, help="weight decay")
    t.add_argument("--sched", type=int, default=1, help="1=ReduceLROnPlateau")
    t.add_argument("--aug", type=int, default=1, help="1=data augmentation")
    t.add_argument("--class_weight", type=int, default=0, help="1=trong so lop (lop it anh)")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--patience", type=int, default=10, help="EarlyStopping")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="danh gia 1 thi nghiem tren tap test")
    common(e)
    e.add_argument("--name", required=True)
    e.set_defaults(fn=cmd_eval)

    f = sub.add_parser("filters", help="ve filter + feature map")
    common(f)
    f.add_argument("--name", required=True)
    f.add_argument("--layer", choices=["last", "first"], default="last")
    f.add_argument("--index", type=int, default=0, help="chi so anh test de ve feature map")
    f.set_defaults(fn=cmd_filters)

    s = sub.add_parser("summary", help="bang tong hop cac thi nghiem")
    common(s)
    s.set_defaults(fn=cmd_summary)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
