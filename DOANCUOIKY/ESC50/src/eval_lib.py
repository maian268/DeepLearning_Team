"""eval_lib.py - Đánh giá, tổng hợp kết quả và phân tích lỗi cho đề tài ESC-50.

Người phụ trách: An (nhóm trưởng, SV3).  Việc: A-03, A-04, A-05.

Đầu vào : results/exp{id}_fold{k}.json  (định dạng ở Phần C9 của tài liệu kế hoạch)
Đầu ra  : DataFrame / mảng numpy để notebook 03_results.ipynb vẽ hình và lưu bảng.

Quy ước: Accuracy và Macro-F1 trong các bảng tổng hợp tính theo PHẦN TRĂM (0-100),
mean ± std qua các fold với std mẫu (ddof=1).
"""
import os
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix

RESULT_DIR = "results"
N_CLASSES = 50

GROUP_NAMES = ["Động vật", "Thiên nhiên & nước", "Âm thanh người",
               "Trong nhà", "Đô thị & ngoài trời"]
GROUP_COLORS = ["#2E7D32", "#1565C0", "#EF6C00", "#6A1B9A", "#C62828"]

# Tên lớp ESC-50 theo thứ tự target 0..49 (dùng khi chưa đọc được esc50.csv)
DEFAULT_CLASS_NAMES = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
    "wind", "pouring_water", "toilet_flush", "thunderstorm",
    "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
    "laughing", "brushing_teeth", "snoring", "drinking_sipping",
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks",
    "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick",
    "glass_breaking",
    "helicopter", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells",
    "airplane", "fireworks", "hand_saw",
]

# Cặp thực nghiệm dùng để trả lời từng RQ: (nhãn, EXP gốc a, EXP so sánh b)
RQ_PAIRS = [
    ("RQ1 (64 -> 128 bins, không SpecAug)", 1, 3),
    ("RQ1 (64 -> 128 bins, có SpecAug)", 2, 4),
    ("RQ2 (thêm SpecAug, 64 bins)", 1, 2),
    ("RQ2 (thêm SpecAug, 128 bins)", 3, 4),
    ("RQ3 (CNN -> CRNN, 128 bins, có SpecAug)", 4, 5),
]


# ----------------------------------------------------------------------------
# Đọc kết quả
# ----------------------------------------------------------------------------
def load_fold(exp_id, fold, result_dir=RESULT_DIR):
    path = os.path.join(result_dir, f"exp{exp_id}_fold{fold}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_exp(exp_id, folds=range(1, 6), result_dir=RESULT_DIR):
    """Trả về dict {fold: kết quả}, chỉ gồm các fold đã có file JSON."""
    out = {}
    for k in folds:
        p = os.path.join(result_dir, f"exp{exp_id}_fold{k}.json")
        if os.path.exists(p):
            out[k] = load_fold(exp_id, k, result_dir)
    return out


def availability_table(exp_ids, result_dir=RESULT_DIR, folds=range(1, 6)):
    """Bảng kiểm tra EXP nào đã có đủ fold nào (dùng cho buổi họp giữa kỳ)."""
    rows = []
    for e in exp_ids:
        row = {"EXP": f"EXP{e}"}
        n = 0
        for k in folds:
            ok = os.path.exists(os.path.join(result_dir, f"exp{e}_fold{k}.json"))
            row[f"fold{k}"] = "có" if ok else "THIẾU"
            n += int(ok)
        row["Số fold"] = n
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Độ đo
# ----------------------------------------------------------------------------
def compute_metrics(labels, preds):
    """Accuracy và Macro-F1 (50 lớp). Trả về (acc, macro_f1) dạng 0-1."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    acc = float((labels == preds).mean())
    f1 = float(f1_score(labels, preds, average="macro",
                        labels=list(range(N_CLASSES)), zero_division=0))
    return acc, f1


def check_consistency(res, tol=1e-3):
    """Đối chiếu số tính lại với acc, macro_f1 ghi trong JSON. True nếu khớp."""
    acc, f1 = compute_metrics(res["labels"], res["preds"])
    ok = abs(acc - res["acc"]) <= tol and abs(f1 - res["macro_f1"]) <= tol
    if not ok:
        print(f"CẢNH BÁO: lệch số liệu ở EXP{res.get('exp_id')} fold{res.get('test_fold')}: "
              f"tính lại acc={acc:.4f}, F1={f1:.4f} nhưng JSON ghi acc={res['acc']}, "
              f"F1={res['macro_f1']} (kiểm tra JSON có lưu theo % không).")
    return ok


def _pool(results):
    rs = list(results.values()) if isinstance(results, dict) else list(results)
    y = np.concatenate([np.asarray(r["labels"]) for r in rs])
    p = np.concatenate([np.asarray(r["preds"]) for r in rs])
    return y, p


def confusion_pooled(results, normalize=False):
    """Confusion matrix 50x50 gộp các fold. normalize=True: chuẩn hóa theo hàng."""
    y, p = _pool(results)
    cm = confusion_matrix(y, p, labels=list(range(N_CLASSES)))
    if normalize:
        return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return cm


def per_class_f1(results):
    y, p = _pool(results)
    return f1_score(y, p, average=None, labels=list(range(N_CLASSES)), zero_division=0)


def per_class_acc(results):
    """Độ chính xác theo từng lớp (recall) từ confusion matrix gộp."""
    cm = confusion_pooled(results)
    return np.diag(cm) / np.maximum(cm.sum(axis=1), 1)


def class_group(class_id):
    """ESC-50 chia 5 nhóm, mỗi nhóm 10 lớp liên tiếp theo target."""
    return int(class_id) // 10


# ----------------------------------------------------------------------------
# Tổng hợp qua 5 fold và kiểm định
# ----------------------------------------------------------------------------
def _std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def summarize(exp_ids, result_dir=RESULT_DIR):
    """Trả về (DataFrame bảng tổng hợp, per_fold).

    per_fold[e] = {fold: (acc%, macroF1%)}  để dùng cho kiểm định cặp.
    """
    rows, per_fold = [], {}
    for e in exp_ids:
        d = load_exp(e, result_dir=result_dir)
        if not d:
            continue
        pf = {}
        for k, r in d.items():
            a, f = compute_metrics(r["labels"], r["preds"])
            pf[k] = (a * 100, f * 100)
        per_fold[e] = pf
        A = np.array([v[0] for v in pf.values()])
        F = np.array([v[1] for v in pf.values()])
        rows.append({
            "EXP": f"EXP{e}",
            "Số fold": len(pf),
            "Accuracy (%)": f"{A.mean():.1f} ± {_std(A):.1f}",
            "Macro-F1 (%)": f"{F.mean():.1f} ± {_std(F):.1f}",
            "acc_mean": A.mean(), "acc_std": _std(A),
            "f1_mean": F.mean(), "f1_std": _std(F),
        })
    return pd.DataFrame(rows), per_fold


def paired_test(per_fold, a, b, metric=1):
    """t-test cặp theo fold: EXP b so với EXP a. metric: 0 = Accuracy, 1 = Macro-F1."""
    common = sorted(set(per_fold[a]) & set(per_fold[b]))
    xa = np.array([per_fold[a][k][metric] for k in common])
    xb = np.array([per_fold[b][k][metric] for k in common])
    diff = xb - xa
    if len(common) >= 2 and np.std(diff, ddof=1) > 0:
        p = float(stats.ttest_rel(xb, xa).pvalue)
    else:
        p = float("nan")
    return {"Cặp so sánh": f"EXP{b} so với EXP{a}", "Số fold chung": len(common),
            "Chênh lệch TB (điểm %)": round(float(diff.mean()), 2),
            "Std chênh lệch": round(_std(diff), 2) if len(common) > 1 else float("nan"),
            "p-value": round(p, 3) if p == p else float("nan")}


def rq_table(per_fold, metric=1):
    """Bảng RQ -> cặp EXP -> chênh lệch -> p-value (dùng cho tab_rq)."""
    rows = []
    for label, a, b in RQ_PAIRS:
        if a in per_fold and b in per_fold:
            r = paired_test(per_fold, a, b, metric)
            rows.append({"Câu hỏi": label, **r})
    return pd.DataFrame(rows)


def gap_table(exp_ids, result_dir=RESULT_DIR):
    """Khoảng cách train - val (điểm % accuracy) tại epoch có val_loss thấp nhất.

    Dùng để trả lời RQ2: SpecAugment có làm giảm khoảng cách này không.
    """
    rows = []
    for e in exp_ids:
        d = load_exp(e, result_dir=result_dir)
        gaps = []
        for r in d.values():
            h = r.get("history") or []
            if not h:
                continue
            best = min(h, key=lambda x: x["val_loss"])
            gaps.append((best["train_acc"] - best["val_acc"]) * 100)
        if gaps:
            rows.append({"EXP": f"EXP{e}", "Số fold": len(gaps),
                         "Khoảng cách train-val (điểm %)":
                             f"{np.mean(gaps):.1f} ± {_std(gaps):.1f}"})
    return pd.DataFrame(rows)


def cost_table(exp_ids, result_dir=RESULT_DIR):
    """Chi phí tính toán từ các trường params, best_epoch, train_time_s trong JSON."""
    rows = []
    for e in exp_ids:
        d = load_exp(e, result_dir=result_dir)
        if not d:
            continue
        first = next(iter(d.values()))
        t = [r.get("train_time_s") for r in d.values() if r.get("train_time_s") is not None]
        be = [r.get("best_epoch") for r in d.values() if r.get("best_epoch") is not None]
        rows.append({"EXP": f"EXP{e}", "Kiến trúc": first.get("arch"),
                     "Mel bins": first.get("n_mels"),
                     "Số tham số": first.get("params"),
                     "Epoch tốt nhất (TB)": round(float(np.mean(be)), 1) if be else None,
                     "Thời gian train/fold (giây, TB)": round(float(np.mean(t)), 0) if t else None})
    return pd.DataFrame(rows)


def select_best_exp(exp_ids, result_dir=RESULT_DIR):
    """Chọn EXP tốt nhất theo val_acc_best trung bình (không nhìn test).

    Trả về (exp_id, cách chọn). Nếu JSON không có val_acc_best thì dùng Macro-F1 test
    chỉ để minh họa và ghi rõ trong báo cáo.
    """
    best, best_v = None, -1.0
    for e in exp_ids:
        d = load_exp(e, result_dir=result_dir)
        v = [r["val_acc_best"] for r in d.values() if r.get("val_acc_best") is not None]
        if len(v) == len(d) and d:
            m = float(np.mean(v))
            if m > best_v:
                best, best_v = e, m
    if best is not None:
        return best, "val_acc_best trung bình (chọn theo validation)"
    df, _ = summarize(exp_ids, result_dir)
    row = df.sort_values("f1_mean", ascending=False).iloc[0]
    return int(row["EXP"].replace("EXP", "")), "Macro-F1 test trung bình (JSON thiếu val_acc_best)"


# ----------------------------------------------------------------------------
# Phân tích lỗi
# ----------------------------------------------------------------------------
def get_classes(meta_csv=None):
    """Danh sách 50 tên lớp. Ưu tiên đọc esc50.csv, không có thì dùng bản mặc định."""
    if meta_csv and os.path.exists(meta_csv):
        df = pd.read_csv(meta_csv).drop_duplicates("target").sort_values("target")
        names = df["category"].tolist()
        if len(names) == N_CLASSES:
            return names
    return list(DEFAULT_CLASS_NAMES)


def top_confused_pairs(cm, classes, k=10, symmetric=False):
    """k cặp lớp bị nhầm nhiều nhất từ confusion matrix (đếm số mẫu, chưa chuẩn hóa).

    symmetric=False: cặp có hướng (thật -> đoán). symmetric=True: gộp hai chiều.
    """
    rows = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i == j:
                continue
            if symmetric:
                if i < j:
                    c = int(cm[i, j] + cm[j, i])
                    if c > 0:
                        rows.append((classes[i], classes[j], c, i, j))
            elif cm[i, j] > 0:
                rows.append((classes[i], classes[j], int(cm[i, j]), i, j))
    rows.sort(key=lambda t: -t[2])
    return pd.DataFrame(rows[:k], columns=["Lớp A (thật)", "Lớp B (đoán)",
                                           "Số lần nhầm", "id_A", "id_B"])


def find_error_examples(results, true_id, pred_id, n=2):
    """Lấy n mẫu có nhãn thật true_id nhưng bị đoán thành pred_id (ưu tiên mẫu sai chắc chắn)."""
    cands = []
    for k, r in results.items():
        y, p, pr = np.asarray(r["labels"]), np.asarray(r["preds"]), np.asarray(r["probs"])
        for i in np.where((y == true_id) & (p == pred_id))[0]:
            cands.append({"fold": k, "idx": int(i), "filename": r["filenames"][i],
                          "p_pred": float(pr[i][pred_id]), "p_true": float(pr[i][true_id])})
    cands.sort(key=lambda c: -c["p_pred"])
    return cands[:n]


def most_confident_errors(results, classes, n=20):
    """n mẫu sai mà mô hình tự tin nhất - dùng để nghe lại và xét lỗi do dữ liệu."""
    rows = []
    for k, r in results.items():
        y, p, pr = np.asarray(r["labels"]), np.asarray(r["preds"]), np.asarray(r["probs"])
        for i in np.where(y != p)[0]:
            rows.append({"fold": k, "filename": r["filenames"][i],
                         "nhãn thật": classes[y[i]], "nhãn đoán": classes[p[i]],
                         "p_đoán": round(float(pr[i][p[i]]), 3),
                         "p_thật": round(float(pr[i][y[i]]), 3)})
    df = pd.DataFrame(rows).sort_values("p_đoán", ascending=False).head(n).reset_index(drop=True)
    df["kết luận (mô hình / dữ liệu)"] = ""
    df["ghi chú khi nghe"] = ""
    return df


# ----------------------------------------------------------------------------
# Dữ liệu GIẢ để test (KHÔNG dùng số liệu này trong báo cáo)
# ----------------------------------------------------------------------------
def make_fake_results(out_dir="tests/fake_results", exps=(1, 2, 3, 4, 5), seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    labels = np.repeat(np.arange(N_CLASSES), 8)  # 400 mẫu, 8 mẫu/lớp
    cfg = {1: ("cnn", 64, False), 2: ("cnn", 64, True), 3: ("cnn", 128, False),
           4: ("cnn", 128, True), 5: ("crnn", 128, True), 6: ("crnn", 128, False)}
    for e in exps:
        arch, n_mels, spec = cfg[e]
        skill = 0.30 + 0.05 * e
        for k in range(1, 6):
            preds = np.where(rng.random(400) < skill, labels, rng.integers(0, N_CLASSES, 400))
            # nhầm có chủ đích: lớp 1 (rooster) hay bị đoán thành 6 (hen), lớp 10 thành 16
            for a, b in ((1, 6), (10, 16)):
                m = (labels == a) & (rng.random(400) < 0.35)
                preds[m] = b
            probs = rng.dirichlet(np.ones(N_CLASSES) * 0.3, 400)
            probs[np.arange(400), preds] += 1.0
            probs = (probs / probs.sum(axis=1, keepdims=True)).round(4)
            acc, f1 = compute_metrics(labels, preds)
            hist = []
            for ep in range(1, 13):
                tr = min(0.98, 0.1 + 0.08 * ep)
                va = tr - (0.10 if spec else 0.25) * min(1.0, ep / 8) + rng.normal(0, 0.01)
                hist.append({"epoch": ep, "train_loss": round(3.5 - 0.25 * ep, 3),
                             "val_loss": round(3.6 - 0.2 * ep + (0.0 if spec else 0.02 * ep), 3),
                             "train_acc": round(tr, 3), "val_acc": round(max(va, 0.0), 3), "lr": 1e-3})
            res = {"exp_id": e, "arch": arch, "n_mels": n_mels, "specaug": spec,
                   "test_fold": k, "val_fold": k % 5 + 1, "seed": 42,
                   "params": 1190000 if arch == "cnn" else 1060000,
                   "best_epoch": int(rng.integers(30, 55)),
                   "train_time_s": float(rng.integers(180, 420)),
                   "val_acc_best": round(float(acc + rng.normal(0, 0.02)), 4),
                   "acc": acc, "macro_f1": f1,
                   "filenames": [f"{k}-{100000 + i}-A-{int(labels[i])}.wav" for i in range(400)],
                   "labels": labels.tolist(), "preds": preds.tolist(),
                   "probs": probs.tolist(), "history": hist}
            with open(os.path.join(out_dir, f"exp{e}_fold{k}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f)
    return out_dir
