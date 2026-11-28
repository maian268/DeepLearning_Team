"""error_analysis.py - A-05: phân tích lỗi (confusion matrix, F1 theo lớp, cặp lớp hay nhầm).

Đầu vào : results/exp{id}_fold{k}.json, cache/logmel{64|128}.pt (để vẽ spectrogram),
          meta/esc50.csv (tên lớp)
Đầu ra  : figures/fig6_confmat_best.png, figures/fig6_f1_per_class.png,
          figures/fig6_errors_pair{1..5}.png, tables/tab_confused_pairs.csv,
          tables/tab_f1_theo_nhom.csv, tables/tab_nghe_lai.csv

Chạy:  python -m src.error_analysis --results results --cache cache --meta data/ESC-50-master/meta/esc50.csv
"""
import os
import argparse

import numpy as np
import pandas as pd

from src import eval_lib as ev


def _save(fig, path, fake):
    import matplotlib.pyplot as plt
    if fake:
        fig.text(0.5, 0.5, "DỮ LIỆU GIẢ - CHỈ ĐỂ TEST", fontsize=30, color="red",
                 alpha=0.25, ha="center", va="center", rotation=25)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(cm_norm, classes, title, path, fake=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(50)); ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticks(range(50)); ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Lớp dự đoán"); ax.set_ylabel("Lớp thật")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.02)
    return _save(fig, path, fake)


def plot_f1_per_class(f1c, classes, title, path, fake=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    order = np.argsort(f1c)
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.barh(range(50), f1c[order], color=[ev.GROUP_COLORS[ev.class_group(i)] for i in order])
    ax.set_yticks(range(50)); ax.set_yticklabels([classes[i] for i in order], fontsize=7)
    ax.set_xlabel("F1 theo lớp"); ax.set_xlim(0, 1)
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(ev.GROUP_COLORS, ev.GROUP_NAMES)],
              loc="lower right", fontsize=8)
    ax.set_title(title)
    return _save(fig, path, fake)


def make_logmel_getter(cache_dir, n_mels):
    """Trả về hàm filename -> log-Mel [n_mels, 216] đọc từ cache; None nếu chưa có cache."""
    path = os.path.join(cache_dir, f"logmel{n_mels}.pt")
    if not os.path.exists(path):
        print(f"CHƯA THẤY cache {path} -> bỏ qua hình spectrogram của các cặp lỗi.")
        return None
    import torch
    cache = torch.load(path, map_location="cpu")
    idx = {f: i for i, f in enumerate(cache["filenames"])}
    return lambda fname: cache["X"][idx[fname]][0].numpy()


def plot_pair_examples(results, pairs, get_logmel, fig_dir, n_pairs=5, n_examples=2,
                       prefix="", fake=False):
    """Vẽ n_examples mẫu sai chắc chắn nhất cho n_pairs cặp lớp nhầm nhiều nhất."""
    import matplotlib.pyplot as plt
    paths = []
    for n, row in pairs.head(n_pairs).iterrows():
        ex = ev.find_error_examples(results, int(row["id_A"]), int(row["id_B"]), n=n_examples)
        k = max(len(ex), 1)
        fig, axes = plt.subplots(1, k, figsize=(5.5 * k, 3.8), squeeze=False)
        for ax, e in zip(axes[0], ex):
            S = get_logmel(e["filename"])
            ax.imshow(S, origin="lower", aspect="auto", cmap="magma")
            ax.set_title(f"Thật: {row['Lớp A (thật)']} | Đoán: {row['Lớp B (đoán)']}\n"
                         f"{e['filename']}  (p_đoán={e['p_pred']:.2f}, p_thật={e['p_true']:.2f})",
                         fontsize=8)
            ax.set_xlabel("Time frame"); ax.set_ylabel("Mel bin")
        fig.suptitle(f"Cặp lỗi {n + 1}: {int(row['Số lần nhầm'])} lần nhầm", fontsize=10)
        paths.append(_save(fig, os.path.join(fig_dir, f"{prefix}fig6_errors_pair{n + 1}.png"), fake))
    return paths


def run(exp_id=None, exp_ids=(1, 2, 3, 4, 5), result_dir="results", cache_dir="cache",
        meta_csv=None, fig_dir="figures", tab_dir="tables", k_pairs=10, n_pairs_fig=5,
        n_examples=2, fake=False, get_logmel=None):
    """Chạy toàn bộ phân tích lỗi cho EXP tốt nhất (chọn theo validation nếu exp_id=None)."""
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    pre = "fake_" if fake else ""

    if exp_id is None:
        exp_id, basis = ev.select_best_exp(list(exp_ids), result_dir)
    else:
        basis = "chọn tay"
    print(f"Phân tích lỗi cho EXP{exp_id} ({basis})")

    results = ev.load_exp(exp_id, result_dir=result_dir)
    if not results:
        raise FileNotFoundError(f"Chưa có kết quả EXP{exp_id} trong {result_dir}")
    classes = ev.get_classes(meta_csv)
    cm = ev.confusion_pooled(results)
    cmn = ev.confusion_pooled(results, normalize=True)
    f1c = ev.per_class_f1(results)

    out = {"exp_id": exp_id}
    out["confmat"] = plot_confusion(
        cmn, classes, f"Confusion matrix gộp {len(results)} fold - EXP{exp_id} (chuẩn hóa theo hàng)",
        os.path.join(fig_dir, f"{pre}fig6_confmat_best.png"), fake)
    out["f1_per_class"] = plot_f1_per_class(
        f1c, classes, f"F1 theo lớp - EXP{exp_id} (màu theo 5 nhóm âm thanh)",
        os.path.join(fig_dir, f"{pre}fig6_f1_per_class.png"), fake)

    grp = (pd.DataFrame({"Nhóm": [ev.GROUP_NAMES[ev.class_group(i)] for i in range(50)], "F1": f1c})
           .groupby("Nhóm")["F1"].mean().round(3).sort_values().reset_index()
           .rename(columns={"F1": "F1 trung bình"}))
    grp.to_csv(os.path.join(tab_dir, f"{pre}tab_f1_theo_nhom.csv"), index=False, encoding="utf-8-sig")

    pairs = ev.top_confused_pairs(cm, classes, k=k_pairs)
    pairs.to_csv(os.path.join(tab_dir, f"{pre}tab_confused_pairs.csv"), index=False, encoding="utf-8-sig")
    out["pairs"] = pairs

    ears = ev.most_confident_errors(results, classes, n=20)
    ears.to_csv(os.path.join(tab_dir, f"{pre}tab_nghe_lai.csv"), index=False, encoding="utf-8-sig")

    if get_logmel is None:
        n_mels = next(iter(results.values()))["n_mels"]
        get_logmel = make_logmel_getter(cache_dir, n_mels)
    out["pair_figs"] = (plot_pair_examples(results, pairs, get_logmel, fig_dir, n_pairs_fig,
                                           n_examples, pre, fake) if get_logmel else [])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--meta", default="data/ESC-50-master/meta/esc50.csv")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--exp", type=int, default=None, help="EXP cần phân tích (mặc định: chọn theo validation)")
    ap.add_argument("--exps", default="1,2,3,4,5")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    out = run(a.exp, [int(e) for e in a.exps.split(",")], a.results, a.cache, a.meta,
              a.figures, a.tables, fake=a.fake)
    print(out["pairs"][["Lớp A (thật)", "Lớp B (đoán)", "Số lần nhầm"]].to_string(index=False))


if __name__ == "__main__":
    main()
