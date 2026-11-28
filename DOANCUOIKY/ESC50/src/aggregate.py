"""aggregate.py - A-04: tổng hợp mean ± std qua 5 fold, kiểm định cặp, bảng ablation.

Đầu vào : results/exp{id}_fold{k}.json
Đầu ra  : tables/tab_results.csv, tables/tab_rq.csv, tables/tab_gap_train_val.csv,
          tables/tab_cost.csv, figures/fig6_ablation_bar.png

Chạy:  python -m src.aggregate --results results --tables tables --figures figures
"""
import os
import argparse

import numpy as np

from src import eval_lib as ev

BAR_COLORS = ["#90A4AE", "#4DB6AC", "#7986CB", "#FFB74D", "#E57373", "#A1887F"]


def plot_ablation_bar(df, path, fake=False):
    """Biểu đồ cột Accuracy và Macro-F1 (mean ± std qua fold) cho các EXP."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(df))
    for ax, (m, s, title) in zip(axes, [("acc_mean", "acc_std", "Accuracy (%)"),
                                         ("f1_mean", "f1_std", "Macro-F1 (%)")]):
        err = df[s].fillna(0)
        ax.bar(x, df[m], yerr=err, capsize=4, color=BAR_COLORS[:len(df)])
        ax.set_xticks(x)
        ax.set_xticklabels(df["EXP"])
        ax.set_ylabel(title)
        ax.set_ylim(0, float((df[m] + err).max()) * 1.25)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(df[m]):
            ax.text(i, v + err.iloc[i] + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    fig.suptitle("So sánh các thí nghiệm (mean ± std qua các fold)")
    if fake:
        fig.text(0.5, 0.5, "DỮ LIỆU GIẢ - CHỈ ĐỂ TEST", fontsize=30, color="red",
                 alpha=0.25, ha="center", va="center", rotation=25)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def aggregate(exp_ids=(1, 2, 3, 4, 5), result_dir="results", tab_dir="tables",
              fig_dir="figures", metric=1, fake=False):
    """Tạo các bảng và biểu đồ tổng hợp. metric: 1 = Macro-F1 (mặc định), 0 = Accuracy.

    fake=True: thêm tiền tố 'fake_' và chữ đỏ trên hình để không lẫn với kết quả thật.
    """
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    pre = "fake_" if fake else ""

    df, per_fold = ev.summarize(list(exp_ids), result_dir)
    if df.empty:
        raise FileNotFoundError(f"Không thấy file kết quả nào trong {result_dir}")

    shown = df[["EXP", "Số fold", "Accuracy (%)", "Macro-F1 (%)"]]
    rq = ev.rq_table(per_fold, metric=metric)
    gap = ev.gap_table(list(df["EXP"].str.replace("EXP", "").astype(int)), result_dir)
    cost = ev.cost_table(list(df["EXP"].str.replace("EXP", "").astype(int)), result_dir)

    out = {}
    for name, table in (("tab_results", shown), ("tab_rq", rq),
                        ("tab_gap_train_val", gap), ("tab_cost", cost)):
        path = os.path.join(tab_dir, f"{pre}{name}.csv")
        table.to_csv(path, index=False, encoding="utf-8-sig")
        out[name] = path
    out["fig6_ablation_bar"] = plot_ablation_bar(
        df, os.path.join(fig_dir, f"{pre}fig6_ablation_bar.png"), fake=fake)

    incomplete = df[df["Số fold"] < 5]["EXP"].tolist()
    if incomplete:
        print("CẢNH BÁO: chưa đủ 5 fold cho", ", ".join(incomplete), "- kết quả chỉ tạm thời.")
    return {"tables": {"results": shown, "rq": rq, "gap": gap, "cost": cost}, "files": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--exps", default="1,2,3,4,5", help="danh sách EXP, ví dụ 1,2,3,4,5,6")
    ap.add_argument("--metric", type=int, default=1, help="1 = Macro-F1, 0 = Accuracy")
    ap.add_argument("--fake", action="store_true", help="dữ liệu giả: thêm tiền tố fake_")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    res = aggregate([int(e) for e in a.exps.split(",")], a.results, a.tables,
                    a.figures, a.metric, a.fake)
    print(res["tables"]["results"].to_string(index=False))
    print()
    print(res["tables"]["rq"].to_string(index=False))
    print("\nĐã lưu:", *res["files"].values(), sep="\n  ")


if __name__ == "__main__":
    main()
