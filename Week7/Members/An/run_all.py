"""Chạy toàn bộ: dữ liệu -> nhóm A, B, C -> đánh giá cuối trên test -> (tùy chọn) dựng báo cáo + slide.
  python run_all.py              # chạy đầy đủ
  python run_all.py --quick      # chạy thử nhanh để kiểm tra pipeline
  python run_all.py --build      # chạy xong thì dựng luôn báo cáo + slide
"""
import argparse
import subprocess
import sys
from src import config as cfg
from src import data, experiments, plots
from src.train import get_device

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--build", action="store_true", help="dựng báo cáo (.docx/.pdf) và slide (.pptx) sau khi chạy")
    a = ap.parse_args()

    cfg.ensure_dirs()
    device = get_device(a.gpu)
    print("Thiết bị:", device)
    print("Gaussian tự viết khớp SciPy:", data.verify_gaussian())
    ds = data.build_datasets(save=True)
    data.data_statistics(ds)
    plots.plot_data(ds)
    plots.plot_architecture(cfg.A_HIDDEN, "fig_architecture.png")
    experiments.dump_config(a.quick)
    for g in ["A", "B", "C"]:
        experiments.run_group(g, ds, quick=a.quick, seeds=a.seeds, device=device)
    experiments.final_evaluation(ds, device)
    if a.build:
        subprocess.run([sys.executable, "build_report.py"], check=False)
        subprocess.run([sys.executable, "build_slides.py"], check=False)
