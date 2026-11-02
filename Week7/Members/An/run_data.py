"""Sinh dữ liệu, kiểm tra hàm Gaussian, lưu CSV, thống kê và vẽ hình dữ liệu.
Chạy:  python run_data.py"""
from src import config as cfg
from src import data, plots

if __name__ == "__main__":
    cfg.ensure_dirs()
    print("Hàm Gaussian tự viết khớp SciPy:", data.verify_gaussian())
    ds = data.build_datasets(save=True)
    stats = data.data_statistics(ds)
    print(stats.round(5).to_string(index=False))
    plots.plot_data(ds)
    plots.plot_architecture(cfg.A_HIDDEN, "fig_architecture.png")
    print(f"Đã lưu CSV vào {cfg.DATA_DIR}, hình vào {cfg.FIG_DIR}")
