"""Chạy một nhóm thực nghiệm.
  python run_experiment.py --group A            # số mẫu train A = 200..2000
  python run_experiment.py --group B            # số nơ-ron tầng ẩn B (1 tầng ẩn)
  python run_experiment.py --group C            # số tầng ẩn C
  python run_experiment.py --group A --quick    # chạy thử nhanh (ít seed, ít epoch)
  python run_experiment.py --group A --seeds 0 1 2 3 4
"""
import argparse
from src import data, experiments

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["A", "B", "C"])
    ap.add_argument("--quick", action="store_true", help="chạy thử nhanh")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="danh sách seed (mặc định theo src/config.py)")
    ap.add_argument("--gpu", action="store_true", help="dùng GPU nếu có")
    a = ap.parse_args()

    from src.train import get_device
    ds = data.build_datasets(save=True)
    experiments.dump_config(a.quick)
    experiments.run_group(a.group, ds, quick=a.quick, seeds=a.seeds, device=get_device(a.gpu))
