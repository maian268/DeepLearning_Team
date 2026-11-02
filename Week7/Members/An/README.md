# BTL01 – Xấp xỉ hàm Gaussian 2D bằng MLP (PyTorch)

Môn **Học sâu** · Bài tập lớn 01 · Nhóm: *[điền tên nhóm]*

Dự án xây dựng mạng MLP bằng **PyTorch** để xấp xỉ hàm mật độ Gaussian hai chiều
`f(x) = exp(-½ (x-μ)ᵀ C⁻¹ (x-μ)) / (2π√det C)` với `μ = (0,0)`, `C = [[0.3, -0.5], [-0.5, 2]]`,
trong miền `x₁ ∈ [-2, 2]`, `x₂ ∈ [-4, 4]`, và khảo sát 3 yếu tố:

| Nhóm | Yếu tố thay đổi | Giá trị |
|---|---|---|
| **A** | Số mẫu train | 200, 400, …, 2000 (10 mức), kiến trúc 2-32-32-1 |
| **B** | Số nơ-ron, MLP 1 tầng ẩn | 1…10 (mở rộng: 15, 20, 30, 50), 1000 mẫu train |
| **C** | Số tầng ẩn | 1…5 tầng, mỗi tầng 16 nơ-ron, 1000 mẫu train |

## 1. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Yêu cầu Python ≥ 3.9. Chạy được trên CPU (GPU tùy chọn bằng cờ `--gpu`). Không phụ thuộc đường dẫn cá nhân.

## 2. Lệnh chạy

```bash
python run_data.py                     # sinh dữ liệu, kiểm tra hàm Gaussian, thống kê, hình dữ liệu
python run_experiment.py --group A     # thực nghiệm A  (kết quả: logs/results_A_*.csv, figures/fig_A_*.png)
python run_experiment.py --group B     # thực nghiệm B
python run_experiment.py --group C     # thực nghiệm C
python run_all.py                      # chạy toàn bộ + chọn mô hình tốt nhất + đánh giá cuối trên test
python run_all.py --quick              # chạy thử nhanh (1 seed, 30 epoch) để kiểm tra pipeline
python run_experiment.py --group A --seeds 0 1 2 3 4   # đổi danh sách seed
python build_report.py                 # dựng report/BaoCao_BTL01.docx và .pdf từ log/hình
python build_slides.py                 # dựng slides/Slide_BTL01.pptx từ log/hình
python run_all.py --build              # chạy toàn bộ rồi dựng luôn báo cáo + slide
```

Thứ tự đề xuất: `run_all.py` → `build_report.py` → `build_slides.py`.
Notebook `notebooks/BTL01_DuLieu_ThucNghiemA.ipynb` chạy riêng phần dữ liệu + thực nghiệm A (kèm giải thích từng bước).

## 3. Cấu trúc thư mục

```
BTL01_Nhom_HoTen_MSSV/
├── README.md, requirements.txt
├── src/
│   ├── config.py        # toàn bộ siêu tham số, seed, danh sách giá trị A/B/C (chỉnh ở đây)
│   ├── data.py          # hàm Gaussian, sinh/chia/lưu dữ liệu, thống kê
│   ├── model.py         # MLP(nn.Module) tùy biến số tầng/nơ-ron
│   ├── train.py         # DataLoader, vòng lặp train/validation, checkpoint, dự đoán, chỉ số
│   ├── experiments.py   # chạy nhóm A/B/C, ghi log, bảng tổng hợp, mô hình cuối
│   ├── plots.py         # biểu đồ (dữ liệu, kiến trúc, A/B/C, bề mặt xấp xỉ)
│   └── analysis.py      # sinh nhận xét mô tả số liệu cho báo cáo và slide
├── run_data.py, run_experiment.py, run_all.py
├── build_report.py, build_slides.py
├── notebooks/           # notebook phần dữ liệu + thực nghiệm A
├── data/                # train_pool.csv, val.csv, test.csv
├── logs/                # results_*_runs.csv, results_*_summary.csv, history_*.csv, config.json, ...
├── figures/             # toàn bộ hình (.png)
├── models/              # best_A.pt, best_B.pt, best_C.pt, best_model.pt
├── report/              # BaoCao_BTL01.docx / .pdf
└── slides/              # Slide_BTL01.pptx
```

## 4. Thiết lập thực nghiệm (xem `src/config.py`)

- **Dữ liệu:** seed `2024`; train gốc 2000 / validation 400 / test 100, sinh từ 3 dòng ngẫu nhiên độc lập
  (`SeedSequence.spawn`), kiểm tra không trùng mẫu. Mức A dùng A mẫu đầu của train gốc.
- **Chuẩn hóa:** `x/(2,4)`, `y/f(μ)` – hằng số theo miền/hàm, không ước lượng từ dữ liệu (không rò rỉ).
- **Huấn luyện:** Adam, lr `3e-3`, batch `32`, tối đa `500` epoch, early stopping theo validation (patience `50`,
  cải thiện tối thiểu 0,1%), hàm kích hoạt `tanh`, mất mát `MSELoss`, seed huấn luyện `0, 1, 2`.
- **Đánh giá:** MSE, RMSE, MAE, R² trên thang đo gốc; báo cáo trung bình ± độ lệch chuẩn theo seed.
  Validation dùng để dừng sớm/chọn checkpoint/chọn mô hình; **test không dùng để chọn cấu hình**.

## 5. Đầu ra

- `data/*.csv`, `logs/results_{A,B,C}_runs.csv` (từng lần chạy), `logs/results_*_summary.csv` (mean ± std),
  `logs/history_*.csv` (loss theo epoch), `logs/config.json`, `logs/final_test_metrics.json`.
- `figures/`: hình dữ liệu, `fig_{A,B,C}_{mse,rmse_mae,curves,time}.png`, `fig_final_*.png`.
- `models/best_{A,B,C}.pt` (tốt nhất mỗi nhóm theo validation) và `models/best_model.pt` (tốt nhất tổng thể).

Nạp lại mô hình:
```python
import torch
from src.model import MLP
ck = torch.load("models/best_model.pt", map_location="cpu")
m = MLP(2, ck["hidden_sizes"], 1, ck["activation"]); m.load_state_dict(ck["state_dict"]); m.eval()
```

## 6. Thành viên, nhiệm vụ, tỷ lệ đóng góp

| Họ tên | MSSV | Nhiệm vụ | Đóng góp (%) |
|---|---|---|---|
| *[điền]* | *[điền]* | *[điền]* | *[điền]* |
| *[điền]* | *[điền]* | *[điền]* | *[điền]* |
| *[điền]* | *[điền]* | *[điền]* | *[điền]* |
| | | **Tổng** | **100** |

## 7. Lưu ý khi nộp

- Chạy `python run_all.py` để sinh log/hình/mô hình thật, rồi `build_report.py`, `build_slides.py`.
- Sau khi dựng, mở báo cáo và slide để điền các ô nền vàng (tên nhóm, thành viên…) và đọc lại phần nhận xét.
- Đổi tên thư mục/zip theo quy định: `BTL01_Nhom_HoTen_MSSV.zip`.
