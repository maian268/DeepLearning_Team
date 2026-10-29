## XẤP XỈ HÀM SỐ BẰNG MẠNG MLP (THỰC NGHIỆM C)

Dự án triển khai mô hình MLP bằng PyTorch để xấp xỉ hàm mật độ Gaussian 2 chiều, tập trung khảo sát ảnh hưởng của số tầng ẩn $C \in \{1, 2, 3, 4\}$.

## 1. Cài đặt môi trường
Cài đặt các gói thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 2. Hướng dẫn chạy thực nghiệm
Thực thi lệnh sau để tự động sinh dữ liệu, huấn luyện mô hình, lưu bảng kết quả và xuất toàn bộ hình ảnh biểu đồ:
```bash
python run_experiment_c.py
```

## 3. Cấu trúc thư mục
- `run_experiment_c.py`: Mã nguồn huấn luyện, validation, lưu checkpoint và vẽ đồ thị.
- `requirements.txt`: Danh sách các thư viện phụ thuộc (`torch`, `numpy`, `matplotlib`, `pandas`, `scipy`).
- `results_experiment_c.csv`: Bảng tổng hợp các chỉ số đánh giá (Mean ± Std).
- `checkpoints/`: Chứa file trọng số mô hình tối ưu `best_model_c.pt`.
- `figures/`: Chứa ảnh biểu đồ hội tụ, sai số (`experiment_c_curves.png`) và bề mặt 3D (`surface_comparison.png`).
- Báo cáo và Slide: Tài liệu báo cáo (`.pdf`, `.tex`) cùng slide thuyết trình (`.pptx`, `.pdf`).

## 4. Tóm tắt kết quả
- **Cấu hình tối ưu:** Mô hình đạt kết quả tốt nhất tại $C = 2$ (Test MSE thấp nhất, cân bằng giữa tốc độ hội tụ và độ chính xác xấp xỉ).
- **Phân tích độ sâu:** Tăng lên $C = 3, 4$ không làm giảm thêm sai số do dư thừa tham số đối với hàm trơn 2D, đồng thời làm tăng chi phí thời gian huấn luyện.
