Dữ liệu và Thực nghiệm A – Ảnh hưởng số mẫu huấn luyện đến MLP
Mô tả

Phần này gồm hai việc: (1) sinh, chia và khảo sát dữ liệu Gaussian hai chiều; (2) Thực nghiệm A khảo sát ảnh hưởng của số mẫu huấn luyện (từ 200 đến 2.000, 10 mức) đến khả năng xấp xỉ hàm mật độ Gaussian hai chiều của mạng MLP viết bằng PyTorch. Toàn bộ mã, giải thích từng bước, bảng sai số và biểu đồ nằm trong BTL01_DuLieu_ThucNghiemA.ipynb.

Thiết lập thực nghiệm
Hàm mục tiêu: Gaussian 2D, với vector kỳ vọng μ = [0, 0] và ma trận hiệp phương sai C = [[0.3, -0.5], [-0.5, 2.0]].
Miền lấy mẫu: x₁ ∈ [-2, 2], x₂ ∈ [-4, 4], nhãn y = f(x₁, x₂) (không nhiễu).
Số mẫu: tập train gốc 2.000, validation 400, test 100. Ba tập sinh từ ba dòng ngẫu nhiên độc lập, kiểm tra không trùng mẫu.
Seed dữ liệu: 2024. Seed huấn luyện (khởi tạo trọng số và xáo trộn batch): 0, 1, 2.
Số mẫu train được khảo sát: A = 200, 400, ..., 2000. Mức A dùng A mẫu đầu tiên của tập train gốc (các tập lồng nhau).
MLP: đầu vào 2 chiều, hai tầng ẩn 32 nơ-ron dùng tanh, đầu ra 1 chiều tuyến tính (2-32-32-1).
Chuẩn hóa: x / (2, 4) và y / f(μ) (hằng số theo miền và theo hàm, không phụ thuộc dữ liệu nên không rò rỉ).
Huấn luyện: batch size 32, tối đa 500 epochs, Adam với learning rate 0.003, hàm mất mát MSE, dừng sớm theo validation (patience 50).
Chọn checkpoint theo validation MSE; sau đó đánh giá trên tập test. Các chỉ số được lưu gồm MSE, RMSE, MAE và R². Mô hình tốt nhất toàn thực nghiệm cũng chọn theo validation, test không dùng để chọn cấu hình.
Cấu trúc thư mục

BTL/

├── README_A.md

├── requirements.txt

├── BTL01_DuLieu_ThucNghiemA.ipynb

└── outputs_A/

    ├── data/
    
    │   ├── train_pool.csv
    │   ├── val.csv
    │   └── test.csv
    ├── logs/
    │   ├── results_A_runs.csv
    │   ├── results_A_summary.csv
    │   ├── results_A_table_pretty.csv
    │   ├── history_A.csv
    │   ├── data_statistics.csv
    │   ├── best_model_test_metrics.json
    │   └── config_A.json
    ├── models/
    │   └── best_model_A.pt
    └── figures/
        ├── fig01_data_scatter3d.png
        ├── fig02_true_surface.png
        ├── fig03_distributions.png
        ├── fig04_mse_vs_A.png
        ├── fig05_rmse_mae_vs_A.png
        ├── fig06_gap_and_time.png
        ├── fig07_learning_curves.png
        ├── fig08_y_vs_yhat_test.png
        └── fig09_surface_approximation.png

Các tệp trong outputs_A/ được tạo khi chạy notebook. requirements.txt cũng do ô cuối của notebook tạo ra (hoặc dùng bản đi kèm).

Cài đặt và chạy

Mở terminal tại thư mục chứa README_A.md, sau đó chạy:

bash
python -m pip install -r requirements.txt
jupyter notebook BTL01_DuLieu_ThucNghiemA.ipynb

Trong Jupyter, chọn Run All. Để kiểm tra nhanh trước khi chạy đầy đủ, đặt QUICK_TEST = True ở ô cấu hình (chỉ 1 seed, 3 mức A, 30 epochs).

Chương trình dùng CPU (mạng nhỏ nên đủ nhanh); đổi DEVICE ở ô cấu hình nếu muốn dùng GPU. Thời gian chạy phụ thuộc vào phần cứng.

Kết quả đầu ra

Sau khi chạy, notebook lưu các kết quả vào outputs_A/:

logs/results_A_runs.csv: kết quả từng lần chạy (mỗi cặp A và seed): số epoch, thời gian, train/validation/test MSE, RMSE, MAE, R².
logs/results_A_summary.csv và logs/results_A_table_pretty.csv: bảng tổng hợp theo A (trung bình ± độ lệch chuẩn qua các seed).
logs/history_A.csv: loss train và validation theo epoch của từng lần chạy.
logs/data_statistics.csv: thống kê mô tả của tập train, validation, test.
logs/best_model_test_metrics.json và models/best_model_A.pt: mô hình tốt nhất (chọn theo validation) và chỉ số test của nó.
logs/config_A.json: toàn bộ cấu hình, seed và phiên bản thư viện để tái lập.
data/*.csv: dữ liệu train gốc, validation, test.
figures/fig01 đến fig03: scatter 3D, bề mặt hàm thật, phân bố dữ liệu.
figures/fig04 đến fig07: MSE theo A, RMSE và MAE theo A, khoảng cách tổng quát hóa và thời gian huấn luyện, đường học.
figures/fig08 và fig09: y thật so với y dự đoán trên test, bề mặt hàm thật so với bề mặt xấp xỉ và bản đồ sai số.

Để chèn một hình vào báo cáo hoặc notebook, thêm một ô Markdown và dùng đường dẫn tương đối từ thư mục dự án, ví dụ:

markdown
![MSE theo A](outputs_A/figures/fig04_mse_vs_A.png)
