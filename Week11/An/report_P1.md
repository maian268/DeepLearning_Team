# BTL03 – Chuyển code Keras sang PyTorch · Phần P1

**Người thực hiện:** Mai Thị Thúy An – MSSV 3123580001
**Phạm vi:** `getting_started`, `Iris_Bayesian_decision_making`, `Func_apxm_MLP`, `MNIST_MLP`, `1_MNIST_CNN`
**Môi trường:** Google Colab, GPU Tesla T4, PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, seed = 42

> Chỗ có dấu … là số liệu hoặc nhận xét cần điền sau khi chạy notebook. Dán ảnh đồ thị từ thư mục `BTL03/P1` trên Drive.

---

## 1. getting_started

**1. Mục tiêu và dữ liệu.** Kiểm tra phiên bản framework và GPU. Không dùng dữ liệu.
**2. Kiến trúc.** Không có mô hình.
**3. Điểm khác biệt khi chuyển.** `tf.__version__` → `torch.__version__`; `device_lib.list_local_devices()` → `torch.cuda.is_available()` và `torch.cuda.get_device_name()`.
**4. Thiết lập huấn luyện.** Không áp dụng.
**5. Kết quả.** PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA available: True, GPU: Tesla T4.
**6. Nhận xét.** …

## 2. Iris_Bayesian_decision_making

**1. Mục tiêu và dữ liệu.** Phân loại 3 lớp hoa Iris (150 mẫu, 4 đặc trưng) bằng quyết định Bayes. Tách ngẫu nhiên 60% train (90 mẫu) và 40% test (60 mẫu), seed 42.
**2. Kiến trúc.** Không có mạng nơ-ron. Hai bộ phân loại: Naive Bayes Gaussian (phương sai đường chéo) và Bayes Gaussian đầy đủ ma trận hiệp phương sai (`MultivariateNormal`), cùng so với `GaussianNB` của sklearn.
**3. Điểm khác biệt khi chuyển.** Bản gốc không dùng Keras. `np.mean/np.cov` → `tensor.mean/torch.cov`. `GaussianNB` dùng phương sai MLE cộng `var_smoothing`, bản PyTorch làm đúng như vậy nên dự đoán trùng sklearn. Bản gốc có tính `sigma` đầy đủ nhưng `GaussianNB` không dùng đến, nên bản full covariance có thể cho kết quả khác.
**4. Thiết lập huấn luyện.** Không có vòng huấn luyện (ước lượng tham số dạng đóng).
**5. Kết quả.** sklearn: … · PyTorch GaussianNB: … · PyTorch full covariance: … · Trùng sklearn (dự đoán, mu, phương sai): …
**6. Nhận xét.** …

## 3. Func_apxm_MLP

**1. Mục tiêu và dữ liệu.** Xấp xỉ hàm Gaussian 2D bằng MLP. Train: lưới 32×32 = 1024 điểm; test: lưới 10×10 = 100 điểm trên [-3, 3]². Tự sinh bằng numpy/scipy.
**2. Kiến trúc.** `Linear(2, B)` → sigmoid → `Linear(B, 1)` → sigmoid. Với B = 16 có 65 tham số. Bảng đối chiếu: `Dense(B, sigmoid)` ↔ `nn.Linear + torch.sigmoid`.
**3. Điểm khác biệt khi chuyển.** `compile + fit` → vòng lặp viết tay (xáo trộn bằng `randperm`, batch 128). `evaluate` → `nn.MSELoss`. Khởi tạo đặt lại thành Glorot-uniform và bias 0 cho giống Keras. Bản gốc chọn mẫu bằng `range(1, 1001)` nên chỉ số 0 không bao giờ được chọn; giữ nguyên.
**4. Thiết lập huấn luyện.** MSE, SGD lr 0.01, batch 128. Thí nghiệm A: B = 16, C = 100, A = 100…1000 (bước 100). Thí nghiệm B: A = 800, C = 100, B = 8…512 (nhân đôi). Thí nghiệm C: A = 800, B = 128, C = 10…190 (bước 10). Chạy trên CPU. Thời gian: …
**5. Kết quả.** [Dán 3 đồ thị `func_apxm_A.png`, `func_apxm_B.png`, `func_apxm_C.png`.]
**6. Nhận xét.** Thí nghiệm A: … · Thí nghiệm B: … · Thí nghiệm C: …

## 4. MNIST_MLP

**1. Mục tiêu và dữ liệu.** Nhận dạng chữ số viết tay MNIST (60000 train, 10000 test, ảnh 28×28, 10 lớp). Chia 255, nhãn dùng số nguyên.
**2. Kiến trúc.** Flatten (784) → Linear 512 + ReLU + Dropout 0.2 → Linear 512 + ReLU + Dropout 0.2 → Linear 10. Tổng tham số: 669,706. Bảng đối chiếu: `Dense(512, relu)` ↔ `nn.Linear(…, 512) + torch.relu`; `Dropout(0.2)` ↔ `nn.Dropout(0.2)`.
**3. Điểm khác biệt khi chuyển.** `categorical_crossentropy` + softmax + one-hot → `CrossEntropyLoss` trên logits với nhãn số. `fit/evaluate` → vòng lặp viết tay, `model.train()/eval()`. Lấy đặc trưng fc2 qua `forward(return_features=True)` thay cho `Model(inputs, outputs=fc2)`.
**4. Thiết lập huấn luyện.** SGD lr 0.01 (không momentum), batch 128, 30 epoch, GPU T4. Thời gian train: …
**5. Kết quả.** Test loss: … · Test accuracy: … (Keras tham chiếu ≈ 0.9657). [Dán `mnist_mlp_curves.png`.]
**6. Nhận xét.** …

## 5. 1_MNIST_CNN

**1. Mục tiêu và dữ liệu.** Như trên, dùng CNN. Ảnh dạng (1, 28, 28).
**2. Kiến trúc.** 3 khối [Conv 3×3 (64/128/256, padding 1) → ReLU → MaxPool 2 → Dropout 0.2] → Flatten (256×3×3 = 2304) → Linear 512 + Dropout → Linear 512 + Dropout → Linear 10. Tổng tham số: 1,817,610.
**3. Điểm khác biệt khi chuyển.** NHWC → NCHW; `padding='same'` → `padding=1`; phải khai báo `in_channels` và `in_features`. `GlobalAveragePooling2D` → `mean(dim=(2, 3))`. Lấy đặc trưng pool3 và fc2 qua `forward(return_feats=True)` thay cho `get_layer()`. Cell `model.get_layer()` ở bản gốc gọi thiếu tham số nên bỏ.
**4. Thiết lập huấn luyện.** SGD lr 0.01, batch 128, 30 epoch, GPU T4. Thời gian train: … (Keras ghi 4.22 phút trên GTX 1080).
**5. Kết quả.** Test loss: … · Test accuracy: … (Keras tham chiếu ≈ 0.9889). [Dán `mnist_cnn_curves.png`, `mnist_pca_raw.png`, `mnist_pca_fc2.png`, `mnist_pca_pool3.png`.]
**6. Nhận xét.** So sánh với MLP: … · So sánh 3 biểu đồ PCA (ảnh gốc, fc2, pool3): …
