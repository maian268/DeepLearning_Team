# BTL03 — P2: Phân loại Giới tính & Phát hiện Đối tượng
## Thông tin chung
| Thông tin | Chi tiết |
|---|---|
| **Phụ trách** | Quyên |
| **Framework** | PyTorch |
| **Phiên bản PyTorch** | 2.14.1+cpu
| **Ngày hoàn thành** | 27/11/2026 |

---

## Cấu trúc thư mục
```bash
P2_Gender_ObjDetection/
├── README_P2.md
├── report_P2.md
├── 2_Human_Gender_CNN_pytorch.ipynb
├── 4.2_Human_Gender_CNN_pytorch.ipynb
├── 5.2_obj_detection_pytorch.ipynb
├── checkpoints/          (tự tạo khi chạy, không nộp)
├── runs/                  (TensorBoard logs, không nộp)
└── data/                  
    ├── gender/
    │   ├── train_x_onehot.npy
    │   ├── train_y_onehot.npy
    │   ├── val_x_onehot.npy
    │   └── val_y_onehot.npy
    └── obj_detection/
    │   ├── vehicles/
    │   └── non-vehicles/
    │    
    └── test.jpg
```
---

## Hướng dẫn tải dữ liệu

### 1. Dữ liệu Giới tính (Gender)
- Tải 4 tệp `.npy` từ: https://drive.google.com/drive/folders/0BxINLo5jshCRYW8xODhSklTQTE0?usp=sharing
- Đặt vào thư mục: `data/gender/`
- Thông tin: ảnh 64×64, 1 kênh xám; nhãn one-hot: [1,0]=Nữ, [0,1]=Nam

### 2. Dữ liệu Phát hiện xe
- Tải `car_data.zip` (~125MB) từ: https://drive.google.com/file/d/1rjUitFrFhB84FVFig1Ad1Vz6CoVPnZ/view?usp=sharing
- Giải nén vào: `data/obj_detection/`
- Kết quả giải nén phải có 2 thư mục: `vehicles/` và `non-vehicles/`
- test.jpg

---

## Cài đặt môi trường
### Trên Colab
```bash
!pip install torchinfo
```
### Trên máy cá nhân
```bash
pip install torch torchvision numpy pillow scikit-learn matplotlib torchinfo
```

---

## Danh sách Notebook & Mục tiêu

| STT | Tên Notebook | Mô tả | Độ chính xác mong đợi |
|-----|-------------|-------|----------------------|
| 1 | 2_Human_Gender_CNN_pytorch.ipynb | CNN cơ bản phân loại giới tính | val_acc ≥ 0.826 |
| 2 | 4.2_Human_Gender_CNN_pytorch.ipynb | CNN + EarlyStopping, ReduceLROnPlateau, TensorBoard | val_acc ≥ 0.877 (epoch ~27) |
| 3 | 5.2_obj_detection_pytorch.ipynb | CNN phân loại có xe / không xe | train_acc ≥ 0.99, tổng tham số = 83,367 |

---

## Cách chạy từng Notebook

### Notebook 1 — 2_Human_Gender_CNN_pytorch.ipynb
1. Đặt 4 tệp `.npy` vào `data/gender/`
2. Chạy toàn bộ notebook từ trên xuống
3. Kết quả mong đợi: val_acc ≥ 0.826 sau khoảng 4 epoch

### Notebook 2 — 4.2_Human_Gender_CNN_pytorch.ipynb
1. Đảm bảo dữ liệu giới tính đã sẵn
2. Chạy toàn bộ notebook
3. EarlyStopping tự động dừng khi val_loss không cải thiện sau 5 epoch
4. Checkpoint tốt nhất được lưu tại: `checkpoints/best_model.pth`
5. (Tùy chọn) Xem biểu đồ huấn luyện với TensorBoard:
```bash
%load_ext tensorboard
%tensorboard --logdir=runs/
```

6. Kết quả mong đợi: val_acc ≥ 0.877 tại khoảng epoch 27

### Notebook 3 — 5.2_obj_detection_pytorch.ipynb
1. Giải nén `car_data.zip` vào `data/obj_detection/`
2. Kiểm tra đường dẫn trong phần "Đọc dữ liệu"
3. Chạy toàn bộ notebook
4. Kết quả mong đợi:
   - Tổng số tham số mô hình: 83,367
   - Train accuracy ≥ 0.99 sau 5 epoch
   - Test accuracy ≥ 0.98

---

## Lưu ý quan trọng
- Seed cố định: `torch.manual_seed(42)`, `np.random.seed(42)`
- Đổi thứ tự kênh: (N,H,W,C) → (N,C,H,W) khi chuyển sang PyTorch
- Không thêm Softmax ở lớp cuối khi dùng CrossEntropyLoss()
- Tính chính xác số đặc trưng trước lớp Linear sau khi Flatten()

---

## Báo cáo chi tiết
Xem file `report_P2.md` bao gồm:
1. Mục tiêu & mô tả dữ liệu
2. Bảng so sánh kiến trúc Keras ↔ PyTorch
3. Điểm khác biệt khi chuyển đổi
4. Thiết lập huấn luyện
5. Kết quả & biểu đồ
6. Nhận xét & khó khăn

---

**Người thực hiện**: Quyên — P2 BTL03