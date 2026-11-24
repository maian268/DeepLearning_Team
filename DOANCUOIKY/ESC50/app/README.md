# ESC-50 Audio Classification Web App

Ứng dụng Web tương tác được xây dựng bằng  Streamlit , cho phép người dùng tải lên hoặc chọn các tệp âm thanh môi trường (định dạng `.wav`, thời lượng 5 giây) để dự đoán và phân loại thuộc  50 lớp âm thanh ESC-50  sử dụng các mô hình học sâu  CNN Baseline  và  CRNN (Convolutional Recurrent Neural Network) .

---

## Cấu trúc Thư mục Ứng dụng

```text
ESC50/
├── app/
│   ├── app.py              # Mã nguồn chính của giao diện Web Streamlit (H-04)
│   ├── README.md           # Tệp tài liệu hướng dẫn sử dụng (tệp này)
│   ├── history.csv         # Lịch sử phân loại (tự động tạo khi ứng dụng thực thi)
│   ├── Assets/             # Thư mục chứa các ảnh thông báo lỗi giao diện
│   │   ├── Thông báo sai định dạng.png
│   │   └── Thông báo vượt giới hạn dung lượng tệp.png
│   └── samples/            # Thư mục chứa các tệp âm thanh WAV mẫu để dùng thử
│       ├── 1-100032-A-0.wav
│       ├── 1-100038-A-14.wav
│       └── ...
```

---

## Yêu cầu Môi trường & Cài đặt

Đảm bảo bạn đã cài đặt các thư viện cần thiết trong môi trường Virtualenv hoặc Conda của dự án:

```bash
pip install streamlit torch numpy pandas matplotlib
```

---

## Hướng dẫn Khởi chạy Ứng dụng

 Mở Terminal (PowerShell / Command Prompt / Bash) tại  thư mục gốc của dự án (`ESC50/`)  và thực hiện lệnh:

```bash
streamlit run app/app.py
```
Sau khi chạy lệnh, giao diện ứng dụng sẽ tự động mở trên trình duyệt mặc định tại đường dẫn `http://localhost:8501`.
---

## Các Tính năng Chính & Cấu hình Giao diện

### 1. Cấu hình Mô hình (Sidebar - Thanh điều hướng bên trái)
*  Kiến trúc mô hình (`arch`):  Cho phép lựa chọn giữa  CRNN  (Convolutional Recurrent Neural Network) và  CNN Baseline .
*  Số dải tần Mel (`n_mels`):  Tùy chọn `128` bins hoặc `64` bins.
*  Ngưỡng cảnh báo tin cậy (Low Confidence Threshold):  Con trượt chọn ngưỡng từ `0.10` đến `0.80` (Mặc định:  `0.30`  hay  30% ).
  * *Chức năng:* Nếu nhãn dự đoán Top-1 có xác suất nhỏ hơn ngưỡng này, hệ thống sẽ đưa ra cảnh báo màu vàng về độ tin cậy thấp (âm thanh bị nhiễu hoặc không điển hình).
*  Đường dẫn Checkpoint (`.pt`):  Nhập đường dẫn tệp trọng số (mặc định: `checkpoints/exp5_fold1.pt`).
  * *Chế độ mô phỏng (Dummy Mode):* Nếu tệp `.pt` chưa tồn tại trong thư mục `checkpoints/`, ứng dụng tự động hiển thị thông báo màu vàng và chạy ở chế độ mô phỏng để tránh lỗi crash ứng dụng.

### 2. Chế độ Hỗ trợ Người Mù màu (Colorblind Accessibility Mode)
* Công tắc bật/tắt ở Sidebar giúp chuyển đổi bảng màu hiển thị sang chuẩn  Cividis / Okabe-Ito :
  * Chuyển phổ Log-Mel Spectrogram sang bảng màu khoa học `cividis`.
  * Chuyển biểu đồ sóng âm (Waveform) và cột xác suất Top-1 sang tông màu Cam đất / Cam vàng tương phản cao.
  * Thêm họa tiết sọc chéo (hatching `//`) cho cột Top-1 giúp phân biệt dễ dàng.

### 3. Nhập dữ liệu Âm thanh (Audio Input Tools)
*  Tải tệp từ máy tính:  Chọn file `.wav` dung lượng tối đa  10 MB .
  * Hệ thống tự động kiểm tra lỗi định dạng và dung lượng tệp. Nếu vi phạm, giao diện hiển thị ảnh thông báo trực quan kèm nút tương tác tải lại tệp.
*  Tệp âm thanh mẫu (Samples):  Chọn trực tiếp các mẫu âm thanh có sẵn từ thư mục `app/samples/`.

### 4. Trực quan hóa Dữ liệu Âm thanh (Audio Visualization)
*  Audio Player:  Trình phát nghe lại âm thanh trực tiếp trên web.
*  Biểu đồ Sóng âm (Waveform):  Hiển thị biên độ âm thanh theo thời gian chuẩn hóa.
*  Biểu đồ Phổ Log-Mel (Log-Mel Spectrogram):  Biến đổi tín hiệu âm thanh dạng sóng thành dạng phổ tần số Mel trực quan.

### 5. Kết quả Dự đoán & Lịch sử Phân loại
*  Dự đoán Top-1:  Hiển thị tên nhãn phân loại âm thanh có xác suất cao nhất kèm chỉ số % độ tin cậy.
*  Biểu đồ Top-5:  Biểu đồ cột nằm ngang thể hiện Top-5 nhãn có xác suất cao nhất.
*  Lịch sử Phân loại (History Log):  Bảng ghi nhận tất cả các lượt phân loại trong phiên làm việc.
  * Hỗ trợ xuất dữ liệu ra tệp  CSV (`esc50_history_*.csv`) .
  * Hỗ trợ nút xóa toàn bộ lịch sử kèm hộp thoại xác nhận an toàn.

---
