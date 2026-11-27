# 🎵 ESC-50 Audio Classification Web App

Ứng dụng web tương tác xây dựng bằng **Streamlit** phục vụ bài toán phân loại **50 lớp âm thanh môi trường (ESC-50)** sử dụng các mô hình học sâu **CNN** và **CRNN**.


## ✨ Tính năng chính

1. **Đa dạng nguồn nhập âm thanh (`.wav`)**
   - **Tải tệp từ máy tính:** Hỗ trợ tệp định dạng `.wav` (khuyến nghị 5 giây, dung lượng tối đa **10 MB**).
   - **Chọn tệp mẫu cố định:** Danh sách các tệp âm thanh `.wav` có sẵn trong thư mục `app/samples/` để kiểm thử nhanh.
   - **Ghi âm trực tiếp từ Micro:** Thu âm trực tiếp trên trình duyệt và tự động nhận diện ngay sau khi dừng ghi.

2. **Kiểm tra dữ liệu đầu vào & Cảnh báo trực quan**
   - Tự động kiểm tra định dạng tệp (`.wav`) và giới hạn dung lượng (`<= 10 MB`).
   - Hiển thị thông báo cảnh báo bằng hình ảnh tương tác (cho phép bấm đóng hoặc chọn lại tệp trực tiếp) khi phát hiện tệp không hợp lệ.

3. **Cấu hình mô hình linh hoạt (Sidebar)**
   - **Kiến trúc mô hình (`arch`):** Chuyển đổi giữa **CRNN** (*Convolutional Recurrent Neural Network*) và **CNN Baseline**.
   - **Số dải tần Mel (`n_mels`):** Hỗ trợ **128** hoặc **64** Mel bins.
   - **Tự động đồng bộ cấu hình từ Checkpoint:** Khi nhập đường dẫn tệp trọng số theo quy ước thí nghiệm (`exp1`–`exp6`), ứng dụng tự động nhận diện kiến trúc (`cnn`/`crnn`) và số dải Mel (`64`/`128`) tương ứng.
   - **Ngưỡng cảnh báo tin cậy:** Thanh trượt điều chỉnh ngưỡng từ `10%` đến `80%` (mặc định `30%`); đưa ra cảnh báo nếu xác suất Top-1 thấp hơn ngưỡng.
   - **Chế độ hỗ trợ người mù màu (Colorblind Mode):** Chuyển đổi bảng màu biểu đồ sang chuẩn khoa học **Cividis / Okabe-Ito** độ tương phản cao kèm họa tiết phân biệt cột Top-1.

4. **Trực quan hóa tín hiệu & Kết quả suy luận**
   - Trình phát âm thanh trực tiếp trên giao diện.
   - Biểu đồ **Sóng âm (Waveform)** theo thời gian thực (tự động chuyển stereo về mono).
   - Biểu đồ **Log-Mel Spectrogram** tương ứng với số dải Mel đã chọn.
   - Hiển thị nhãn dự đoán **Top-1**, biểu đồ cột ngang và bảng xác suất chi tiết **Top-5**.

5. **Lưu trữ & Xuất lịch sử phân loại**
   - Tự động lưu lịch sử các lần dự đoán vào tệp `app/history.csv` (duy trì dữ liệu qua các phiên làm việc và chống ghi trùng khi tải lại trang).
   - Hỗ trợ tải xuống báo cáo lịch sử định dạng `.CSV` (mã hóa `UTF-8-SIG` tương thích tốt với Excel) hoặc xóa toàn bộ lịch sử có hộp thoại xác nhận.

---

## 📁 Cấu trúc thư mục `app/`

```text
ESC50/
├── app/
│   ├── Assets/
│   │   ├── Thông báo sai định dạng.png
│   │   └── Thông báo vượt giới hạn dung lượng tệp.png
│   ├── samples/                # Thư mục chứa các tệp .wav mẫu để thử nghiệm
│   ├── app.py                  # Mã nguồn chính của ứng dụng Streamlit
│   ├── history.csv             # Tệp lưu lịch sử phân loại (tự động tạo khi chạy)
│   └── README.md               # Tài liệu hướng dẫn sử dụng
├── checkpoints/                # Thư mục chứa các tệp trọng số mô hình (.pt)
└── src/
    ├── data_lib.py             # Hàm tiền xử lý wav_to_logmel
    └── infer.py                # Hàm suy luận predict()
