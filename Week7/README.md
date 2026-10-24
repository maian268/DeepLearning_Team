1. Bài nhóm yêu cầu xây dựng một mạng nơ-ron MLP bằng PyTorch để xấp xỉ hàm Gaussian hai chiều, sau đó thực nghiệm để tìm hiểu xem số lượng dữ liệu, số nơ-ron và số tầng ẩn ảnh hưởng thế nào đến độ chính xác của mô hình.
2. Nhóm phải viết code cho mạng chạy được, thiết kế thí nghiệm, thu thập kết quả, giải thích kết quả và trình bày thành báo cáo khoa học.
3. Năm đầu việc chính:
   
  3.1 Sinh và khảo sát dữ liệu
- Viết hàm Gaussian đúng công thức.
- Tạo train, validation và test độc lập.
- Dùng seed để tái lập kết quả.
- Vẽ dữ liệu 3D, bề mặt hàm thật và thống kê phân bố.
  
  3.2. Xây dựng và huấn luyện MLP
- Tự viết mô hình bằng torch.nn.Module, nn.Linear.
- Sử dụng Dataset/DataLoader, MSELoss và optimizer.
- Tự cài đặt vòng lặp huấn luyện, validation và lưu mô hình tốt nhất.
- Đánh giá bằng MSE và MAE hoặc RMSE.
  
  3.3. Chạy ba nhóm thực nghiệm
- A: Thay đổi số mẫu huấn luyện từ 200 đến 2.000.
- B: Thay đổi số nơ-ron của mạng một tầng ẩn, từ 1 đến 10.
- C: Thay đổi số tầng ẩn từ 1 đến 4.
Mỗi nhóm cần có bảng kết quả, biểu đồ và nhận xét.

  3.4. Viết báo cáo khoa học
Trình bày dữ liệu, kiến trúc MLP, quy trình huấn luyện, kết quả thực nghiệm, phân tích và kết luận. Báo cáo đề xuất dài 10–20 trang nội dung chính, nộp Word và PDF.

  3.5. Chuẩn bị sản phẩm nộp và thuyết trình
Slide khoảng 10–15 trang, có đồ thị và phân tích. Nộp kèm mã nguồn, README, requirements.txt, CSV/log, hình ảnh và trọng số mô hình nếu phù hợp.

LƯU Ý:
- Không được dùng TensorFlow/Keras thay cho việc tự triển khai MLP bằng PyTorch.
- Không dùng tập test để lựa chọn siêu tham số. Validation dùng để chọn mô hình, test dùng để đánh giá cuối cùng.
- Các thí nghiệm phải công bằng: mỗi nhóm chỉ thay đổi yếu tố đang khảo sát, những yếu tố khác giữ ổn định.
Phải lưu kết quả thực tế, không chỉ chụp hình biểu đồ. Thầy khuyến khích chạy ít nhất 3 seed và báo cáo trung bình ± độ lệch chuẩn.
- Phải phân tích vì sao mô hình tốt hoặc kém, bao gồm hiện tượng thiếu khớp (underfitting) và quá khớp (overfitting), chứ không chỉ liệt kê MSE.
