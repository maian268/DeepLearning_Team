An: Dữ liệu + thực nghiệm A

Khảo sát số lượng mẫu huấn luyện

Code

- Tự viết hàm Gaussian, sinh dữ liệu và chia train/validation/test.

- Tự xây dựng MLP và vòng lặp huấn luyện bằng PyTorch
  
- Chạy thực nghiệm A với 10 mức dữ liệu từ 200 đến 2.000 mẫu.
  
- Xuất CSV, bảng sai số, biểu đồ và mô hình tốt nhất.
  
Báo cáo: Bài toán, dữ liệu, phương pháp và kết quả A.
Slide: Bài toán, dữ liệu và biểu đồ A.
2. @Quyên : Thực nghiệm B
Khảo sát số nơ-ron tầng ẩn
Code
- Tự sinh dữ liệu, tự viết MLP và quy trình train/validation.
- Chạy mạng một tầng ẩn với 1 đến 10 nơ-ron.
- Đánh giá MSE, MAE/RMSE, lưu loss và checkpoint.
- Vẽ đồ thị sai số theo số nơ-ron, phân tích underfitting/overfitting.
Báo cáo: Cơ sở MLP, phương pháp thực nghiệm B và kết quả B.
Slide: Kiến trúc MLP, kết quả B và nhận xét.
3. @Phương Thanh : Thực nghiệm C
Khảo sát số tầng ẩn
Code
- Tự sinh dữ liệu, xây dựng MLP nhiều tầng và vòng lặp huấn luyện.
- Chạy mạng có 1, 2, 3 và 4 tầng ẩn.
- Đo sai số, thời gian huấn luyện và loss theo epoch.
- Vẽ bề mặt hàm thật, bề mặt xấp xỉ và đồ thị hội tụ.
Báo cáo: Phương pháp, kết quả C và phân tích độ sâu mạng.
Slide: Đồ thị bề mặt, hội tụ và kết quả C.
4. @Tí Hon : Thực nghiệm độ tin cậy và đánh giá tổng hợp
Code
- Tự triển khai đầy đủ quy trình sinh dữ liệu, MLP, huấn luyện và đánh giá.
- Cố định một cấu hình MLP, chạy ít nhất 3 seed để khảo sát độ ổn định.
- Lưu MSE, MAE/RMSE, thời gian chạy và báo cáo trung bình ± độ lệch chuẩn.
- Vẽ biểu đồ \(y\) thật và \(\hat y\), bề mặt xấp xỉ và phân bố sai số.
- Tự viết README và requirements cho phần code của mình.
Báo cáo: Thiết lập thực nghiệm, kết quả độ ổn định, thảo luận và hạn chế.
Slide: Kết quả nhiều seed, đánh giá mô hình và kết luận.
