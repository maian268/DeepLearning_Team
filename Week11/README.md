# BTL03 – Phần P1: Nhập môn + MLP + MNIST CNN (Keras → PyTorch)

Người thực hiện: Mai Thị Thúy An – 3123580001

## Nội dung

Notebook gốc (Keras)	Notebook PyTorch

getting_started	getting_started_pytorch.ipynb

Iris_Bayesian_decision_making	Iris_Bayesian_decision_making_pytorch.ipynb

Func_apxm_MLP	Func_apxm_MLP_pytorch.ipynb

MNIST_MLP	MNIST_MLP_pytorch.ipynb

1_MNIST_CNN	1_MNIST_CNN_pytorch.ipynb

## Môi trường

Google Colab, GPU Tesla T4 (riêng Func_apxm_MLP và Iris chạy trên CPU)

PyTorch 2.14.1+cu130, torchvision 0.29.1+cu130

Thư viện thêm: torchinfo (cài trong notebook bằng !pip -q install torchinfo); scikit-learn, scipy, seaborn, matplotlib có sẵn trên Colab

Seed: 42

## Dữ liệu

MNIST: tự tải bằng torchvision.datasets.MNIST(download=True).

Iris: sklearn.datasets.load_iris().

Hàm Gaussian 2D: tự sinh trong notebook.

## Cách chạy

Mở notebook trên Colab (File → Upload notebook).

Với MNIST_MLP và 1_MNIST_CNN: Runtime → Change runtime type → T4 GPU.

Runtime → Restart session and run all.

Đồ thị và mô hình được lưu vào Google Drive/BTL03/P1/ (cần cấp quyền Drive khi được hỏi).

## Kết quả

Notebook	Chỉ số	PyTorch	Keras tham chiếu

MNIST_MLP	Test accuracy	…	≈ 0.9657
1_MNIST_CNN	Test accuracy	…	≈ 0.9889
Iris	Accuracy (sklearn / PyTorch GaussianNB / full cov)	…	0.9667 (một lần chạy của bản gốc)
Func_apxm_MLP	MSE test (B = 128, C = 190)	…	–
