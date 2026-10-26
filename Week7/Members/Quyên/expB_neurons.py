import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import json
import logging
from datetime import datetime

# ================= CẤU HÌNH CHUNG =================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Chạy trên: {DEVICE}")

# Tham số hàm Gaussian
MU = np.array([0.0, 0.0])
C = np.array([[0.3, -0.5],
              [-0.5, 2.0]])
C_INV = np.linalg.inv(C)
DET_C = np.linalg.det(C)

# Siêu tham số cố định cho Thực nghiệm B
NUM_TRAIN = 1000       # cố định
NUM_VAL = 300
NUM_TEST = 100
BATCH_SIZE = 32
EPOCHS = 500
LR = 1e-3
ACTIVATION = nn.ReLU()
# Số nơ-ron khảo sát
HIDDEN_NEURONS_LIST = list(range(1, 11))  # 1 → 10
SAVE_DIR = "expB_results"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "expB_run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Thiết bị chạy: %s", DEVICE)

# ================= 1. HÀM TÍNH GAUSSIAN 2D =================
def gaussian_2d(x):
    """
    Tính giá trị hàm mật độ Gaussian 2D theo công thức:
    f(x) = 1/(2π√det(C)) * exp(-0.5*(x-μ)^T · C⁻¹ · (x-μ))
    x: mảng shape (N, 2)
    """
    x = np.atleast_2d(x)
    diff = x - MU
    quadratic = np.einsum('ki,ij,kj->k', diff, C_INV, diff)
    norm_factor = 1.0 / (2 * np.pi * np.sqrt(DET_C))
    return norm_factor * np.exp(-0.5 * quadratic)

# ================= 2. SINH DỮ LIỆU =================
def generate_data(n_samples, x1_min=-2, x1_max=2, x2_min=-4, x2_max=4):
    x1 = np.random.uniform(x1_min, x1_max, n_samples)
    x2 = np.random.uniform(x2_min, x2_max, n_samples)
    X = np.column_stack([x1, x2])
    y = gaussian_2d(X)
    return X, y

# Sinh dữ liệu
logger.info(
    "Tạo dữ liệu: train=%d, validation=%d, test=%d",
    NUM_TRAIN, NUM_VAL, NUM_TEST
)
X_train, y_train = generate_data(NUM_TRAIN)
X_val, y_val = generate_data(NUM_VAL)
X_test, y_test = generate_data(NUM_TEST)

# Chuyển sang Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# Tạo DataLoader
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=NUM_VAL, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=NUM_TEST, shuffle=False)

# Trực quan hóa dữ liệu
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='steelblue', s=10, alpha=0.6, label='Train')
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y = f(x_1, x_2)$")
ax.set_title("Phân bố dữ liệu huấn luyện – Hàm Gaussian 2D")
ax.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/data_distribution.png", dpi=200)
plt.close()
logger.info("Đã lưu biểu đồ phân bố dữ liệu.")

# ================= 3. ĐỊNH NGHĨA MÔ HÌNH MLP =================
class MLPRegressor(nn.Module):
    """MLP 1 tầng ẩn, tùy chỉnh số nơ-ron"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.layers(x)

# ================= 4. HÀM HUẤN LUYỆN =================
def train_one_model(hidden_dim):
    logger.info("Bắt đầu huấn luyện mô hình có %d nơ-ron ẩn.", hidden_dim)
    model = MLPRegressor(hidden_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    best_state = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        # Huấn luyện
        model.train()
        total_train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * Xb.size(0)
        avg_train_loss = total_train_loss / len(train_ds)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = model(Xb)
                loss = criterion(pred, yb)
                total_val_loss += loss.item() * Xb.size(0)
        avg_val_loss = total_val_loss / len(val_ds)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Lưu mô hình tốt nhất
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch + 1 == EPOCHS:
            logger.info(
                "Ẩn %d nơ-ron | Epoch %d/%d | Train MSE: %.6f | Val MSE: %.6f",
                hidden_dim, epoch + 1, EPOCHS, avg_train_loss, avg_val_loss
            )

    # Tải mô hình tốt nhất & đánh giá trên tập TEST
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_test = model(X_test_t.to(DEVICE))
        y_test_np = y_test_t.numpy()
        pred_test_np = pred_test.cpu().numpy()

        mse = np.mean((y_test_np - pred_test_np) ** 2)
        mae = np.mean(np.abs(y_test_np - pred_test_np))
        rmse = np.sqrt(mse)

    # Lưu trọng số
    model_path = f"{SAVE_DIR}/model_B_hid{hidden_dim}.pth"
    torch.save(best_state, model_path)
    logger.info(
        "Hoàn tất ẩn %d nơ-ron | Best Val MSE: %.6f | Test MSE: %.6f | "
        "Test MAE: %.6f | Test RMSE: %.6f | Mô hình: %s",
        hidden_dim, best_val_loss, mse, mae, rmse, model_path
    )

    return {
        "hidden_neurons": hidden_dim,
        "best_val_mse": best_val_loss,
        "test_mse": mse,
        "test_mae": mae,
        "test_rmse": rmse,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history
    }

# ================= 5. CHẠY THỰC NGHIỆM B =================
if __name__ == "__main__":
    results = []
    logger.info("Log được ghi tại: %s", LOG_FILE)
    logger.info(
        "Bắt đầu thực nghiệm B | Số nơ-ron: %s | Epoch tối đa: %d | Batch size: %d",
        HIDDEN_NEURONS_LIST, EPOCHS, BATCH_SIZE
    )

    for h in HIDDEN_NEURONS_LIST:
        res = train_one_model(h)
        results.append(res)

    # Lưu kết quả ra CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if "history" not in k} for r in results])
    df.to_csv(f"{SAVE_DIR}/expB_summary.csv", index=False)
    logger.info("Đã lưu kết quả tóm tắt tại %s/expB_summary.csv", SAVE_DIR)
    logger.info("Kết quả tóm tắt:\n%s", df.round(6).to_string(index=False))

    # ---------- Biểu đồ 1: Sai số theo số nơ-ron ----------
    plt.figure(figsize=(8, 5))
    plt.plot(df["hidden_neurons"], df["test_mse"], 'o-', color='#c0392b', label="Test MSE", linewidth=2)
    plt.plot(df["hidden_neurons"], df["best_val_mse"], 's--', color='#2980b9', label="Val MSE", linewidth=2)
    plt.xlabel("Số nơ-ron tầng ẩn", fontsize=12)
    plt.ylabel("Sai số trung bình phương (MSE)", fontsize=12)
    plt.title("Sai số theo số nơ-ron tầng ẩn – Thực nghiệm B", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/expB_error_curve.png", dpi=200)
    plt.close()
    logger.info("Đã lưu biểu đồ sai số.")

    # ---------- Biểu đồ 2: Đường học – 3 trường hợp điển hình ----------
    plt.figure(figsize=(10, 5))
    for idx, h in enumerate([1, 5, 10]):
        r = results[h-1]
        plt.subplot(1, 3, idx+1)
        plt.plot(r["train_loss_history"], label="Train", color='#27ae60')
        plt.plot(r["val_loss_history"], label="Validation", color='#e67e22')
        plt.title(f"{h} nơ-ron", fontsize=11)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/expB_learning_curves.png", dpi=200)
    plt.close()
    logger.info("Đã lưu biểu đồ đường học.")

    # ---------- Biểu đồ 3: Bề mặt dự đoán vs thật ----------
    model_best = MLPRegressor(5).to(DEVICE)
    model_best.load_state_dict(torch.load(f"{SAVE_DIR}/model_B_hid5.pth"))
    model_best.eval()

    xx, yy = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-4, 4, 50))
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    y_true_grid = gaussian_2d(X_grid).reshape(xx.shape)

    with torch.no_grad():
        y_pred_grid = model_best(torch.tensor(X_grid, dtype=torch.float32).to(DEVICE))\
                        .cpu().numpy().reshape(xx.shape)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xx, yy, y_true_grid, cmap='viridis', alpha=0.8)
    ax1.set_title("Bề mặt hàm thật", fontsize=12)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$y$")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xx, yy, y_pred_grid, cmap='plasma', alpha=0.8)
    ax2.set_title("Bề mặt dự đoán – 5 nơ-ron", fontsize=12)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_zlabel("$\hat{y}$")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/expB_surface_comparison.png", dpi=200)
    plt.close()

    logger.info("Đã lưu biểu đồ so sánh bề mặt.")
    logger.info("Hoàn tất thực nghiệm B. File log: %s", LOG_FILE)