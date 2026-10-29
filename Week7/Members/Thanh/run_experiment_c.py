import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs("figures", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# -------------------------------------------------------------
# 1. HÀM GAUSSIAN 2D & TỰ ĐỘNG SINH DỮ LIỆU
# -------------------------------------------------------------
MU = np.array([0.0, 0.0])
COV = np.array([[0.3, -0.5], 
                [-0.5, 2.0]])
DET_COV = np.linalg.det(COV)
INV_COV = np.linalg.inv(COV)

def gaussian_2d(X):
    """
    X: ma trận (N, 2)
    Trả về y: ma trận (N, 1) tính theo công thức hàm mật độ Gaussian 2D
    """
    diff = X - MU
    quad = np.sum((diff @ INV_COV) * diff, axis=1)
    coef = 1.0 / (2.0 * np.pi * np.sqrt(DET_COV))
    return (coef * np.exp(-0.5 * quad)).reshape(-1, 1)

def generate_dataset(n_samples, seed=42):
    """Sinh ngẫu nhiên x1 trong [-2, 2], x2 trong [-4, 4]"""
    np.random.seed(seed)
    x1 = np.random.uniform(-2.0, 2.0, size=(n_samples, 1))
    x2 = np.random.uniform(-4.0, 4.0, size=(n_samples, 1))
    X = np.hstack([x1, x2])
    y = gaussian_2d(X)
    return X, y

# Sinh tập mẫu cố định
X_train_np, y_train_np = generate_dataset(1000, seed=10)
X_val_np, y_val_np     = generate_dataset(200, seed=20)
X_test_np, y_test_np   = generate_dataset(100, seed=99)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_t = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)

X_val_t, y_val_t   = to_t(X_val_np), to_t(y_val_np)
X_test_t, y_test_t = to_t(X_test_np), to_t(y_test_np)

train_loader = DataLoader(
    TensorDataset(to_t(X_train_np), to_t(y_train_np)),
    batch_size=32,
    shuffle=True
)

# -------------------------------------------------------------
# 2. XÂY DỰNG MẠNG MLP TÙY BIẾN NHIỀU TẦNG ẨN
# -------------------------------------------------------------
class DeepMLP(nn.Module):
    def __init__(self, in_features=2, hidden_dim=32, num_hidden_layers=1, out_features=1):
        super(DeepMLP, self).__init__()
        layers = []
        # Tầng ẩn 1
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU())
        
        # Các tầng ẩn tiếp theo từ 2 đến C
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        # Tầng ngõ ra
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 3. VÒNG LẶP HUẤN LUYỆN, LƯU LOSS THEO EPOCH & ĐO THỜI GIAN
# -------------------------------------------------------------
def train_and_evaluate(model, epochs=250, lr=0.003):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_state = None
    epoch_losses = []
    
    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(bx)
            
        # Đánh giá validation ở cuối mỗi epoch
        model.eval()
        with torch.no_grad():
            v_pred = model(X_val_t)
            v_loss = criterion(v_pred, y_val_t).item()
            
        epoch_losses.append(v_loss)
        
        # Checkpoint lưu model tốt nhất
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = model.state_dict().copy()
            
    train_time = time.time() - start_time
    
    # Nạp mô hình tốt nhất để kiểm thử độc lập
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_mse = criterion(test_pred, y_test_t).item()
        test_rmse = np.sqrt(test_mse)
        test_mae = torch.mean(torch.abs(test_pred - y_test_t)).item()
        
    return test_mse, test_rmse, test_mae, train_time, epoch_losses, model

# -------------------------------------------------------------
# 4. CHẠY MẠNG VỚI C = 1, 2, 3, 4 TẦNG ẨN
# -------------------------------------------------------------
if __name__ == "__main__":
    print(f"=== BẮT ĐẦU KHẢO SÁT SỐ TẦNG ẨN (THỰC NGHIỆM C) TRÊN {device} ===")
    depths = [1, 2, 3, 4]
    seeds = [42, 100, 2026]
    
    all_summary = []
    convergence_curves = {}
    best_model_c = None
    lowest_mse = float('inf')
    
    for C in depths:
        mses, rmses, maes, times = [], [], [], []
        print(f"\n-> Đang huấn luyện mô hình có {C} tầng ẩn...")
        
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            
            model = DeepMLP(in_features=2, hidden_dim=32, num_hidden_layers=C).to(device)
            mse, rmse, mae, t_elapsed, loss_hist, trained_m = train_and_evaluate(model)
            
            mses.append(mse)
            rmses.append(rmse)
            maes.append(mae)
            times.append(t_elapsed)
            
            if s == seeds[0]:
                convergence_curves[C] = loss_hist
                
            if mse < lowest_mse:
                lowest_mse = mse
                best_model_c = trained_m
                torch.save(best_model_c.state_dict(), "checkpoints/best_model_c.pt")
                
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_summary.append({
            "C (Tầng ẩn)": C,
            "Số tham số": param_count,
            "Test MSE (Mean)": np.mean(mses),
            "Test MSE (Std)": np.std(mses),
            "Test RMSE (Mean)": np.mean(rmses),
            "Test MAE (Mean)": np.mean(maes),
            "Thời gian train (s)": np.mean(times)
        })
        print(f"   Hoàn thành C={C} | MSE: {np.mean(mses):.6e} | Time: {np.mean(times):.2f}s")
        
    df = pd.DataFrame(all_summary)
    df.to_csv("results_experiment_c.csv", index=False)
    print("\n=== KẾT QUẢ TỔNG HỢP THỰC NGHIỆM C ===")
    print(df.to_string(index=False))

    # -------------------------------------------------------------
    # 5. VẼ ĐỒ THỊ HỘI TỤ, SAI SỐ VÀ THỜI GIAN
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # Đồ thị Test MSE theo số tầng
    axes[0].errorbar(df["C (Tầng ẩn)"], df["Test MSE (Mean)"], yerr=df["Test MSE (Std)"], 
                     fmt='-o', color='royalblue', capsize=5, linewidth=2)
    axes[0].set_title("Sai số Test MSE theo Số Tầng Ẩn (C)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Số tầng ẩn C")
    axes[0].set_ylabel("Test MSE")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Đồ thị Thời gian huấn luyện
    axes[1].plot(df["C (Tầng ẩn)"], df["Thời gian train (s)"], '-s', color='crimson', linewidth=2)
    axes[1].set_title("Thời Gian Huấn Luyện theo Số Tầng Ẩn", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Số tầng ẩn C")
    axes[1].set_ylabel("Thời gian (giây)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # Đồ thị Hội tụ Loss qua các Epoch
    for C, curve in convergence_curves.items():
        axes[2].plot(curve, label=f"{C} tầng ẩn", linewidth=1.8)
    axes[2].set_title("Đồ Thị Hội Tụ (Validation Loss)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE Loss")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("figures/experiment_c_curves.png", dpi=300)
    plt.close()
    print("\n-> Đã lưu đồ thị hội tụ và sai số tại: figures/experiment_c_curves.png")

    # -------------------------------------------------------------
    # 6. VẼ BỀ MẶT HÀM THẬT VÀ BỀ MẶT XẤP XỈ 3D
    # -------------------------------------------------------------
    grid_x1 = np.linspace(-2, 2, 60)
    grid_x2 = np.linspace(-4, 4, 60)
    X1, X2 = np.meshgrid(grid_x1, grid_x2)
    grid_pts = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
    
    Y_real = gaussian_2d(grid_pts).reshape(60, 60)
    
    best_model_c.eval()
    with torch.no_grad():
        pts_t = torch.tensor(grid_pts, dtype=torch.float32).to(device)
        Y_approx = best_model_c(pts_t).cpu().numpy().reshape(60, 60)
        
    fig3d = plt.figure(figsize=(14, 6))
    
    ax_real = fig3d.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax_real.plot_surface(X1, X2, Y_real, cmap='viridis', edgecolor='none', alpha=0.9)
    ax_real.set_title("Bề Mặt Hàm Gaussian Thật $f(x_1, x_2)$", fontsize=12, fontweight='bold')
    ax_real.set_xlabel("$x_1$")
    ax_real.set_ylabel("$x_2$")
    ax_real.set_zlabel("$f(x)$")
    
    ax_approx = fig3d.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax_approx.plot_surface(X1, X2, Y_approx, cmap='plasma', edgecolor='none', alpha=0.9)
    ax_approx.set_title(f"Bề Mặt Xấp Xỉ Bằng MLP Tối Ưu (C={df.loc[df['Test MSE (Mean)'].idxmin(), 'C (Tầng ẩn)']})", fontsize=12, fontweight='bold')
    ax_approx.set_xlabel("$x_1$")
    ax_approx.set_ylabel("$x_2$")
    ax_approx.set_zlabel("$\hat{y}$")
    
    plt.tight_layout()
    plt.savefig("figures/surface_comparison.png", dpi=300)
    plt.close()
    print("-> Đã lưu so sánh bề mặt 3D tại: figures/surface_comparison.png")
    print("=== HOÀN THÀNH TOÀN BỘ YÊU CẦU THỰC NGHIỆM C ===")