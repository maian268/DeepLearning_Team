import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import generate_splits, save_split, gaussian_2d
from model import MLP
from train import train_model
from evaluate import predict

SEEDS = [42, 123, 2026]
CONFIG = {
    "train_n": 1200,
    "val_n": 300,
    "test_n": 100,
    "data_seed": 2026,
    "hidden_layers": 2,
    "hidden_width": 64,
    "activation": "tanh",
    "batch_size": 64,
    "epochs": 1000,
    "learning_rate": 1e-3,
    "patience": 100,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dirs():
    for d in ["data", "results", "figures", "checkpoints"]:
        os.makedirs(d, exist_ok=True)

def plot_data(train_split):
    X, y = train_split
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y[:, 0], s=8, alpha=0.45)
    ax.set(xlabel="x1", ylabel="x2", zlabel="y", title="Training data")
    fig.tight_layout()
    fig.savefig("figures/data_scatter_3d.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(X[:, 0], bins=30); axes[0].set_title("Distribution of x1")
    axes[1].hist(X[:, 1], bins=30); axes[1].set_title("Distribution of x2")
    axes[2].hist(y[:, 0], bins=30); axes[2].set_title("Distribution of y")
    fig.tight_layout()
    fig.savefig("figures/data_distributions.png", dpi=160)
    plt.close(fig)

def plot_surface(model):
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-4, 4, 100)
    xx1, xx2 = np.meshgrid(x1, x2)
    X = np.column_stack([xx1.ravel(), xx2.ravel()]).astype(np.float32)
    y_true = gaussian_2d(X).reshape(xx1.shape)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy().reshape(xx1.shape)
    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(xx1, xx2, y_true, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_title("True Gaussian surface"); ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_zlabel("y")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(xx1, xx2, y_pred, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set_title("MLP approximation"); ax2.set_xlabel("x1"); ax2.set_ylabel("x2"); ax2.set_zlabel("y_hat")
    fig.tight_layout()
    fig.savefig("figures/gaussian_surface.png", dpi=160)
    plt.close(fig)

def main():
    ensure_dirs()
    train_split, val_split, test_split = generate_splits(
        CONFIG["train_n"], CONFIG["val_n"], CONFIG["test_n"], CONFIG["data_seed"])
    save_split(train_split, "data/train.csv")
    save_split(val_split, "data/validation.csv")
    save_split(test_split, "data/test.csv")
    plot_data(train_split)

    run_rows, all_histories = [], []
    for seed in SEEDS:
        model = MLP(hidden_width=CONFIG["hidden_width"],
                    hidden_layers=CONFIG["hidden_layers"],
                    activation=CONFIG["activation"])
        model, history, train_info = train_model(
            model, train_split, val_split, seed=seed, epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"], learning_rate=CONFIG["learning_rate"],
            patience=CONFIG["patience"], device=DEVICE)
        metrics, y_true, y_pred = predict(model, test_split, DEVICE)
        row = {"seed": seed, **metrics, **train_info}
        run_rows.append(row)
        for item in history:
            all_histories.append({"seed": seed, **item})
        pd.DataFrame(history).to_csv(f"results/loss_history_seed_{seed}.csv", index=False)
        torch.save(model.state_dict(), f"checkpoints/model_seed_{seed}.pth")

        # Save the predictions from the final seed as a simple auditable CSV.
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred,
                      "error": y_pred - y_true}).to_csv(
                          f"results/predictions_seed_{seed}.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(y_true, y_pred, alpha=0.8)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0].plot([lo, hi], [lo, hi], "r--")
        axes[0].set(xlabel="True y", ylabel="Predicted y_hat",
                    title=f"Actual vs predicted (seed={seed})")
        errors = y_pred - y_true
        axes[1].hist(errors, bins=20, edgecolor="black")
        axes[1].set(xlabel="Prediction error", ylabel="Count", title="Error distribution")
        fig.tight_layout()
        fig.savefig(f"figures/evaluation_seed_{seed}.png", dpi=160)
        plt.close(fig)

    results = pd.DataFrame(run_rows)
    results.to_csv("results/seed_results.csv", index=False)
    pd.DataFrame(all_histories).to_csv("results/loss_history_all.csv", index=False)

    metrics_cols = ["mse", "mae", "rmse", "train_seconds"]
    summary = results[metrics_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std_sample"]
    summary.to_csv("results/summary_mean_std.csv")

    # Plot train/validation losses for all seeds.
    hist_df = pd.DataFrame(all_histories)
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed in SEEDS:
        part = hist_df[hist_df.seed == seed]
        ax.plot(part.epoch, part.train_loss, label=f"Train {seed}", alpha=0.8)
        ax.plot(part.epoch, part.val_loss, linestyle="--", label=f"Validation {seed}", alpha=0.8)
    ax.set(xlabel="Epoch", ylabel="MSE loss", title="Training and validation loss")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/loss_curve.png", dpi=160)
    plt.close(fig)

    print("\nTest results by seed:")
    print(results.to_string(index=False))
    print("\nMean ± sample standard deviation:")
    print(summary)
    print(f"\nDevice: {DEVICE}")
    print("Each seed checkpoint is saved separately. Use validation loss for model selection; test metrics are final evaluation only.")

if __name__ == "__main__":
    main()
