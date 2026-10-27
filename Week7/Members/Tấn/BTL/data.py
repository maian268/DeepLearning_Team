import numpy as np
import pandas as pd

MU = np.array([0.0, 0.0], dtype=np.float64)
COV = np.array([[0.3, -0.5], [-0.5, 2.0]], dtype=np.float64)

def gaussian_2d(X):
    """Evaluate the 2D Gaussian density specified in the assignment."""
    X = np.asarray(X, dtype=np.float64)
    inv_cov = np.linalg.inv(COV)
    det_cov = np.linalg.det(COV)
    diff = X - MU
    exponent = -0.5 * np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    return np.exp(exponent) / (2.0 * np.pi * np.sqrt(det_cov))

def generate_splits(train_n=1200, val_n=300, test_n=100, seed=2026):
    """Generate independent train/validation/test samples uniformly in the given domain."""
    rng = np.random.default_rng(seed)
    def sample(n):
        x1 = rng.uniform(-2.0, 2.0, n)
        x2 = rng.uniform(-4.0, 4.0, n)
        X = np.column_stack([x1, x2])
        y = gaussian_2d(X).reshape(-1, 1)
        return X.astype(np.float32), y.astype(np.float32)
    return sample(train_n), sample(val_n), sample(test_n)

def save_split(split, path):
    X, y = split
    pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y[:, 0]}).to_csv(path, index=False)
