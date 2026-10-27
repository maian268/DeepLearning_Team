import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

def predict(model, split, device="cpu"):
    X, y = split
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy()
    y_true = y.reshape(-1)
    y_pred = pred.reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {"mse": float(mse), "mae": float(mae), "rmse": rmse}, y_true, y_pred
