import copy
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # More reproducible runs; CPU is the default device in this project.
    torch.use_deterministic_algorithms(True, warn_only=True)

def make_loader(split, batch_size=64, shuffle=False, seed=42):
    X, y = split
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)

def train_model(model, train_split, val_split, seed, epochs=1000,
                batch_size=64, learning_rate=1e-3, patience=100, device="cpu"):
    set_seed(seed)
    model = model.to(device)
    train_loader = make_loader(train_split, batch_size, True, seed)
    val_loader = make_loader(val_split, batch_size, False, seed)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_val_loss": best_val,
                            "train_seconds": elapsed}
