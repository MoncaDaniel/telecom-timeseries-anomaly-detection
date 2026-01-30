# src/training/train.py
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

from src.features.windowing import make_windows
from src.features.scaling import fit_global_minmax_scaler, transform_with_scaler
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.utils.paths import ensure_parent_dir, ensure_dir

def _sample_to_array(sample: Dict[str, Any], selected_kpis: List[str]) -> np.ndarray:
    df = pd.DataFrame(sample["KPIs"])
    df = df[selected_kpis]
    return df.values.astype(np.float32)

def _reconstruction_errors(model, device, X_np: np.ndarray, batch_size: int = 256) -> np.ndarray:
    X_t = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, X_t), batch_size=batch_size, shuffle=False)

    errs = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = model(xb)
            e = torch.mean((yhat - yb) ** 2, dim=(1, 2))  # per-window MSE
            errs.append(e.detach().cpu().numpy())
    return np.concatenate(errs)

def train_from_config(cfg: Dict[str, Any], dataset) -> None:
    seed = int(cfg["project"]["seed"])
    selected_kpis = cfg["features"]["selected_kpis"]
    window_size = int(cfg["features"]["window_size"])

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])

    percentile = float(cfg["thresholding"]["percentile"])

    artifacts = cfg["artifacts"]
    output_dir = artifacts["output_dir"]
    model_path = artifacts["model_path"]
    scaler_path = artifacts["scaler_path"]
    threshold_path = artifacts["threshold_path"]
    metadata_path = artifacts["metadata_path"]

    ensure_dir(output_dir)
    ensure_parent_dir(model_path)
    ensure_parent_dir(scaler_path)
    ensure_parent_dir(threshold_path)
    ensure_parent_dir(metadata_path)

    # 1) Collect NORMAL samples arrays to fit scaler
    normal_arrays = []
    for i in range(len(dataset)):
        if dataset[i]["labels"]["anomaly_present"] == "No":
            normal_arrays.append(_sample_to_array(dataset[i], selected_kpis))

    if len(normal_arrays) == 0:
        raise RuntimeError("No normal samples found. Cannot train.")

    # 2) Fit global scaler on normal
    scaler = fit_global_minmax_scaler(normal_arrays)

    # 3) Build training windows from normal data
    windows_list = []
    for arr in normal_arrays:
        arr_scaled = transform_with_scaler(scaler, arr)
        X_i = make_windows(arr_scaled, window_size=window_size)
        if X_i.shape[0] > 0:
            windows_list.append(X_i)

    X_train = np.concatenate(windows_list, axis=0)
    if X_train.shape[0] == 0:
        raise RuntimeError("No windows created. Try smaller window_size.")

    # 4) Model + training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = X_train.shape[2]

    model = LSTMAutoencoder(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"[train] epoch {epoch}/{epochs} - loss={epoch_loss:.6f}")

    # 5) Threshold from normal reconstruction errors
    train_err = _reconstruction_errors(model, device, X_train, batch_size=256)
    threshold = float(np.percentile(train_err, percentile))

    # 6) Save artifacts
    torch.save(
        {"state_dict": model.state_dict(), "n_features": n_features},
        model_path
    )
    joblib.dump(scaler, scaler_path)

    with open(threshold_path, "w") as f:
        json.dump({"percentile": percentile, "threshold": threshold}, f, indent=2)

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "selected_kpis": selected_kpis,
                "window_size": window_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "device_used": device,
                "n_train_windows": int(X_train.shape[0]),
                "threshold": threshold,
            },
            f,
            indent=2
        )

    print(f"[train] saved model -> {model_path}")
    print(f"[train] saved scaler -> {scaler_path}")
    print(f"[train] saved threshold -> {threshold_path}")
