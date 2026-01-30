# src/inference/score.py
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

from src.features.windowing import make_windows
from src.features.scaling import transform_with_scaler
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.utils.paths import ensure_parent_dir

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
            e = torch.mean((yhat - yb) ** 2, dim=(1, 2))
            errs.append(e.detach().cpu().numpy())
    return np.concatenate(errs)

def score_sample_from_config(cfg: Dict[str, Any], dataset) -> None:
    selected_kpis = cfg["features"]["selected_kpis"]
    window_size = int(cfg["features"]["window_size"])

    model_path = cfg["artifacts"]["model_path"]
    scaler_path = cfg["artifacts"]["scaler_path"]
    threshold_path = cfg["artifacts"]["threshold_path"]

    sample_index = int(cfg["scoring"]["sample_index"])
    output_csv = cfg["scoring"]["output_csv"]
    ensure_parent_dir(output_csv)

    # Load artifacts
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = torch.load(model_path, map_location=device)
    n_features = pack["n_features"]
    model = LSTMAutoencoder(n_features=n_features).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)
    with open(threshold_path, "r") as f:
        thr = json.load(f)["threshold"]

    # Build windows
    sample = dataset[sample_index]
    arr = _sample_to_array(sample, selected_kpis)
    arr_scaled = transform_with_scaler(scaler, arr)

    X = make_windows(arr_scaled, window_size=window_size)
    err = _reconstruction_errors(model, device, X)

    is_anom = err > thr

    # Save per-window scores
    out = pd.DataFrame({
        "window_index": np.arange(len(err)),
        "error": err,
        "is_anomaly": is_anom.astype(int),
        "covers_time_start": np.arange(len(err)),
        "covers_time_end": np.arange(len(err)) + window_size - 1,
    })
    out.to_csv(output_csv, index=False)

    print(f"[score] sample_index={sample_index}")
    print(f"[score] label anomaly_present={sample['labels']['anomaly_present']} | type={sample['anomalies']['type'] if sample['anomalies']['exists'] else 'None'}")
    print(f"[score] threshold={thr:.6f} | pct_windows_flagged={float(is_anom.mean()):.4f}")
    print(f"[score] saved -> {output_csv}")
