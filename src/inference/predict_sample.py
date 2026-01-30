import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from src.features.windowing import make_windows
from src.features.scaling import transform_with_scaler
from src.models.lstm_autoencoder import LSTMAutoencoder


@dataclass
class ModelArtifacts:
    model: torch.nn.Module
    scaler: Any
    recon_threshold: float
    selected_kpis: List[str]
    window_size: int
    device: str


def load_artifacts(
    model_path: str,
    scaler_path: str,
    threshold_path: str,
    selected_kpis: List[str],
    window_size: int,
    device: Optional[str] = None,
) -> ModelArtifacts:
    """
    (définition) artifacts = fichiers générés par le training:
      - model weights (poids du modèle)
      - scaler (normalisation)
      - threshold (seuil reconstruction)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = torch.load(model_path, map_location=device)
    model = LSTMAutoencoder(pack["n_features"]).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)

    with open(threshold_path, "r") as f:
        recon_threshold = float(json.load(f)["threshold"])

    return ModelArtifacts(
        model=model,
        scaler=scaler,
        recon_threshold=recon_threshold,
        selected_kpis=selected_kpis,
        window_size=int(window_size),
        device=device,
    )


def sample_to_dataframe(sample: Dict[str, Any], selected_kpis: List[str]) -> pd.DataFrame:
    """
    Convertit un sample HF -> DataFrame time series (1 ligne = 1 pas de temps).
    """
    df = pd.DataFrame(sample["KPIs"])
    # on garde uniquement les KPIs choisis
    df = df[selected_kpis].copy()
    return df


def score_sample(
    artifacts: ModelArtifacts,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Retourne:
      - df_kpis (time series)
      - window_errors (score par fenêtre)
      - window_flags (au-dessus du seuil recon)
      - pct_windows_flagged
    """
    df_kpis = sample_to_dataframe(sample, artifacts.selected_kpis)
    arr = df_kpis.values.astype("float32")

    # scaling (normalisation)
    arr_scaled = transform_with_scaler(artifacts.scaler, arr)

    # windowing (fenêtrage)
    X = make_windows(arr_scaled, artifacts.window_size)  # shape: (n_windows, window_size, n_features)
    if len(X) == 0:
        return {
            "df_kpis": df_kpis,
            "window_errors": np.array([]),
            "window_flags": np.array([]),
            "pct_windows_flagged": 0.0,
        }

    X_t = torch.tensor(X).to(artifacts.device)

    with torch.no_grad():
        recon = artifacts.model(X_t)
        # (définition) MSE = Mean Squared Error (moyenne des carrés des erreurs)
        errors = torch.mean((recon - X_t) ** 2, dim=(1, 2)).cpu().numpy()

    flags = (errors > artifacts.recon_threshold).astype(int)
    pct = float(flags.mean()) if len(flags) else 0.0

    return {
        "df_kpis": df_kpis,
        "window_errors": errors,
        "window_flags": flags,
        "pct_windows_flagged": pct,
    }
