import json
import numpy as np
import pandas as pd
import torch
import joblib

from src.features.windowing import make_windows
from src.features.scaling import transform_with_scaler
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.utils.paths import ensure_parent_dir


def _sample_to_array(sample, selected_kpis):
    df = pd.DataFrame(sample["KPIs"])
    return df[selected_kpis].values.astype("float32")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: arrays of 0/1
    Returns dict with confusion matrix + accuracy/precision/recall/F1.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_from_config(cfg, dataset):
    selected_kpis = cfg["features"]["selected_kpis"]
    window_size = int(cfg["features"]["window_size"])

    n_per_class = int(cfg["evaluation"]["n_per_class"])
    decision_thresholds = cfg["evaluation"]["decision_thresholds"]
    selection_metric = str(cfg["evaluation"].get("selection_metric", "f1")).lower()

    output_csv = cfg["evaluation"]["output_csv"]
    metrics_json = cfg["evaluation"]["metrics_json"]
    scan_csv = cfg["evaluation"]["scan_csv"]

    ensure_parent_dir(output_csv)
    ensure_parent_dir(metrics_json)
    ensure_parent_dir(scan_csv)

    model_path = cfg["artifacts"]["model_path"]
    scaler_path = cfg["artifacts"]["scaler_path"]
    threshold_path = cfg["artifacts"]["threshold_path"]

    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load artifacts
    pack = torch.load(model_path, map_location=device)
    model = LSTMAutoencoder(pack["n_features"]).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)
    with open(threshold_path) as f:
        recon_threshold = float(json.load(f)["threshold"])

    # Collect indices by class
    idx_yes, idx_no = [], []
    for i in range(len(dataset)):
        lab = dataset[i]["labels"]["anomaly_present"]
        if lab == "Yes":
            idx_yes.append(i)
        else:
            idx_no.append(i)

    if len(idx_no) == 0 or len(idx_yes) == 0:
        raise RuntimeError("Need both 'Yes' and 'No' samples for evaluation.")

    # Sample indices (balanced)
    choose_yes = rng.choice(idx_yes, size=min(n_per_class, len(idx_yes)), replace=False)
    choose_no = rng.choice(idx_no, size=min(n_per_class, len(idx_no)), replace=False)
    chosen = np.concatenate([choose_yes, choose_no])
    rng.shuffle(chosen)

    # ---- Compute sample-level pct_windows_flagged once ----
    rows = []
    for idx in chosen:
        sample = dataset[int(idx)]

        arr = _sample_to_array(sample, selected_kpis)
        arr = transform_with_scaler(scaler, arr)
        X = make_windows(arr, window_size)

        if len(X) == 0:
            continue

        X_t = torch.tensor(X).to(device)
        with torch.no_grad():
            recon = model(X_t)
            errors = torch.mean((recon - X_t) ** 2, dim=(1, 2)).cpu().numpy()

        pct_flagged = float((errors > recon_threshold).mean())
        label = sample["labels"]["anomaly_present"]
        true_incident = 1 if label == "Yes" else 0

        rows.append({
            "sample_index": int(idx),
            "label_anomaly_present": label,
            "true_incident": true_incident,
            "pct_windows_flagged": pct_flagged,
        })

    df = pd.DataFrame(rows)

    # Save the per-sample summary (independent of decision threshold)
    df.to_csv(output_csv, index=False)

    # ---- Scan decision thresholds ----
    scan_rows = []
    best = None

    for t in decision_thresholds:
        t = float(t)
        y_true = df["true_incident"].to_numpy()
        y_pred = (df["pct_windows_flagged"].to_numpy() >= t).astype(int)

        metrics = _compute_metrics(y_true, y_pred)
        row = {
            "decision_threshold": t,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["confusion_matrix"]["tp"],
            "fp": metrics["confusion_matrix"]["fp"],
            "fn": metrics["confusion_matrix"]["fn"],
            "tn": metrics["confusion_matrix"]["tn"],
        }
        scan_rows.append(row)

        # track best threshold
        score = row.get(selection_metric, None)
        if score is None:
            raise ValueError(f"Unknown selection_metric='{selection_metric}'. Use 'f1' or 'recall' or 'precision' or 'accuracy'.")
        if best is None or score > best["score"]:
            best = {"threshold": t, "score": score, "metrics": metrics}

    scan_df = pd.DataFrame(scan_rows).sort_values("decision_threshold")
    scan_df.to_csv(scan_csv, index=False)

    # ---- Save a clean JSON payload for README ----
    payload = {
        "n_samples_evaluated": int(len(df)),
        "window_size": window_size,
        "reconstruction_error_threshold": recon_threshold,
        "decision_thresholds_tested": [float(x) for x in decision_thresholds],
        "selection_metric": selection_metric,
        "best_decision_threshold": float(best["threshold"]),
        "best_metrics": best["metrics"],
    }

    with open(metrics_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[evaluate] saved per-sample summary -> {output_csv}")
    print(f"[evaluate] saved threshold scan -> {scan_csv}")
    print(f"[evaluate] saved metrics -> {metrics_json}")
    print("[evaluate] best decision threshold:")
    print(json.dumps(payload, indent=2))
