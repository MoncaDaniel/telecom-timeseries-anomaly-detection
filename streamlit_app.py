# streamlit_app.py
import random

import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.data.load_telecomts import load_telecomts
from src.inference.predict_sample import load_artifacts, score_sample


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_kpis(df: pd.DataFrame, kpis_to_plot: list[str], title: str):
    fig = go.Figure()
    x = np.arange(len(df))

    for col in kpis_to_plot:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col].values,
                mode="lines",
                name=col,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time index",
        yaxis_title="Value",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def plot_anomaly_score(errors: np.ndarray, recon_threshold: float, flags: np.ndarray):
    x = np.arange(len(errors))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=errors,
            mode="lines",
            name="Reconstruction error (MSE)",
        )
    )

    fig.add_hline(
        y=recon_threshold,
        line_dash="dash",
        annotation_text="Reconstruction threshold",
        annotation_position="top left",
    )

    if len(flags):
        idx = np.where(flags == 1)[0]
        if len(idx):
            fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=errors[idx],
                    mode="markers",
                    name="Flagged windows",
                )
            )

    fig.update_layout(
        title="Anomaly score (window-level)",
        xaxis_title="Window index",
        yaxis_title="Reconstruction error (MSE)",
        template="plotly_white",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


st.set_page_config(page_title="Telecom Network Anomaly Detection", layout="wide")

st.title("Telecom Network Anomaly Detection")
st.caption("Interactive monitoring prototype based on an LSTM Autoencoder (unsupervised anomaly detection).")

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.header("Configuration")

    config_path = st.text_input("Config file", value="configs/score.yaml")
    cfg = load_config(config_path)

    st.subheader("Model / Data")
    dataset_name = cfg["data"]["dataset_name"]
    data_files_pattern = cfg["data"]["data_files_pattern"]
    selected_kpis = cfg["features"]["selected_kpis"]
    window_size = int(cfg["features"]["window_size"])

    model_path = cfg["artifacts"]["model_path"]
    scaler_path = cfg["artifacts"]["scaler_path"]
    threshold_path = cfg["artifacts"]["threshold_path"]

    st.text(f"Dataset: {dataset_name}")
    st.text(f"Window size: {window_size}")

    st.divider()

    st.subheader("Display")
    kpis_to_plot = st.multiselect(
        "KPIs to plot",
        options=selected_kpis,
        default=selected_kpis,
    )

    st.divider()

    st.subheader("Decision rule")
    decision_threshold = st.slider(
        "Incident if pct windows flagged ≥",
        min_value=0.0,
        max_value=0.50,
        value=0.05,
        step=0.01,
    )

    st.divider()

    st.subheader("Sample selection")


# -----------------------
# Cache loaders
# -----------------------
@st.cache_resource
def cached_dataset(dataset_name: str, pattern: str):
    return load_telecomts(dataset_name, pattern)


@st.cache_resource
def cached_artifacts(model_path, scaler_path, threshold_path, selected_kpis, window_size):
    return load_artifacts(
        model_path=model_path,
        scaler_path=scaler_path,
        threshold_path=threshold_path,
        selected_kpis=selected_kpis,
        window_size=window_size,
    )


dataset = cached_dataset(dataset_name, data_files_pattern)
artifacts = cached_artifacts(model_path, scaler_path, threshold_path, selected_kpis, window_size)

# -----------------------
# Session state: keep a separate key for current index
# -----------------------
default_idx = int(cfg.get("scoring", {}).get("sample_index", 0))
if "current_sample_index" not in st.session_state:
    st.session_state.current_sample_index = default_idx

# -----------------------
# Sidebar controls (need dataset length now)
# -----------------------
with st.sidebar:
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Random sample", use_container_width=True):
            st.session_state.current_sample_index = random.randint(0, len(dataset) - 1)
            st.rerun()

    with col_b:
        manual_idx = st.number_input(
            "Sample index",
            min_value=0,
            max_value=len(dataset) - 1,
            value=int(st.session_state.current_sample_index),
            step=1,
        )
        st.session_state.current_sample_index = int(manual_idx)

    st.caption(f"Total samples: {len(dataset)}")


sample_index = int(st.session_state.current_sample_index)

# -----------------------
# Load sample + inference
# -----------------------
sample = dataset[sample_index]
labels = sample.get("labels", {})
anoms = sample.get("anomalies", {}) if isinstance(sample.get("anomalies", {}), dict) else {}

result = score_sample(artifacts, sample)
df_kpis = result["df_kpis"]
errors = result["window_errors"]
flags = result["window_flags"]
pct_flagged = float(result["pct_windows_flagged"])

pred_incident = "Yes" if pct_flagged >= decision_threshold else "No"
true_incident = labels.get("anomaly_present", "Unknown")
anomaly_type = anoms.get("type", "None")

# -----------------------
# Main layout
# -----------------------
col_left, col_mid, col_right = st.columns([0.28, 0.44, 0.28], gap="large")

with col_left:
    st.subheader("Scenario context")
    ctx = {
        "Sample index": int(sample_index),
        "Zone": labels.get("zone", "Unknown"),
        "Application": labels.get("application", "Unknown"),
        "Mobility": labels.get("mobility", "Unknown"),
        "Congestion": labels.get("congestion", "Unknown"),
        "True incident": true_incident,
        "Anomaly type": anomaly_type,
    }
    ctx_table = pd.DataFrame({"field": list(ctx.keys()), "value": list(ctx.values())})
    st.table(ctx_table)

    st.subheader("Decision inputs")
    inputs = {
        "Reconstruction threshold (window-level)": float(artifacts.recon_threshold),
        "Decision threshold (scenario-level)": float(decision_threshold),
        "Pct windows flagged": round(pct_flagged, 4),
    }
    inputs_table = pd.DataFrame({"field": list(inputs.keys()), "value": list(inputs.values())})
    st.table(inputs_table)

    st.subheader("Interpretation")
    st.write(
        "Window-level anomalies are identified using reconstruction error. "
        "A scenario-level incident is predicted using a decision rule based on the "
        "percentage of flagged windows."
    )

with col_mid:
    st.subheader("Network KPIs")
    if len(kpis_to_plot) == 0:
        st.info("Select at least one KPI to plot.")
    else:
        st.plotly_chart(
            plot_kpis(df_kpis, kpis_to_plot, "KPIs over time"),
            use_container_width=True,
        )

    st.subheader("Anomaly score")
    if len(errors) == 0:
        st.warning("No windows available for this sample (sequence too short).")
    else:
        st.plotly_chart(
            plot_anomaly_score(errors, artifacts.recon_threshold, flags),
            use_container_width=True,
        )

with col_right:
    st.subheader("Decision summary")
    st.metric("Predicted incident", pred_incident)
    st.metric("True incident", true_incident)
    st.metric("Pct windows flagged", f"{pct_flagged*100:.2f}%")
    st.metric("Window size", int(window_size))

    st.subheader("Operational statement")
    if pred_incident == "Yes":
        st.write(
            "The scenario exceeds the operational decision threshold, indicating a probable incident "
            "requiring investigation."
        )
    else:
        st.write(
            "The scenario remains below the operational decision threshold under the current configuration."
        )

    with st.expander("Raw sample metadata"):
        st.json(
            {
                "start_time": sample.get("start_time"),
                "end_time": sample.get("end_time"),
                "sampling_rate": sample.get("sampling_rate"),
                "labels": labels,
                "anomalies": anoms,
            }
        )
