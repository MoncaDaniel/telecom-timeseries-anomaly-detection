# Telecom Network Anomaly Detection

End-to-end data science project for **unsupervised anomaly detection in telecommunication networks**, using multivariate time-series data and an LSTM Autoencoder.

This project is designed as a **portfolio-grade implementation** demonstrating:
- applied machine learning on network KPIs,
- production-ready pipelines (training, scoring, evaluation),
- and an interactive monitoring interface.

---

## 🎯 Business context

Modern telecommunication networks (mobile, fixed, fiber) continuously generate large volumes of time-series data describing network performance (KPIs).

**Objective**
Detect abnormal network behavior (e.g. jamming, congestion, interference) in order to:
- anticipate service degradation,
-# Telecom Time-Series Anomaly Detection

## 1. Project Overview
This project implements an **end-to-end anomaly detection system for telecommunications networks**, based on multivariate time-series of network KPIs.

The objective is to detect **network incidents** (e.g. radio interference, congestion, service degradation) by learning normal network behavior and identifying significant deviations.

The project is designed as a **production-oriented data science portfolio**, covering:
- Data ingestion and preprocessing
- Unsupervised machine learning
- Evaluation with operational metrics
- CLI-based pipelines (train / score / evaluate)
- Interactive monitoring dashboard (Streamlit)

---

## 2. Business Context (Telecommunications)
Telecommunication networks continuously generate **Key Performance Indicators (KPIs)** such as:
- Signal strength
- Transmission errors
- Modulation quality
- Traffic volume

Abnormal behavior in these KPIs may indicate:
- Radio interference (e.g. jamming)
- Network congestion
- Equipment malfunction
- Quality of Service (QoS) degradation

Early detection is critical to:
- Maintain service quality
- Reduce customer impact
- Optimize operational costs

In real-world telecom environments, **incident labels are rare or unreliable**, which motivates the use of **unsupervised anomaly detection**.

---

## 3. Dataset
The project uses the public dataset:

**AliMaatouk/TelecomTS** (Hugging Face)

Dataset characteristics:
- Multivariate time-series
- Multiple simulated operational scenarios
- Synthetic anomalies with metadata (type, duration, affected KPIs)
- Labels available **only for evaluation**, not used for training

---

## 4. Methodology

### 4.1 Feature Selection
A subset of representative KPIs is used:
- `RSRP` – Reference Signal Received Power
- `DL_BLER` / `UL_BLER` – Downlink / Uplink Block Error Rate
- `UL_MCS` – Uplink Modulation and Coding Scheme
- `TX_Bytes` / `RX_Bytes` – Traffic volume

These KPIs jointly reflect **radio quality, reliability, and load**.

### 4.2 Windowing (Sliding Windows)
Time-series are split into **fixed-length overlapping windows** (e.g. 20 time steps).

Why windowing:
- LSTM models require fixed-length sequences
- Enables local, time-aware anomaly scoring
- Matches real operational monitoring (rolling analysis)

### 4.3 Scaling
KPIs are normalized using **Min-Max scaling** to ensure:
- Comparable feature ranges
- Stable neural network training
- No dominance of high-magnitude variables (e.g. bytes)

### 4.4 Model: LSTM Autoencoder
An **LSTM Autoencoder** is trained exclusively on normal behavior.

Architecture:
- **Encoder**: compresses temporal patterns into a latent representation
- **Decoder**: reconstructs the original sequence

Anomaly detection principle:
- Normal patterns are well reconstructed
- Abnormal patterns produce higher reconstruction error

The anomaly score is the **Mean Squared Error (MSE)** between input and reconstruction.

Why this approach:
- Fully unsupervised
- Robust to missing or incomplete labels
- Commonly used in industrial monitoring systems

---

## 5. Decision Logic

### 5.1 Window-Level Detection
Each time window is flagged as anomalous if: reconstruction_error > threshold

The reconstruction threshold is derived from the distribution of errors on training data (percentile-based).

### 5.2 Scenario-Level Incident Detection
A telecom **incident** is defined at scenario level, not window level.

Decision rule:
incident = (percentage of anomalous windows ≥ decision threshold)

This two-level approach avoids:
- Overreacting to isolated noise
- Missing persistent degradations

---

## 6. Evaluation Strategy
An evaluation pipeline is implemented to:
- Scan multiple scenario-level decision thresholds
- Compute operational metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- Select the threshold that optimizes F1-score

This explicitly highlights the **precision–recall trade-off**, which is critical in telecom operations:
- High precision reduces false alerts (alert fatigue)
- Recall can be tuned depending on operational risk tolerance

---

## 7. Results (Current Version)
Typical results on a balanced evaluation subset:
- Precision: **~1.0**
- Recall: **~0.5**
- F1-score: **~0.66**

Interpretation:
- The system is conservative (few false positives)
- Some partial or weak incidents are missed
- This trade-off is intentional and adjustable via decision thresholds

---

## 8. Application (Streamlit Dashboard)
An interactive dashboard provides:
- KPI time-series visualization
- Window-level anomaly scores
- Configurable decision thresholds
- Comparison between predicted and ground-truth incidents
- Operational interpretation of results

The interface is designed to be **minimal, professional, and operations-oriented**.

---

## 9. Project Structure
network-anomaly-detection/
├── app.py                  # CLI entrypoint
├── streamlit_app.py        # Streamlit dashboard
├── configs/                # YAML configs (train / score / evaluate)
├── src/
│   ├── data/               # Dataset loading
│   ├── features/           # Scaling & windowing
│   ├── models/             # LSTM Autoencoder
│   ├── training/           # Training pipeline
│   └── inference/          # Scoring & evaluation
├── notebooks/              # Exploratory analysis
├── reports/                # Evaluation outputs
├── models/                 # Trained artifacts (local)
└── requirements.txt

---

## 10. Reproducibility
The entire pipeline is driven by **YAML configuration files**:
- No hard-coded parameters
- Clear separation between code and configuration
- Reproducible experiments and evaluations

---

## 11. How to Run Locally

### 11.1 Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 11.2 Train the Model
python3 app.py train --config configs/train.yaml

### 11.3 Evaluate the Model
python3 app.py evaluate --config configs/eval.yaml

### 11.4 Launch the Dashboard
streamlit run streamlit_app.py

----

## 12. Limitations and Future Improvements
Possible next steps:
- Feature engineering (derivatives, rolling statistics)
- Model variants (CNN autoencoder, Transformer-based encoder)
- Per-anomaly-type decision thresholds
- Online / streaming inference
- Data drift monitoring and retraining strategy

---

## 13. Disclaimer
This project uses **synthetic data** for demostration purposes only. It does not represent any real telecom operator data.

---

# **Autor**

**Daniel Moncada Leon**

Data Science Portfolio Project – Telecommunications & Time-Series Anomaly Detection
