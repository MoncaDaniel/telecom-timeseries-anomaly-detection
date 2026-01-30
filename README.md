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
- improve operational excellence,
- support network supervision teams.

This project addresses typical challenges encountered by telecom data teams:
- multivariate time-series modeling,
- anomaly detection without reliable labels,
- translating model outputs into actionable operational signals.

---

## 🧠 Modeling approach

### Why unsupervised learning?
In real networks:
- anomalies are rare,
- labels are often missing or delayed,
- new failure modes appear over time.

👉 I use an **LSTM Autoencoder**, trained only on “normal” patterns, to detect deviations.

### Model
- **Architecture**: LSTM Autoencoder
- **Input**: sliding windows of multivariate KPIs
- **Output**: reconstruction error (MSE)
- **Anomaly score**: reconstruction error per window

A scenario-level incident is predicted based on the **percentage of anomalous windows**.

---

## 📊 Dataset

**Source**:
[`AliMaatouk/TelecomTS`](https://huggingface.co/datasets/AliMaatouk/TelecomTS)

**Description**:
- Synthetic yet realistic telecom network data
- Multiple KPIs (RSRP, BLER, MCS, traffic volumes…)
- Various scenarios:
  - normal behavior
  - jamming
  - congestion
  - interference

> ⚠️ Note: anomalies are intentionally over-represented for research purposes.
> This does not reflect real production ratios.

---

## 🏗 Project structure


network-anomaly-detection/
├── app.py                # CLI entrypoint (train / score / evaluate)
├── streamlit_app.py      # Interactive monitoring app
├── configs/              # YAML configuration files
├── data/
│   ├── raw/
│   ├── processed/
├── models/               # Trained models and artifacts
├── reports/              # Evaluation outputs and metrics
├── src/
│   ├── data/             # Data loading
│   ├── features/         # Scaling, windowing
│   ├── models/           # LSTM Autoencoder
│   ├── training/         # Training pipeline
│   ├── inference/        # Scoring and evaluation
│   └── utils/
└── notebooks/            # Exploratory analysis
