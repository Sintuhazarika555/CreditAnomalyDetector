## CREDIT ANOMALY DETECTOR 

In real-world fraud detection, labeled fraud cases are scarce, delayed, or non-existent during model training. This project tackles the challenge through an **unsupervised anomaly detection** framework using an **Isolation Forest** ensemble.

The algorithm operates on two fundamental assumptions:

1. **Rarity:** Anomalies constitute a fraction of the total dataset (< 0.2%).
2. **Geometric Distinctness:** Anomalous transactions deviate noticeably from dense, normal spending clusters in high-dimensional feature space.

![preview1](<Screenshot 2026-10-04 035951.png>)
![preview2](<Screenshot 2026-10-04 040024.png>)

---

## System Architecture

```text
credit-anomaly-detector/
├── data/                  # Stores raw transaction data (creditcard.csv)
├── models/                # Serialized model artifacts (isolation_forest.joblib)
├── reports/               # Auto-generated latent space visualization plots
├── src/
│   ├── __init__.py        # Package marker
│   ├── load_data.py       # Configurable, memory-safe data ingestion
│   ├── model.py           # RobustScaler & Isolation Forest architecture
│   ├── train.py           # Offline training & artifact serialization
│   └── visualize.py       # Evaluation metrics & PCA latent projection
├── app.py                 # Interactive Streamlit dashboard
├── main.py                # Headless batch CLI orchestrator
└── requirements.txt       # Project dependencies

```
---

