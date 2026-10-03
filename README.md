```markdown
# 💳 Unsupervised Credit Card Fraud & Anomaly Detector

An end-to-end unsupervised machine learning system designed to detect fraudulent credit card transactions without using ground-truth labels during training. The system pairs an automated Command-Line Interface (CLI) batch pipeline with an interactive Streamlit web dashboard for real-time risk assessment and latent space exploration.

```

---

## 📌 Project Overview

In real-world fraud detection, labeled fraud cases are scarce, delayed, or non-existent during model training. This project tackles the challenge through an **unsupervised anomaly detection** framework using an **Isolation Forest** ensemble.

The algorithm operates on two fundamental assumptions:

1. **Rarity:** Anomalies constitute a fraction of the total dataset (< 0.2%).
2. **Geometric Distinctness:** Anomalous transactions deviate noticeably from dense, normal spending clusters in high-dimensional feature space.

---

## 🏗️ System Architecture

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

## 🔬 Core Technical Highlights

* **Unsupervised Pipeline:** Trains on 29 continuous behavioral features ($V_1 \dots V_{28}$ and purchase amount) while withholding ground-truth labels during model fitting.
* **Robust Feature Preprocessing:** Uses `RobustScaler` rather than standard z-score normalization. By scaling via median and Interquartile Range (IQR), extreme spend amounts cannot skew the normalization boundaries.
* **Isolation Forest Ensemble:** Isolates anomalies via random partitioning trees (`iTrees`). Sparse, outlying transactions isolate in significantly shorter average path lengths than dense, regular transactions.
* **Dimensionality Reduction (PCA):** Projects the 29-dimensional feature space into 2 principal components to inspect cluster separation and outlier distributions.
* **Dual-Interface Delivery:**
* **CLI Batch Engine (`main.py`):** Runs automated training runs, records classification reports, outputs static diagnostic plots to `reports/`, and saves model binaries to `models/`.
* **Streamlit Web Application (`app.py`):** Provides an interactive dashboard featuring dynamic parameter sliders, interactive Plotly PCA scatter plots, and a human-readable risk simulation engine.



---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/credit-anomaly-detector.git](https://github.com/your-username/credit-anomaly-detector.git)
cd credit-anomaly-detector

```

### 2. Set Up a Virtual Environment (Python 3.11 or 3.12 Recommended)

```bash
# Windows
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 4. Dataset Setup

Download the dataset from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Place the extracted `creditcard.csv` file inside the `data/` directory:

```text
credit-anomaly-detector/data/creditcard.csv

```

---

## 🚀 Usage

### Option 1: Run the Headless CLI Pipeline

Executes automated ingestion, model fitting, metric evaluation against ground truth, and exports artifacts (`reports/anomaly_clusters.png` and `models/isolation_forest.joblib`):

```bash
python main.py

```

### Option 2: Launch the Streamlit Web Application

Launches an interactive browser application at `http://localhost:8501`:

```bash
streamlit run app.py

```

#### Streamlit Features:

* **Batch Analysis Tab:** Adjust sample sizes and contamination rates on the fly, inspect metrics, and explore hoverable Plotly 2D PCA projections.
* **Live Inspector Tab:** Simulate real-time transactions by specifying human-readable behavioral conditions (location risk, device verification, velocity spikes) mapped to latent model parameters.

---

## 📊 Evaluation & Diagnostics

Although trained without supervision, the model is evaluated post-hoc using ground truth (`Class`):

* **Confusion Matrix:** Measures True Negatives (correctly identified legitimate transactions), False Positives (normal transactions flagged as fraud), False Negatives (missed frauds), and True Positives (caught frauds).
* **Isolation Score:** Computes continuous decision function values where negative values signify strong anomaly characteristics and positive values indicate nominal inlier behavior.

---

## 📦 Requirements

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
streamlit>=1.28.0
plotly>=5.17.0
joblib>=1.3.0

```markdown
# 💳 Unsupervised Credit Card Fraud & Anomaly Detector

An end-to-end unsupervised machine learning system designed to detect fraudulent credit card transactions without using ground-truth labels during training. The system pairs an automated Command-Line Interface (CLI) batch pipeline with an interactive Streamlit web dashboard for real-time risk assessment and latent space exploration.

```

---

## 📌 Project Overview

In real-world fraud detection, labeled fraud cases are scarce, delayed, or non-existent during model training. This project tackles the challenge through an **unsupervised anomaly detection** framework using an **Isolation Forest** ensemble.

The algorithm operates on two fundamental assumptions:

1. **Rarity:** Anomalies constitute a fraction of the total dataset (< 0.2%).
2. **Geometric Distinctness:** Anomalous transactions deviate noticeably from dense, normal spending clusters in high-dimensional feature space.

---

## 🏗️ System Architecture

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

## 🔬 Core Technical Highlights

* **Unsupervised Pipeline:** Trains on 29 continuous behavioral features ($V_1 \dots V_{28}$ and purchase amount) while withholding ground-truth labels during model fitting.
* **Robust Feature Preprocessing:** Uses `RobustScaler` rather than standard z-score normalization. By scaling via median and Interquartile Range (IQR), extreme spend amounts cannot skew the normalization boundaries.
* **Isolation Forest Ensemble:** Isolates anomalies via random partitioning trees (`iTrees`). Sparse, outlying transactions isolate in significantly shorter average path lengths than dense, regular transactions.
* **Dimensionality Reduction (PCA):** Projects the 29-dimensional feature space into 2 principal components to inspect cluster separation and outlier distributions.
* **Dual-Interface Delivery:**
* **CLI Batch Engine (`main.py`):** Runs automated training runs, records classification reports, outputs static diagnostic plots to `reports/`, and saves model binaries to `models/`.
* **Streamlit Web Application (`app.py`):** Provides an interactive dashboard featuring dynamic parameter sliders, interactive Plotly PCA scatter plots, and a human-readable risk simulation engine.



---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/credit-anomaly-detector.git](https://github.com/your-username/credit-anomaly-detector.git)
cd credit-anomaly-detector

```

### 2. Set Up a Virtual Environment (Python 3.11 or 3.12 Recommended)

```bash
# Windows
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 4. Dataset Setup

Download the dataset from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Place the extracted `creditcard.csv` file inside the `data/` directory:

```text
credit-anomaly-detector/data/creditcard.csv

```

---

## 🚀 Usage

### Option 1: Run the Headless CLI Pipeline

Executes automated ingestion, model fitting, metric evaluation against ground truth, and exports artifacts (`reports/anomaly_clusters.png` and `models/isolation_forest.joblib`):

```bash
python main.py

```

### Option 2: Launch the Streamlit Web Application

Launches an interactive browser application at `http://localhost:8501`:

```bash
streamlit run app.py

```

#### Streamlit Features:

* **Batch Analysis Tab:** Adjust sample sizes and contamination rates on the fly, inspect metrics, and explore hoverable Plotly 2D PCA projections.
* **Live Inspector Tab:** Simulate real-time transactions by specifying human-readable behavioral conditions (location risk, device verification, velocity spikes) mapped to latent model parameters.

---

## 📊 Evaluation & Diagnostics

Although trained without supervision, the model is evaluated post-hoc using ground truth (`Class`):

* **Confusion Matrix:** Measures True Negatives (correctly identified legitimate transactions), False Positives (normal transactions flagged as fraud), False Negatives (missed frauds), and True Positives (caught frauds).
* **Isolation Score:** Computes continuous decision function values where negative values signify strong anomaly characteristics and positive values indicate nominal inlier behavior.

---

## 📦 Requirements

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
streamlit>=1.28.0
plotly>=5.17.0
joblib>=1.3.0

```

```

```