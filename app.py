import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import os

from src.load_data import load_credit_card_data  # for dataset loading
from src.model import AnomalyDetector # for anomaly detection model

# Page Configuration
st.set_page_config(
    page_title="Credit Card Anomaly Detection",
    page_icon="💳",
    layout="wide"
)

st.title("Unsupervised Credit Card Fraud & Anomaly Detector")
st.caption("Powered by Isolation Forest, RobustScaler, and PCA Latent Projection")

# Sidebar Configuration
st.sidebar.header("Pipeline Controls")

sample_size = st.sidebar.slider(
    "Data Sample Size",
    min_value=10000,
    max_value=100000,
    value=50000,
    step=5000,
    help="Higher values take slightly longer to compute PCA."
)

contamination = st.sidebar.slider(
    "Contamination (Expected Outlier %)",
    min_value=0.001,
    max_value=0.010,
    value=0.002,
    step=0.001,
    format="%.3f"
)

# Cache data loading so it does not reload on every UI click
@st.cache_data
def get_cached_data(n_samples):
    return load_credit_card_data(sample_size=n_samples)

with st.spinner("Loading dataset..."):
    raw_df = get_cached_data(sample_size)

# Tab Layout
tab1, tab2 = st.tabs(["📊 Batch Analysis & Latent Space", "🔍 Live Single Transaction Inspector"])

with tab1:
    col_ctrl, col_metrics = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Model Execution")
        st.write(f"Total Transactions: **{len(raw_df):,}**")
        if "Class" in raw_df.columns:
            actual_fraud_count = (raw_df["Class"] == 1).sum()
            st.write(f"Actual Frauds in Sample: **{actual_fraud_count}** ({actual_fraud_count/len(raw_df)*100:.2f}%)")
        
        run_btn = st.button("🚀 Run Isolation Forest", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Fitting Isolation Forest..."):
            detector = AnomalyDetector(contamination=contamination, n_estimators=120)
            X = detector.preprocess(raw_df)
            preds, scores = detector.fit_predict(X)
            
            # Append predictions
            results_df = raw_df.copy()
            results_df['Anomaly_Flag'] = preds
            results_df['Anomaly_Score'] = scores
            # 1 = Anomaly, 0 = Normal
            results_df['Pred_Binary'] = np.where(preds == -1, 1, 0)
            
            flagged = (preds == -1).sum()

        with col_metrics:
            st.subheader("Detection Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Flagged Outliers", f"{flagged:,}")
            m2.metric("Outlier Rate", f"{flagged / len(raw_df) * 100:.2f}%")
            
            if "Class" in results_df.columns:
                cm = confusion_matrix(results_df["Class"], results_df["Pred_Binary"])
                tp = cm[1][1]
                fn = cm[1][0]
                recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
                m3.metric("Fraud Recall", f"{recall:.1f}%")

        # PCA Latent Space Visualization
        st.markdown("---")
        st.subheader("Latent Space Projection (2D PCA)")
        
        with st.spinner("Computing PCA decomposition..."):
            pca = PCA(n_components=2, random_state=42)
            components = pca.fit_transform(X)
            
            plot_df = pd.DataFrame(components, columns=["PC1", "PC2"])
            plot_df["Status"] = np.where(results_df["Anomaly_Flag"] == -1, "Anomaly", "Normal")
            plot_df["Amount"] = results_df["Amount"]
            plot_df["Score"] = results_df["Anomaly_Score"].round(4)
            
            fig = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="Status",
                color_discrete_map={"Normal": "#3b82f6", "Anomaly": "#ef4444"},
                opacity=0.6,
                hover_data=["Amount", "Score"],
                title="PCA Projection of Transaction Space"
            )
            fig.update_layout(height=550, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        # Extreme Outliers Table
        st.subheader("Top Flagged Suspicious Transactions")
        st.dataframe(
            results_df[results_df["Anomaly_Flag"] == -1]
            .sort_values(by="Anomaly_Score")
            [['Amount', 'Anomaly_Score'] + [col for col in ['Class'] if col in results_df.columns]]
            .head(15),
            use_container_width=True
        )

with tab2:
    st.subheader("Simulate Incoming Transaction")
    st.write("Input synthetic behavioral vectors (\(V_1 - V_4\)) and purchase amount to evaluate in real-time.")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        input_amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=50000.0, value=250.0, step=10.0)
    with col_in2:
        input_v1 = st.slider("V1 (Latent Behavioral Vector 1)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    with col_in3:
        input_v2 = st.slider("V2 (Latent Behavioral Vector 2)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)

    eval_btn = st.button("🔍 Evaluate Risk", use_container_width=True)

    if eval_btn:
        # Construct synthetic feature vector
        sample_vec = np.zeros((1, 29))
        sample_vec[0, 0] = (input_amount - 88.34) / 100.0  # Approx median & IQR scale
        sample_vec[0, 1] = input_v1
        sample_vec[0, 2] = input_v2

        # Mock heuristic decision score based on distance from zero
        distance = np.linalg.norm(sample_vec)
        is_risky = distance > 4.5

        if is_risky:
            st.error(f"⚠️ HIGH RISK TRANSACTION! Anomaly Score: {distance:.2f} (Exceeds baseline threshold)")
        else:
            st.success(f"✅ Normal Transaction. Anomaly Score: {distance:.2f} (Within nominal boundaries)")