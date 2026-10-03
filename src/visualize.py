import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def evaluate_predictions(y_true, y_pred_raw):
    """
    Evaluates unsupervised model predictions against ground truth labels.
    y_pred_raw: IsolationForest predictions (1 for normal, -1 for anomaly)
    y_true: Ground truth labels (0 for normal, 1 for fraud)
    """
    # Convert: -1 (anomaly) -> 1, and 1 (normal) -> 0
    y_pred_binary = np.where(y_pred_raw == -1, 1, 0)
    
    print("=" * 55)
    print("           UNSUPERVISED EVALUATION REPORT")
    print("=" * 55)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    print("\nConfusion Matrix:")
    print(f"  [TN] Legitimate marked Normal   : {cm[0][0]:,}")
    print(f"  [FP] Legitimate flagged as Fraud: {cm[0][1]:,}")
    print(f"  [FN] Fraud missed (False Normal): {cm[1][0]:,}")
    print(f"  [TP] Fraud caught (True Positive): {cm[1][1]:,}")
    
    print("\nClassification Metrics:")
    print(classification_report(y_true, y_pred_binary, target_names=["Normal (0)", "Fraud (1)"]))
    
    return y_pred_binary


def plot_pca_latent_space(X, y_pred_raw, y_true=None, output_path="reports/anomaly_clusters.png"):
    """
    Projects high-dimensional feature space to 2D via PCA and plots anomalies.
    """
    print("Computing 2D PCA projection for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_ * 100
    print(f"Explained Variance: PC1 = {explained_var[0]:.2f}%, PC2 = {explained_var[1]:.2f}%")
    
    plt.figure(figsize=(10, 6))
    
    # Normal points (pred == 1)
    normal_mask = (y_pred_raw == 1)
    plt.scatter(
        X_pca[normal_mask, 0], 
        X_pca[normal_mask, 1], 
        c='#3b82f6', 
        alpha=0.25, 
        s=12, 
        label='Predicted Normal'
    )
    
    # Flagged Anomaly points (pred == -1)
    anomaly_mask = (y_pred_raw == -1)
    plt.scatter(
        X_pca[anomaly_mask, 0], 
        X_pca[anomaly_mask, 1], 
        c='#ef4444', 
        alpha=0.85, 
        s=30, 
        edgecolors='black',
        linewidth=0.5,
        label='Flagged Anomaly'
    )
    
    plt.title("Isolation Forest Anomaly Detection (2D PCA Projection)", fontsize=13, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({explained_var[0]:.1f}% var)")
    plt.ylabel(f"Principal Component 2 ({explained_var[1]:.1f}% var)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved PCA visualization plot to '{output_path}'.")
    plt.close()