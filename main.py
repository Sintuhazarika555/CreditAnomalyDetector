import os
import sys
import joblib  # for saving/loading model artifacts
from src.load_data import load_credit_card_data  # for loading the dataset
from src.model import AnomalyDetector   # for the Isolation Forest model
from src.visualize import evaluate_predictions, plot_pca_latent_space  # for evaluation and visualization

def run_cli_pipeline(sample_size=70000, contamination=0.002, save_model=True):
    print("=" * 65)
    print("      CREDIT CARD ANOMALY DETECTION — CLI BATCH PIPELINE")
    print("=" * 65)
    
    # 1. Load Data
    print(f"\n[1/5] Loading transaction data (sample: {sample_size:,})...")
    df = load_credit_card_data(sample_size=sample_size)
    y_true = df['Class'].values if 'Class' in df.columns else None

    # 2. Preprocess & Scale
    print("\n[2/5] Preprocessing features with RobustScaler...")
    detector = AnomalyDetector(contamination=contamination, n_estimators=150)
    X = detector.preprocess(df)

    # 3. Fit Unsupervised Isolation Forest
    print("\n[3/5] Fitting Isolation Forest on unlabelled features...")
    predictions, scores = detector.fit_predict(X)

    # 4. Unsupervised Evaluation against ground truth
    print("\n[4/5] Evaluating performance...")
    if y_true is not None:
        evaluate_predictions(y_true, predictions)

    # 5. Export Visualizations and Model Artifacts
    print("\n[5/5] Exporting artifacts...")
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save PCA plot image
    plot_path = os.path.join("reports", "anomaly_clusters.png")
    plot_pca_latent_space(X, predictions, y_true, output_path=plot_path)

    # Save model artifact
    if save_model:
        model_path = os.path.join("models", "isolation_forest.joblib")
        joblib.dump(detector, model_path)
        print(f"  -> Model saved to: {model_path}")

    print("\n" + "=" * 65)
    print("Batch pipeline completed successfully!")
    print("To launch the interactive dashboard, run: streamlit run app.py")
    print("=" * 65)

if __name__ == "__main__":
    run_cli_pipeline()