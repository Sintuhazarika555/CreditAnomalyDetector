import os
import joblib
from load_data import load_credit_card_data
from model import AnomalyDetector

def train_and_save():
    os.makedirs("models",exist_ok=True)

    print("Loading data for offline training...")
    df = load_credit_card_data(sample_size= 100000)

    detector = AnomalyDetector(contamination=0.002,n_estimators=150)
    X = detector.preprocess(df)

    detector.fit_predict(X)

    # Save the detector object (contains scaler + trained IsolationForest)
    model_path = os.path.join("models", "isolation_forest.joblib")
    joblib.dump(detector, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save()