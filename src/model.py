import numpy as np 
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination = 0.002 ,n_estimators=150,random_state=42):

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=256,  # 256 samples per tree is optimal for isolation forests
            random_state=self.random_state,
            n_jobs=-1   # Use all CPU cores for fast parallel training
        )

    def preprocess(self,df):

        df_copy = df.copy()

        # Scaling
        df_copy['Scaled_Amount'] = self.scaler.fit_transform(df_copy[['Amount']])

        # Features to drop
        features_to_drop = [col for col in ['Time','Amount','Class'] if col in df_copy.columns]
        X = df_copy.drop(columns=features_to_drop)

        return X

    def fit_predict(self,X):
        
        print(f"Training Isolation Forest with {self.n_estimators} trees on {X.shape[0]:,} samples...")
        self.model.fit(X)

        prediction = self.model.predict(X)

        scores = self.model.decision_function(X)

        return prediction, scores

if __name__ == "__main__":

    from load_data import load_credit_card_data

    #loading data
    df = load_credit_card_data(sample_size= 60000)

    #initializing the anomaly detector
    detector = AnomalyDetector(contamination = 0.002)

    #preprocess
    X = detector.preprocess(df)
    # supervised L : X and Y (labels) are used to train the model, while in unsupervised L, only X is used to train the model.

    # fit and predict
    preds , scores = detector.fit_predict(X)

    # Add predictions back to dataframe for inspection
    df['Anomaly_Flag'] = preds
    df['Anomaly_Score'] = scores

    n_flagged = (preds == -1).sum()
    print(f"\nModel training complete.")
    print(f"Total transactions flagged as anomalies: {n_flagged}")
    print("\nTop 5 most extreme anomalies detected:")
    print(df[df['Anomaly_Flag'] == -1][['Amount', 'Anomaly_Score', 'Class']].sort_values(by='Anomaly_Score').head())