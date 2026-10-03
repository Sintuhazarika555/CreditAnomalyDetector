import pandas as pd 
import os

def load_credit_card_data(sample_size=None,random_state=42):

    csv_path = os.path.join("data","creditcard.csv ")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"Loading data from {csv_path}...")
    df=pd.read_csv(csv_path)

    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} rows from the dataset...")
        df = df.sample(n=sample_size,random_state=random_state).reset_index(drop=True)

    print(f"Loaded {len(df):,} transactions with {df.shape[1]} columns.")
    return df

if __name__ == "__main__":

    # Test our loader with a 50,000 row sample to check speed and structure
    df = load_credit_card_data(sample_size=5000)
    print("\nDataset columns:")
    print(df.columns.tolist()[:10], "... (and remaining V features)")
    print("\nFraud vs Normal distribution in sample:")
    print(df['Class'].value_counts())