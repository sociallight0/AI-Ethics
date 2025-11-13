# utils.py
import pandas as pd

def preprocess_compas_data(input_path, output_path):
    """Preprocess COMPAS dataset for fairness audit."""
    df = pd.read_csv(input_path)
    # Select relevant columns and handle missing values
    df = df[['sex', 'race', 'decile_score', 'two_year_recid']].dropna()
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
