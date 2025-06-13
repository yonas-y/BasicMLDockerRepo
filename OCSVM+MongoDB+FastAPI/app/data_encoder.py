"""
Encodes the data for scaling and to be used as input for ML models.
"""

import pandas as pd

def encode_data(df_data):
    df_data = df_data.drop(columns=["_id", "label"], errors="ignore")

    # Identify categorical columns
    categorical_cols = df_data.select_dtypes(include=['object']).columns

    # Encode them (One-hot encoding)
    df_data_encoded = pd.get_dummies(df_data, columns=categorical_cols)

    return df_data_encoded