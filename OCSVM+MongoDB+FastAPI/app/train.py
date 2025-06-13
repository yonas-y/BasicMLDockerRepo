"""
Trains the OCSVM model!
"""

from model import create_model, save_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def train_ocsvm(df, label_column="label", normal_class=2):

    # Select only the normal class data!
    if label_column in df.columns:
        df_normal = df[df[label_column] == normal_class]

    df_normal = df_normal.drop(columns=["_id", label_column], errors="ignore")

    # Identify categorical columns
    categorical_cols = df_normal.select_dtypes(include=['object']).columns

    # Encode them (One-hot encoding)
    df_normal = pd.get_dummies(df_normal, columns=categorical_cols)

    # Scale
    scaler = StandardScaler()
    df_normal_scaled = scaler.fit_transform(df_normal)

    model = create_model()
    model.fit(df_normal_scaled)

    save_model(model, scaler)
    print("✅ Model and scaler saved")
