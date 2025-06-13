"""
Trains the OCSVM model!
"""

from model import create_model, save_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np


def train_ocsvm(df, label_column="label", normal_class=2):
    if label_column in df.columns:
        df = df[df[label_column] == normal_class]

    X = df.drop(columns=["_id", label_column], errors="ignore")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = create_model()
    model.fit(X_scaled)

    save_model(model, scaler)
    print("✅ Model and scaler saved")
