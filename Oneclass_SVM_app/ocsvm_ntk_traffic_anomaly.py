"""
A function to train an OCSVM model to detect anomalies in network traffic (or similar real-world behavior) using
KDD dataset.
"""
# Load the libraries!!!
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os

# Simulate on a real-world data
from sklearn.datasets import fetch_covtype

def train_ocsvm_model(data_fraction: float = 1.0):
    # Load and prepare data
    X, y = fetch_covtype(return_X_y=True)

    # Limit size based on fraction
    total_samples = min(10000, len(X))  # safety cap for speed
    num_samples = int(data_fraction * total_samples)
    X = X[:num_samples]
    y = y[:num_samples]

    # Use only "normal" class for training
    X_train = X[y == 2]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    model.fit(X_train_scaled)

    # Evaluate on training data
    y_pred = model.predict(X_train_scaled)
    y_true = np.ones_like(y_pred)  # ground truth = normal

    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    # Save
    os.makedirs("output", exist_ok=True)
    joblib.dump(model, "output/ocsvm_ntk_traffic_anomaly_d_model.pkl")
    joblib.dump(scaler, "output/scaler.pkl")

    return model, scaler, {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "num_samples_used": len(X_train)
    }
