"""
Train an OCSVM model to detect anomalies in network traffic (or similar real-world behavior) using
KDD dataset.
"""
# Load the libraries!!!

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Simulate on a real-world data
from sklearn.datasets import fetch_covtype

# Load and prepare data
X, y = fetch_covtype(return_X_y=True)
X = X[:10000]
y = y[:10000]

# Filter only "normal" samples (e.g., class 2 = normal)
X_train = X[y == 2]  # Use only class 2 as normal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train One-Class SVM
model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
model.fit(X_train_scaled)

# Save the model in mounted volume
os.makedirs("output", exist_ok=True)
joblib.dump(model, "output/ocsvm_ntk_traffic_anomaly_d_model.pkl")
joblib.dump(scaler, "output/scaler.pkl")

print("✅ Model and scaler saved in 'model/' folder")
