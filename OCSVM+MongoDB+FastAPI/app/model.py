"""
Creates, Saves and Loads the Model.
"""

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
from config import MODEL_PATH, SCALER_PATH, MODEL_DIR
import os

def create_model():
    return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)

def save_model(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
