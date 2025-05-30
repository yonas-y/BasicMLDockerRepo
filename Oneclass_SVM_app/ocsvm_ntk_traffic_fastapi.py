# fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_covtype

app = FastAPI(title="OCSVM Anomaly Detection API")

# Global variables for the model and scaler
model = None
scaler = None

MODEL_PATH = "output/ocsvm_ntk_traffic_anomaly_d_model.pkl"
SCALER_PATH = "output/scaler.pkl"

# Load model and scaler
def load_model():
    global model, scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# Initial load
load_model()

# Request schema
class Features(BaseModel):
    data: list

@app.post("/predict/")
def predict(features: Features):
    try:
        X = np.array(features.data).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        return {"prediction": "normal" if pred == 1 else "anomaly"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain/")
def retrain():
    try:
        # Load and prepare new training data
        X, y = fetch_covtype(return_X_y=True)
        X = X[:10000]
        y = y[:10000]
        X_train = X[y == 2]  # Assume class 2 is normal

        # Retrain
        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)

        new_model = OneClassSVM(kernel="rbf", nu=0.05, gamma='scale')
        new_model.fit(X_train_scaled)

        # Save
        os.makedirs("output", exist_ok=True)
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)

        # Reload in memory
        load_model()

        return {"status": "Model retrained and reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
