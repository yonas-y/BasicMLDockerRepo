"""
FastAPI implementation for the OCSVM network traffic anomaly detector!
"""

# ocsvm_ntk_traffic_fastapi.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np
import joblib
from ocsvm_ntk_traffic_anomaly import train_ocsvm_model

app = FastAPI(title="OCSVM Anomaly Detection API")

def load_model():
    return joblib.load("output/ocsvm_ntk_traffic_anomaly_d_model.pkl")

def load_scaler():
    return joblib.load("output/scaler.pkl")

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Request schema
class InferenceInput(BaseModel):
    features: list

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
def retrain(data_fraction: float= Query(1.0, ge=0.01, le=1.0)):
    try:
        global model, scaler
        model, scaler, metrics = train_ocsvm_model(data_fraction=data_fraction)
        return {
            "message": "Model retrained successfully",
            "data_fraction": data_fraction,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
