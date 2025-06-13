"""
FastAPI implementation for the OCSVM network traffic anomaly detector!
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import numpy as np
from app.model import load_model
from app.train import train_ocsvm, evaluate_model
from app.data_loader import load_data

app = FastAPI(title="OCSVM Anomaly Detection API")

# Load model and scaler
model, scaler = load_model()

# Request schema
class InferenceInput(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "OCSVM Anomaly Detection API is running"}

@app.post("/predict/")
def predict(features: InferenceInput):
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
        df = load_data()
        df_frac = df[:int(len(df) * data_fraction)]

        # Evaluate the performance of the model on the training data!
        df_normal = df_frac[df_frac["label"] == 2]

        performance_metric = evaluate_model(df_normal, model, scaler)

        return {
            "message": "Model retrained successfully!",
            "data_fraction": data_fraction,
            "metrics": performance_metric
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))