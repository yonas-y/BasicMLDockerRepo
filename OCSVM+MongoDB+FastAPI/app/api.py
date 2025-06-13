"""
FastAPI implementation for the OCSVM network traffic anomaly detector!
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import numpy as np
from model import load_model

app = FastAPI(title="OCSVM Anomaly Detection API")

# Load model and scaler
model, scaler = load_model()

# Request schema
class InferenceInput(BaseModel):
    features: list

@app.post("/predict/")
def predict(features: InferenceInput):
    try:
        X = np.array(features.data).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        return {"prediction": "normal" if pred == 1 else "anomaly"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
