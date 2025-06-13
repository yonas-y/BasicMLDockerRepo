"""
Contains Configuration (DB URI, names, paths) for OCSVM+MongoDB+FastAPI.
"""

# app/config.py
MONGO_URI = "mongodb://host.docker.internal:27017"  # or "localhost:27017" if not in Docker
DB_NAME = "NSL-KDD"
COLLECTION_NAME = "TrainData"
MODEL_DIR = "output"
MODEL_PATH = f"{MODEL_DIR}/ocsvm_model.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"