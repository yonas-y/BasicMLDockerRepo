"""
Contains Configuration (DB URI, names, paths) for OCSVM+MongoDB+FastAPI.
"""

# app/config.py
MONGO_URI = "mongodb://192.168.80.1:27017"  # or "localhost:27017" if not in Docker
uri = f"mongodb://{host_ip}:{port}"
client = MongoClient(uri, serverSelectionTimeoutMS=3000)
DB_NAME = "NSL-KDD"
COLLECTION_NAME = "TrainData"
MODEL_DIR = "output"
MODEL_PATH = f"{MODEL_DIR}/ocsvm_model.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"