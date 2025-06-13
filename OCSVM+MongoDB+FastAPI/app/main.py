"""
A function to train an OCSVM model to detect anomalies in network traffic (or similar real-world behavior) using
KDD dataset.
"""

from app.data_loader import load_data
from app.train import train_ocsvm, evaluate_model
from app.model import load_model
from app.data_encoder import encode_data
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

if __name__ == "__main__":
    df = load_data()
    train_ocsvm(df)

    # Load the model and scaler!
    model, scaler = load_model()
    print("✅ Model and scaler loaded!")

    # Evaluate the performance of the model on the training data!
    df_normal = df[df["label"] == 2]

    performance_metric = evaluate_model(df_normal, model, scaler)

    print("✅ Model Performance on the Training Data!\n")
    print(f"Precision: {performance_metric['precision']:.3f}, "
          f"Recall: {performance_metric['recall']:.3f}, "
          f"F1 Score: {performance_metric['f1_score']:.3f}")
