"""
Trains the OCSVM model!
"""

from app.model import create_model, save_model
from app.data_encoder import encode_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

def evaluate_model(df, model, scaler):
    # Encode the data and scale.
    df_encoded = encode_data(df)
    df_scaled = scaler.transform(df_encoded)

    # Predict the model output.
    y_pred = model.predict(df_scaled)
    y_true = np.ones_like(y_pred)  # ground truth = normal

    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "num_samples_used": len(df_scaled)
    }

def train_ocsvm(df, label_column="label", normal_class=2):

    # Select only the normal class data!
    if label_column in df.columns:
        df_normal = df[df[label_column] == normal_class]

    # Encode the categorical features!
    df_normal_encoded = encode_data(df_normal)

    # Scale
    scaler = StandardScaler()
    df_normal_encoded_scaled = scaler.fit_transform(df_normal_encoded)

    model = create_model()
    model.fit(df_normal_encoded_scaled)

    save_model(model, scaler)
    print("✅ Model and scaler saved!")
