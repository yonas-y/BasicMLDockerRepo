"""
Trains the OCSVM model!
"""

from model import create_model, save_model
from data_encoder import encode_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


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
    print("✅ Model and scaler saved")
