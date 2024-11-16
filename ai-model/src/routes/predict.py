# from fastapi import APIRouter, FastAPI
# import app from main
#
# router = APIRouter(prefix="/api/v1/predict", tags=["predict"])
#
# @router.get("/", tags=["predict"])
# async def read_users():
#     return [{"username": "Rick"}, {"username": "Morty"}]
#
# app.include_router(router)

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import sys
sys.path.append('./src/libs')
from connection.connection import db
from datetime import datetime, timedelta

device = "cpu"

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=4, pred_steps=20):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, pred_steps)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

async def load_and_preprocess_data(aggregation_points=15, stats="mean"):
    # Read CSV file
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    df = db.gasprices.find()
    # df = db.gasprices.find({"timestamp": {"$gte": seven_days_ago}})
    df = await df.to_list(length=None)
#     print(df)

    df = pd.DataFrame(list(df))
#     print(df)
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
#     print(df)

    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp')

    # Aggregate points by taking mean
    df_aggregated = df.groupby(np.arange(len(df))//aggregation_points).agg({
        'gas_price_gwei': stats,
        'timestamp': 'first'
    }).reset_index(drop=True)

    return df_aggregated

def load_model(sequence_length=100, pred_steps=8, aggregation_points=30):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = LSTMPredictor(pred_steps=pred_steps).to(device)

    # Load saved model
    checkpoint = torch.load('./src/models/old_models/gas_price_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

async def predict(model, sequence_length=17, pred_steps=8, aggregation_points=30):
    # Load and preprocess new data
    df_aggregated = await load_and_preprocess_data(aggregation_points)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_aggregated['gas_price_gwei'].values.reshape(-1, 1))
    # Prepare last sequence for prediction
    last_sequence = torch.FloatTensor(scaled_data[-sequence_length:]).reshape(1, sequence_length, 1).to(device)
    # Get prediction
    with torch.no_grad():
        prediction = model(last_sequence).cpu().numpy()

    # Inverse transform prediction
    prediction = scaler.inverse_transform(prediction)

    return prediction[0]

if __name__ == "__main__":
    # Example usage
    model = load_model()
    prediction = predict(model)
    print("Predicted next 20 gas prices:", prediction)