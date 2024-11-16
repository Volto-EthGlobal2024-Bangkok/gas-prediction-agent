import torch

import torch.nn as nn
from get_prediction import load_and_preprocess_data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
sys.path.append('./src/libs')
from connection.connection import db

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

def load_and_preprocess_data(aggregation_points=15, stats="mean"):
    # Read CSV file

    df = db.gasprices.find({"timestamp": {"$gte": new Date(Date.now() - 7*24*60*60*1000)}})
    df = pd.DataFrame(list(df))
    print(df)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp')

    # Aggregate points by taking mean
    df_aggregated = df.groupby(np.arange(len(df))//aggregation_points).agg({
        'gas_price_gwei': stats,
        'timestamp': 'first'
    }).reset_index(drop=True)

    return df_aggregated

def load_model(sequence_length=100, pred_steps=20, aggregation_points=15):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = LSTMPredictor(pred_steps=pred_steps).to(device)

    # Load saved model
    checkpoint = torch.load('./saved_models/gas_price_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def predict(model, sequence_length=100, pred_steps=20, aggregation_points=15):
    # Load and preprocess new data
    df_aggregated = load_and_preprocess_data(aggregation_points)

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

if _name_ == "_main_":
    # Example usage
    predictions = load_model_and_predict()
    print("Predicted next 20 gas prices:", predictions)