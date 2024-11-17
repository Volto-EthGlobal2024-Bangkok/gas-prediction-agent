import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import argparse
import json

class GasPriceDataset(Dataset):
    """Custom Dataset for gas price sequences and their corresponding targets."""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMPredictor(nn.Module):
    """LSTM model for predicting gas prices.
    
    Args:
        input_size (int): Number of input features (default: 1 for univariate time series)
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of LSTM layers
        pred_steps (int): Number of future steps to predict
    """
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, pred_steps=20):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, pred_steps)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        x = self.relu(x)
        return self.fc2(x)

def load_and_preprocess_data(aggregation_points=30):
    """Load and preprocess gas price data.
    
    Args:
        aggregation_points (int): Number of points to aggregate together
    
    Returns:
        DataFrame: Preprocessed gas price data
    """
    df = pd.read_csv('gas_prices.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    # Aggregate points by taking mean
    df_aggregated = df.groupby(np.arange(len(df))//aggregation_points).agg({
        'GasPrice(Gwei)': 'mean',
        'Timestamp': 'first'
    }).reset_index(drop=True)
    
    return df_aggregated

def create_sequences(data, seq_length, pred_steps=20, val_split=0.05, step=5):
    X, y = [], []
    values = data.flatten() if isinstance(data, np.ndarray) else data['GasPrice(Gwei)'].values
    
    for i in range(0, len(values) - seq_length - pred_steps, step):
        X.append(values[i:(i + seq_length)])
        y.append(values[(i + seq_length):(i + seq_length + pred_steps)])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reserve last sequence for testing, no validation
    test_size = 1
    train_size = len(X) - pred_steps - test_size
    
    print(f"Total sequences: {len(X)}")
    print(f"Train/Test split: {train_size}/{test_size}")
    
    # Split chronologically, keeping most recent data for testing
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[-test_size:]  # Last sequence
    y_test = y[-test_size:]  # Last sequence
    
    print(f"Training data shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Return None for validation data
    return X_train, y_train, None, None, X_test, y_test

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    
    best_loss = float('inf')
    patience = 500
    patience_counter = 0
    
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss/len(train_loader)
        scheduler.step(avg_train_loss)
        
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
            save_path = 'saved_models'
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, os.path.join(save_path, 'gas_price_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

def load_saved_model(model, device):
    save_path = os.path.join('saved_models', 'gas_price_model.pth')
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded saved model")
        return True
    return False

def predict_gas_prices(use_saved_model=False, sequence_length=100, aggregation_points=15, pred_steps=20):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    df_aggregated = load_and_preprocess_data(aggregation_points)
    
    # Add debug print statements
    print(f"Length of aggregated data: {len(df_aggregated)}")
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_aggregated['GasPrice(Gwei)'].values.reshape(-1, 1))
    
    # Create sequences using the specified sequence length and prediction steps
    X_train, y_train, _, _, X_test, y_test = create_sequences(scaled_data, sequence_length, pred_steps, step=10)
    
    print(f"Shape of X: {X_train.shape}")  # Debug print
    print(f"Shape of y: {y_train.shape}")  # Debug print
    
    # Create dataset for training only
    train_dataset = GasPriceDataset(
        sequences=X_train.reshape(-1, sequence_length, 1),
        targets=y_train
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model with modified architecture and pred_steps
    model = LSTMPredictor(pred_steps=pred_steps).to(device)
    
    if use_saved_model:
        model_loaded = load_saved_model(model, device)
        if not model_loaded:
            print("No saved model found. Training new model...")
            train_model(model, train_loader, None, num_epochs=1000, device=device)
    else:
        train_model(model, train_loader, None, num_epochs=1000, device=device)
    
    # Prepare last sequence for prediction
    last_sequence = torch.FloatTensor(scaled_data[-sequence_length:]).reshape(1, sequence_length, 1).to(device)
    
    # Get the actual values for comparison
    last_actual = scaler.inverse_transform(y_test[-1].reshape(1, -1))
    
    # Modify sequence selection to get 10 sequences
    num_sequences = min(10, len(X_train))  # Changed from 5 to 10
    step = max(1, (len(X_train) - 1) // (num_sequences - 1)) if num_sequences > 1 else 1
    indices = range(0, len(X_train), step)[:num_sequences]
    
    train_sequences = torch.FloatTensor(X_train[indices]).reshape(-1, sequence_length, 1).to(device)
    train_actuals = y_train[indices]
    
    model.eval()
    with torch.no_grad():
        # Get predictions for training sequences
        train_predictions = model(train_sequences).cpu().numpy()
        # Get prediction for test sequence
        test_prediction = model(last_sequence).cpu().numpy()
    
    # Inverse transform predictions and actuals
    train_predictions = scaler.inverse_transform(train_predictions)
    train_actuals = scaler.inverse_transform(train_actuals)
    test_prediction = scaler.inverse_transform(test_prediction)
    
    # Get timestamps for plotting
    timestamps = df_aggregated['Timestamp'].values
    last_timestamp = timestamps[-1]
    # Calculate the time delta between aggregated points
    time_delta = (timestamps[-1] - timestamps[-2])
    future_timestamps = [timestamps[-1] + (i + 1) * time_delta for i in range(pred_steps)]
    future_timestamps = pd.DatetimeIndex(future_timestamps)

    # Calculate standard deviations for the last 8 steps
    last_8_values = scaled_data[-8:].flatten()
    std_dev = np.std(last_8_values)
    std_dev_unscaled = float(scaler.inverse_transform([[std_dev]])[0][0])
    
    # Add confidence intervals (±1 standard deviation)
    upper_bound = test_prediction[0] + std_dev_unscaled
    lower_bound = test_prediction[0] - std_dev_unscaled

    # Prepare data for JSON export
    prediction_data = {
        'training_sequences': [],
        'test_sequence': {
            'timestamps': [str(ts) for ts in future_timestamps],
            'predictions': test_prediction[0].tolist(),
            'actuals': last_actual[0].tolist(),
            'upper_bound': upper_bound.tolist(),
            'lower_bound': lower_bound.tolist(),
            'standard_deviation': std_dev_unscaled
        }
    }

    # Create separate plots for each training sequence and save data
    for i in range(len(train_sequences)):
        idx = indices[i]
        sequence_timestamps = timestamps[idx + sequence_length:idx + sequence_length + pred_steps]
        
        # Calculate standard deviation for this sequence
        sequence_data = scaled_data[idx:idx+8].flatten()  # Get 8 points from this sequence
        std_dev = np.std(sequence_data)
        std_dev_unscaled = float(scaler.inverse_transform([[std_dev]])[0][0])
        
        # Calculate confidence intervals
        upper_bound = train_predictions[i] + std_dev_unscaled
        lower_bound = train_predictions[i] - std_dev_unscaled
        
        # Save plot with confidence intervals
        plt.figure(figsize=(15, 8))
        plt.plot(sequence_timestamps, train_predictions[i], 'b--', label='Prediction', 
                marker='o', markersize=4)
        plt.plot(sequence_timestamps, train_actuals[i], 'g--', label='Actual', 
                marker='o', markersize=4)
        plt.fill_between(sequence_timestamps, lower_bound, upper_bound, 
                        color='blue', alpha=0.2, label='±1 Standard Deviation')
    
        plt.title(f'Training Sequence {i+1} - Prediction vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Gas Price (Gwei)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'prediction_comparison_training_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save sequence data to dictionary
        prediction_data['training_sequences'].append({
            'sequence_number': i + 1,
            'timestamps': [str(ts) for ts in sequence_timestamps],
            'predictions': train_predictions[i].tolist(),
            'actuals': train_actuals[i].tolist(),
            'upper_bound': upper_bound.tolist(),
            'lower_bound': lower_bound.tolist(),
            'standard_deviation': float(std_dev_unscaled)
        })

    # Update test sequence plot to include confidence intervals
    plt.figure(figsize=(15, 8))
    plt.plot(future_timestamps, test_prediction[0], 'b-', label='Prediction', marker='o')
    plt.plot(future_timestamps, last_actual[0], 'r-', label='Actual', marker='o')
    plt.fill_between(future_timestamps, lower_bound, upper_bound, 
                     color='blue', alpha=0.2, label='±1 Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Gas Price (Gwei)')
    plt.title('Predicted vs Actual Gas Prices (Test Sequence)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_comparison_test.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save all prediction data to JSON file
    with open('prediction_data.json', 'w') as f:
        json.dump(prediction_data, f, indent=4)
    
    return test_prediction[0]

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Predict gas prices using LSTM model')
    parser.add_argument('--sequence-length', type=int, default=100,
                        help='Length of input sequence (default: 100)')
    parser.add_argument('--aggregation-points', type=int, default=15,
                        help='Number of points to aggregate (default: 15)')
    parser.add_argument('--pred-steps', type=int, default=20,
                        help='Number of steps to predict (default: 20)')
    parser.add_argument('--use-saved-model', action='store_true',
                        help='Use saved model instead of training new one')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Example usage with parsed arguments
    predictions = predict_gas_prices(
        use_saved_model=args.use_saved_model,
        sequence_length=args.sequence_length,
        aggregation_points=args.aggregation_points,
        pred_steps=args.pred_steps
    )
    print(f"Predicted next {args.pred_steps} gas prices:", predictions)
