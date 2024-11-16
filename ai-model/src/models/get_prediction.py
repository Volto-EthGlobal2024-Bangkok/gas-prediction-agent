import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import argparse

class GasPriceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

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
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Sort by timestamp to ensure chronological order
    df = df.sort_values('Timestamp')

    # Aggregate points by taking mean
    df_aggregated = df.groupby(np.arange(len(df))//aggregation_points).agg({
        'GasPrice(Gwei)': stats,
        'Timestamp': 'first'
    }).reset_index(drop=True)

    return df_aggregated

def create_sequences(data, seq_length, pred_steps=20, val_split=0.05):
    X, y = [], []
    values = data.flatten() if isinstance(data, np.ndarray) else data['GasPrice(Gwei)'].values
    
    for i in range(len(values) - seq_length - pred_steps):
        X.append(values[i:(i + seq_length)])
        y.append(values[(i + seq_length):(i + seq_length + pred_steps)])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reserve last sequence for testing, no validation
    test_size = 1
    train_size = len(X) - seq_length - test_size
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    best_loss = float('inf')
    patience = 200
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss/len(train_loader)
        
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
    X_train, y_train, _, _, X_test, y_test = create_sequences(scaled_data, sequence_length, pred_steps)
    
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
            train_model(model, train_loader, None, num_epochs=500, device=device)
    else:
        train_model(model, train_loader, None, num_epochs=500, device=device)
    
    # Prepare last sequence for prediction
    last_sequence = torch.FloatTensor(scaled_data[-sequence_length:]).reshape(1, sequence_length, 1).to(device)
    
    # Get the actual values for comparison
    last_actual = scaler.inverse_transform(y_test[-1].reshape(1, -1))
    
    # Modify this section to handle smaller datasets
    num_train_sequences = min(5, len(X_train) // sequence_length)  # Ensure we don't request more sequences than available
    if num_train_sequences == 0:
        num_train_sequences = 1  # Ensure at least one sequence is plotted
    
    # Select non-overlapping sequences by taking every sequence_length'th sequence
    indices = [(-(i + 1) * sequence_length) for i in range(num_train_sequences)]
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
    
    # Create two visualizations
    # 1. Training and test sequences
    plt.figure(figsize=(15, 8))
    
    # Plot training sequences
    for i in range(num_train_sequences):
        offset = i * (pred_steps + 5) #sequence_length  # Use sequence_length instead of (pred_steps + 5)
        x_train = range(offset, offset + pred_steps)
        plt.plot(x_train, train_predictions[i], 'b--', alpha=0.5, marker='o', markersize=4, label='Training Predictions' if i == 0 else None)
        plt.plot(x_train, train_actuals[i], 'g--', alpha=0.5, marker='o', markersize=4, label='Training Actuals' if i == 0 else None)
    
    # Plot test sequence
    x_test = range(offset + pred_steps + 5, offset + 2*pred_steps + 5)
    plt.plot(x_test, test_prediction[0], 'b-', label='Test Prediction', marker='o')
    plt.plot(x_test, last_actual[0], 'r-', label='Test Actual', marker='o')
    
    plt.xlabel('Time Points')
    plt.ylabel('Gas Price (Gwei)')
    plt.title('Predicted vs Actual Gas Prices (Training and Test Sequences)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Test sequence only
    plt.figure(figsize=(15, 8))
    plt.plot(range(pred_steps), test_prediction[0], 'b-', label='Prediction', marker='o')
    plt.plot(range(pred_steps), last_actual[0], 'r-', label='Actual', marker='o')
    plt.xlabel('Time Points')
    plt.ylabel('Gas Price (Gwei)')
    plt.title('Predicted vs Actual Gas Prices (Test Sequence)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_comparison_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
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
