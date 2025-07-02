import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os

# Step 1: Data Preprocessing
class Data_prepro(Dataset):
    def __init__(self, data, seq_length, prediction_window, state, train_test_rate):
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.scalers = {}
        self.scaled_df = pd.DataFrame(index=data.index)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaled_df[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
            self.scalers[col] = scaler
        self.X, self.Y = self._create_seq(self.scaled_df.values)
        
        if state=='train':
            n = int(len(self.X)*train_test_rate)
            self.X = self.X[:n]
            self.Y = self.Y[:n]
        if state=='test':
            n = int(len(self.X)*train_test_rate)
            self.X = self.X[n:]
            self.Y = self.Y[n:]
            
    def _create_seq(self, data):
        X, Y = [], []
        for i in range (len(data) - self.seq_length - self.prediction_window +1):
            target_idx = i + self.seq_length
            X.append(data[i: target_idx])
            Y.append(data[target_idx: target_idx + self.prediction_window, 3]) # Index 3 is 'Close'
        return torch.FloatTensor(X), torch.FloatTensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
            
    def inverse_transform(self, scaled_data):
        ori_data = self.scalers['Close'].inverse_transform(scaled_data.reshape(-1, 1)).flatten()
        return ori_data

# Step 2: Build the Standard Transformer Model 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=400):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model) # pe as in positional encoding
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # unsqueeze add an extra dimension at position 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # The size of position * div_term is [max_seq_length, d_model/2], Pytorch automatically map it to the even columns of pe
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add position encoding to the input
        # x shape: [batch_size, seq_length, d_model]
        x = x + self.pe[:, :x.size(1), :] # :x.size(1) refers to only take the sequence length of pe
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, n_encoder_layers, dim_feedforward, dropout, seq_length, prediction_window):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)
        self.fc_out = nn.Linear(seq_length, prediction_window)
        
    def forward(self, src):
        src_mask = None # Source mask for transformer, we don't need it here
        embedded = self.embedding(src)
        embedded = self.positional_encoding(embedded)
        transformer_output = self.transformer_encoder(embedded, src_mask)
        output = self.output_layer(transformer_output)
        output = output.squeeze(-1)
        output = self.fc_out(output)
        return output
        
# Step 3: Transformer Model Training and Evaluation
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, scheduler):
    train_losses = []
    val_losses = []
    model = model.to(device)
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            # Backward and optimize
            optimizer.zero_grad() # Clear the previous gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #Gradient clipping
            optimizer.step() # Update parameters with gradients
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader) # Batch based average of the loss
        train_losses.append(train_loss)
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculation
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        # Output losses
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    return train_losses, val_losses

def test_model(model, test_loader, criterion, dataset_test, device):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item()
            # Store predictions and actuals
            predictions.extend(outputs.cpu().detach().numpy())
            actuals.extend(batch_Y.cpu().detach().numpy())
    test_loss = test_loss / len(test_loader)
    # Convert the predictions and actuals back to original scale
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    predic_ori = dataset_test.inverse_transform(predictions)
    actual_ori = dataset_test.inverse_transform(actuals)
    # Metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    print(f'Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    return predic_ori, actual_ori, mse, rmse, mae

def prediction(model, data, dataset_test, seq_length, prediction_window, device):
    model.eval()
    # Get the last sequence of the data from test data
    last_seq = dataset_test.scaled_df.values[-seq_length:]
    last_seq = torch.FloatTensor(last_seq).unsqueeze(0).to(device) # Convert to tensor
    with torch.no_grad():
        prediction = model(last_seq)
    prediction = prediction.cpu().detach().numpy()[0] # Remove batch_size dimension
    prediction_ori = dataset_test.inverse_transform(prediction)
    last_date = data.iloc[-1]['Date']
    print(f'Predictions for the next {prediction_window} days after Date {last_date}:')
    for i, pred in enumerate(prediction_ori):
        print(f'Day {i+1}: {pred:.4f}\n')
        
   
def main(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Hyperparameters
    train_test_rate = 0.85
    train_val_rate = 0.85
    seq_length = 365
    prediction_window = 3
    batch_size = 32
    input_dim = 5 # Date, Low, High, Open, Volume
    d_model = 64
    nhead = 8 # Ori=4
    n_encoder_layers = 5 # Ori=3
    dim_feedforward = 256
    num_epochs = 50
    dropout = 0.2
    # Load and Preprocess data
    data = pd.read_csv(data_path)
    dataset_train = Data_prepro(data, seq_length, prediction_window, state='train', train_test_rate=train_test_rate)
    dataset_test = Data_prepro(data, seq_length, prediction_window, state='test', train_test_rate=train_test_rate)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    train_size = int(train_test_rate * dataset_train.__len__())
    train_indices = list(range(int(train_val_rate * train_size)))
    val_indices = list(range(len(train_indices), len(dataset_train)))
    train_subset = Subset(dataset_train, train_indices)
    val_subset = Subset(dataset_train, val_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    # Create model
    model = TimeSeriesTransformer(
        input_dim, 
        d_model, 
        nhead, 
        n_encoder_layers, 
        dim_feedforward, 
        dropout, 
        seq_length, 
        prediction_window
        )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    # Train model
    print('Training model...')
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        num_epochs, 
        device,
        scheduler
        )
    # Test model
    print('Testing model...')
    predictions, actuals, mse, rmse, mae = test_model(
        model, 
        test_loader, 
        criterion, 
        dataset_test, 
        device
        )
    # Save model
    torch.save(model.state_dict(), 'NASDAQ_transformer.pth')
    print('Model saved as NASDAQ_transformer.pth')
    # Make future prediction
    prediction(
        model, 
        data, 
        dataset_test, 
        seq_length, 
        prediction_window, 
        device
        )
    # Plot results
    plt.figure(figsize=(16, 14))
    
    # 1. Model Loss During Training subplot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training loss', color='blue')
    plt.plot(val_losses, label='Validation loss', color='orange')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Actual vs. Prediction scatter plot
    plt.subplot(2, 2, 2)
    min_val = min(min(predictions), min(actuals))
    max_val = max(max(predictions), max(actuals))
    plt.scatter(actuals, predictions, alpha=0.5, s=10)
    
    # Add y=x line
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', label='y=x')
    
    # Add +/- 3% lines
    plt.plot([min_val, max_val], [min_val*1.03, max_val*1.03], 'k--', label='+3%')
    plt.plot([min_val, max_val], [min_val*0.97, max_val*0.97], 'k--', label='-3%')
    
    plt.title('Actual vs. Prediction')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3 & 4. Raw data vs. Prediction
    # Get the close prices from the original data
    close_prices = data['Close'].values
    
    # Calculate indices for proper alignment
    total_samples = len(close_prices)
    test_start_idx = int(total_samples * train_test_rate)
    
    # Create time indices for original data
    x_indices = list(range(total_samples))
    
    # We need to map test predictions back to the original timeline
    # Each prediction corresponds to a specific point after sequence_length
    pred_indices = []
    actual_indices = []
    
    # Calculate starting point for predictions
    # This is where the test data begins + sequence_length (as we need sequence_length data points to make first prediction)
    pred_start = test_start_idx + seq_length
    
    # Generate prediction indices
    for i in range(len(predictions)):
        # The prediction index is offset by the sequence length since predictions 
        # are for the day after the sequence
        pred_idx = pred_start + i
        if pred_idx < total_samples:  # Ensure we don't go out of bounds
            pred_indices.append(pred_idx)
            actual_indices.append(pred_idx)
    
    # Limit predictions to the available indices
    predictions = predictions[:len(pred_indices)]
    actuals_aligned = actuals[:len(actual_indices)]
    
    # 3. Full Raw data vs. Prediction
    plt.subplot(2, 2, 3)
    plt.plot(x_indices, close_prices, label='Raw data', color='blue')
    plt.plot(pred_indices, predictions, label='Predicted data', color='orange')
    plt.title('Raw data vs. Prediction')
    plt.xlabel('Duration (day)')
    plt.ylabel('Close Price ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Zoom in Raw data vs. Prediction
    # Focus on the test data portion
    plt.subplot(2, 2, 4)
    
    # Set zoom range to focus on test data
    # We'll show some of the training data for context if possible
    context_days = 200  # Show some days before test data for context
    start_idx = max(0, test_start_idx - context_days)
    
    plt.plot(x_indices[start_idx:], close_prices[start_idx:], label='Raw data', color='blue')
    plt.plot(pred_indices, predictions, label='Predicted data', color='orange')
    plt.title('Zoom in Raw data vs. Prediction')
    plt.xlabel('Duration (day)')
    plt.ylabel('Close Price ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('nasdaq_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == '__main__':
    data_path = 'C://Users//JunmingLao//Desktop//Transformer based stock prediction//yahoo NASDAQ historical data 5y.csv'
    main(data_path)