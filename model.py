#!/usr/bin/env python3

# ============================================
# Install & Import Dependencies
# ============================================
# !pip install scipy pandas

import pandas as pd
import numpy as np
from scipy.io import loadmat
from datetime import datetime, timedelta

# ============================================
# Helper Function: MATLAB datenum → datetime
# ============================================
def matlab2datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) \
           + timedelta(days=matlab_datenum % 1) \
           - timedelta(days=366)

# ============================================s
# Load .mat Dataset
# ============================================
data = loadmat('NEUSTG_19502020_12stations.mat')

lat = data['lattg'].flatten()
lon = data['lontg'].flatten()
sea_level = data['sltg']
station_names = [s[0] for s in data['sname'].flatten()]
time = data['t'].flatten()
time_dt = np.array([matlab2datetime(t) for t in time])

# ============================================
# Select Target Stations
# ============================================
SELECTED_STATIONS = [
    'Annapolis', 'Atlantic_City', 'Charleston', 'Washington', 'Wilmington'
]

selected_idx = [station_names.index(st) for st in SELECTED_STATIONS]
selected_names = [station_names[i] for i in selected_idx]
selected_lat = lat[selected_idx]
selected_lon = lon[selected_idx]
selected_sea_level = sea_level[:, selected_idx]  # time × selected_stations

# ============================================
# Build Preview DataFrame
# ============================================
df_preview = pd.DataFrame({
    'time': np.tile(time_dt[:5], len(selected_names)),
    'station_name': np.repeat(selected_names, 5),
    'latitude': np.repeat(selected_lat, 5),
    'longitude': np.repeat(selected_lon, 5),
    'sea_level': selected_sea_level[:5, :].T.flatten()
})

# ============================================
# Print Data Head
# ============================================
print(f"Number of stations: {len(selected_names)}")
print(f"Sea level shape (time x stations): {selected_sea_level.shape}")
df_preview.head()

# ============================================
# Convert Hourly → Daily per Station
# ============================================
# Convert time to pandas datetime
time_dt = pd.to_datetime(time_dt)

# Build hourly DataFrame for selected stations
df_hourly = pd.DataFrame({
    'time': np.tile(time_dt, len(selected_names)),
    'station_name': np.repeat(selected_names, len(time_dt)),
    'latitude': np.repeat(selected_lat, len(time_dt)),
    'longitude': np.repeat(selected_lon, len(time_dt)),
    'sea_level': selected_sea_level.flatten()
})

# ============================================
# Compute Flood Threshold per Station
# ============================================
threshold_df = df_hourly.groupby('station_name')['sea_level'].agg(['mean','std']).reset_index()
threshold_df['flood_threshold'] = threshold_df['mean'] + 1.5 * threshold_df['std']

df_hourly = df_hourly.merge(threshold_df[['station_name','flood_threshold']], on='station_name', how='left')

# ============================================
# Daily Aggregation + Flood Flag
# ============================================
df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
    'sea_level': 'mean',
    'latitude': 'first',
    'longitude': 'first',
    'flood_threshold': 'first'
}).reset_index()

# Flood flag: 1 if any hourly value exceeded threshold that day
hourly_max = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')])['sea_level'].max().reset_index()
df_daily = df_daily.merge(hourly_max, on=['station_name','time'], suffixes=('','_max'))
df_daily['flood'] = (df_daily['sea_level_max'] > df_daily['flood_threshold']).astype(int)

# ============================================
# Feature Engineering (3d & 7d means)
# ============================================
df_daily['sea_level_3d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(
    lambda x: x.rolling(3, min_periods=1).mean())
df_daily['sea_level_7d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(
    lambda x: x.rolling(7, min_periods=1).mean())

# Preview
df_daily.head()

# ============================================
# LSTM Approach with PyTorch
# ============================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*50}")
print(f"Using device: {device}")
print(f"{'='*50}\n")

# ============================================
# Reshape Data for LSTM (Sequence Format)
# ============================================
FEATURES = ['sea_level', 'sea_level_3d_mean', 'sea_level_7d_mean']
HIST_DAYS = 7
FUTURE_DAYS = 14

X_train_lstm, y_train_lstm = [], []

for stn, grp in df_daily.groupby('station_name'):
    grp = grp.sort_values('time').reset_index(drop=True)
    for i in range(len(grp) - HIST_DAYS - FUTURE_DAYS):
        # Keep sequence structure: (7 days, 3 features)
        hist = grp.loc[i:i+HIST_DAYS-1, FEATURES].values  # Shape: (7, 3)
        future = grp.loc[i+HIST_DAYS:i+HIST_DAYS+FUTURE_DAYS-1, 'flood'].values
        X_train_lstm.append(hist)
        y_train_lstm.append(future)

X_train_lstm = np.array(X_train_lstm)  # Shape: (samples, 7, 3)
y_train_lstm = np.array(y_train_lstm)  # Shape: (samples, 14)

print(f"LSTM Training Data Shapes:")
print(f"X_train_lstm: {X_train_lstm.shape} (samples, sequence_length, features)")
print(f"y_train_lstm: {y_train_lstm.shape} (samples, future_days)\n")

# Normalize features for LSTM
scaler = StandardScaler()
n_samples, n_timesteps, n_features = X_train_lstm.shape
X_train_lstm_reshaped = X_train_lstm.reshape(-1, n_features)
X_train_lstm_scaled = scaler.fit_transform(X_train_lstm_reshaped)
X_train_lstm = X_train_lstm_scaled.reshape(n_samples, n_timesteps, n_features)

# Prepare test data for LSTM
X_test_lstm = []
for stn, grp in df_daily.groupby('station_name'):
    mask = (grp['time'] >= hist_start) & (grp['time'] <= hist_end)
    hist_block = grp.loc[mask, FEATURES].values
    if len(hist_block) == 7:  # ensure full 7-day block
        X_test_lstm.append(hist_block)

X_test_lstm = np.array(X_test_lstm)  # Shape: (stations, 7, 3)

# Normalize test data with same scaler
n_test_samples, n_test_timesteps, n_test_features = X_test_lstm.shape
X_test_lstm_reshaped = X_test_lstm.reshape(-1, n_test_features)
X_test_lstm_scaled = scaler.transform(X_test_lstm_reshaped)
X_test_lstm = X_test_lstm_scaled.reshape(n_test_samples, n_test_timesteps, n_test_features)

print(f"X_test_lstm shape: {X_test_lstm.shape}\n")

# ============================================
# PyTorch Dataset Class
# ============================================
class FloodDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets and data loaders
train_dataset = FloodDataset(X_train_lstm, y_train_lstm)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ============================================
# LSTM Model Architecture
# ============================================
class FloodLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=4, output_size=14, dropout=0.2):
        super(FloodLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # Shape: (batch, hidden_size)
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # Output probabilities for each of 14 days
        
        return out

# ============================================
# Initialize Model, Loss, Optimizer
# ============================================
model = FloodLSTM(
    input_size=3,
    hidden_size=64,
    num_layers=2,
    output_size=14,
    dropout=0.2
).to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

print(f"Model Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ============================================
# Training Loop
# ============================================
num_epochs = 30
best_loss = float('inf')
patience = 10
patience_counter = 0

print("Starting LSTM Training...")
print(f"{'Epoch':<8} {'Train Loss':<12} {'LR':<10}")
print("-" * 35)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    scheduler.step(avg_loss)
    
    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1:<8} {avg_loss:<12.6f} {current_lr:<10.6f}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print("\nTraining completed!")

# Load best model
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()

# ============================================
# LSTM Predictions
# ============================================
print("\n" + "="*50)
print("LSTM Predictions")
print("="*50 + "\n")

X_test_tensor = torch.FloatTensor(X_test_lstm).to(device)

with torch.no_grad():
    y_pred_lstm = model(X_test_tensor).cpu().numpy()

y_pred_lstm_bin = (y_pred_lstm > 0.5).astype(int)

print(f"LSTM Predictions shape: {y_pred_lstm.shape}")
print(f"Prediction probabilities (first station, first 5 days): {y_pred_lstm[0, :5]}")
print(f"Binary predictions (first station, first 5 days): {y_pred_lstm_bin[0, :5]}\n")

# ============================================
# LSTM Evaluation
# ============================================
y_true_flat_lstm = y_true.flatten()
y_pred_flat_lstm = y_pred_lstm_bin.flatten()

tn_lstm, fp_lstm, fn_lstm, tp_lstm = confusion_matrix(y_true_flat_lstm, y_pred_flat_lstm).ravel()
acc_lstm = accuracy_score(y_true_flat_lstm, y_pred_flat_lstm)
f1_lstm = f1_score(y_true_flat_lstm, y_pred_flat_lstm)
mcc_lstm = matthews_corrcoef(y_true_flat_lstm, y_pred_flat_lstm)

print("="*50)
print("LSTM Results")
print("="*50)
print("\n=== Confusion Matrix ===")
print(f"TP: {tp_lstm} | FP: {fp_lstm} | TN: {tn_lstm} | FN: {fn_lstm}")
print("\n=== Metrics ===")
print(f"Accuracy: {acc_lstm:.3f}")
print(f"F1 Score: {f1_lstm:.3f}")
print(f"MCC: {mcc_lstm:.3f}")

# ============================================
# Comparison: XGBoost vs LSTM
# ============================================
print("\n" + "="*50)
print("Model Comparison: XGBoost vs LSTM")
print("="*50)
print(f"\n{'Metric':<15} {'XGBoost':<12} {'LSTM':<12} {'Difference':<12}")
print("-" * 50)
print(f"{'Accuracy':<15} {acc:<12.3f} {acc_lstm:<12.3f} {acc_lstm-acc:<+12.3f}")
print(f"{'F1 Score':<15} {f1:<12.3f} {f1_lstm:<12.3f} {f1_lstm-f1:<+12.3f}")
print(f"{'MCC':<15} {mcc:<12.3f} {mcc_lstm:<12.3f} {mcc_lstm-mcc:<+12.3f}")
print(f"\n{'True Positives':<15} {tp:<12} {tp_lstm:<12} {tp_lstm-tp:<+12}")
print(f"{'False Positives':<15} {fp:<12} {fp_lstm:<12} {fp_lstm-fp:<+12}")
print(f"{'True Negatives':<15} {tn:<12} {tn_lstm:<12} {tn_lstm-tn:<+12}")
print(f"{'False Negatives':<15} {fn:<12} {fn_lstm:<12} {fn_lstm-fn:<+12}")

# Save LSTM predictions for further analysis
print("\n" + "="*50)
print("Saved files:")
print("  - best_lstm_model.pth (PyTorch model state)")
print("="*50)