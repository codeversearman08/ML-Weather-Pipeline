import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. Data Preparation for Time-Series
# ==========================================
print("Preparing sequential data for LSTM forecasting...")

dataset_path = 'kaggle_daily_weather.csv'
target_feature = 'Temperature'
SEQUENCE_LENGTH = 14 # Use past 14 days to predict the next day

# Note: The original dataset is excluded due to size limits. 
# A synthetic dataset is generated below to demonstrate execution.
if os.path.exists(dataset_path):
    print("Loading original Kaggle dataset...")
    df = pd.read_csv(dataset_path)
    data = df[[target_feature]].values
else:
    print("Notice: Original dataset not found. Generating synthetic temperature time-series...")
    # Generate a synthetic sine wave with noise to simulate seasonal temperature
    time = np.arange(0, 1000)
    data = (20 + 10 * np.sin(time / 30) + np.random.normal(0, 2, 1000)).reshape(-1, 1)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for the LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

# Split chronological data (80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ==========================================
# 2. LSTM Model Architecture
# ==========================================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1) # Linear output for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

# ==========================================
# 3. Training & Evaluation
# ==========================================
if __name__ == "__main__":
    print("\nTraining LSTM Model...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.15, verbose=1)
    
    print("\nEvaluating Model...")
    predictions = model.predict(X_test)
    
    # Inverse transform to get actual temperature values back
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    r2 = r2_score(y_test_inv, predictions_inv)
    
    print(f"\n{'='*40}")
    print(f"--- LSTM Forecasting Results ---")
    print(f"{'='*40}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²:   {r2:.2f}")
