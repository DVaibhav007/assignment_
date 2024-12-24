import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Prepare the dataset for LSTM
def prepare_lstm_dataset(data, k):
    X, y = [], []
    grouped = data.groupby('part_id')
    
    for _, group in grouped:
        group = group[['temperature_1', 'temperature_2', 'temperature_3', 'temperature_4', 'label']].values
        if len(group) >= k:  # Ensure enough rows for the window
            for i in range(len(group) - k + 1):
                window = group[i:i+k, :-1]  # All temperature columns
                label = group[i + k - 1, -1]  # Label of the kth part
                X.append(window)
                y.append(label)
    
    return np.array(X), np.array(y)

# Load data
data = pd.read_csv("data/data.csv")

# Set window size
k = 5
X_lstm, y_lstm = prepare_lstm_dataset(data, k)

# Normalize features for LSTM
scaler = StandardScaler()
X_lstm = X_lstm.reshape(-1, X_lstm.shape[-1])  # Flatten for scaling
X_lstm = scaler.fit_transform(X_lstm)
X_lstm = X_lstm.reshape(-1, k, X_lstm.shape[-1])  # Reshape back to 3D

# Split into train and test sets
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y_lstm, test_size=0.3, random_state=42, stratify=y_lstm
)

# Build the LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(k, X_lstm.shape[-1]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)
print("LSTM Accuracy:", accuracy_score(y_test_lstm, y_pred_lstm))
