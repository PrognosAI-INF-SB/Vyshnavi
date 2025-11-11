import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from preprocess_data import load_and_preprocess
import os

# --- Create sequences function ---
def create_sequences(data, sensor_cols, window=20):  # Reduced window for speed
    X, y = [], []
    for unit in data['unit_number'].unique():
        unit_data = data[data['unit_number'] == unit]
        for i in range(len(unit_data) - window):
            seq_x = unit_data[sensor_cols].iloc[i:i+window].values
            seq_y = unit_data['RUL'].iloc[i+window]
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)

# --- Train model function ---
def train_and_save_model():
    data, sensor_cols = load_and_preprocess()
    
    if data is None:
        raise ValueError("Data could not be loaded. Check preprocess_data.py")

    # 🔹 Optional: use subset of units for faster debugging
    data = data[data['unit_number'] <= 5]

    # Ensure models folder exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Create sequences
    X, y = create_sequences(data, sensor_cols, window=20)  # Reduced window

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define LSTM model
    model = Sequential([
        LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Checkpoint to save best model
    checkpoint = ModelCheckpoint('models/best_rul_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # Train model (reduced epochs for speed)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,          # 🔹 Reduced epochs for faster run
        batch_size=32,
        callbacks=[checkpoint],
        verbose=1
    )

    # Save final model
    model.save('models/final_rul_model.h5')
    print("✅ Training complete. Models saved in 'models/' folder.")
    
    return model, history, (X_val, y_val)

