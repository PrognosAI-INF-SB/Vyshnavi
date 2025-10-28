import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- GLOBAL CONSTANTS ---
TIME_STEPS = 30     
EPOCHS = 30         
BATCH_SIZE = 32
RANDOM_SEED = 42

# Alert Thresholds (in Days/Cycles) - Milestone 4
CRITICAL_RUL = 2.0  # RUL < 2.0
WARNING_RUL = 5.0   # 2.0 <= RUL < 5.0

# --- 1. Data Loading and Preprocessing ---
def load_and_preprocess_data(file_path="sensor_data.csv"):
    """Loads, cleans, scales, and creates sequences."""
    df = pd.read_csv(file_path)

    sensor_cols = ['SensorA', 'SensorB', 'SensorC']
    
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['SensorA'] = df['SensorA'].fillna(df['SensorA'].mean())
    df['SensorB'] = df['SensorB'].fillna(df['SensorB'].mean())
    df['SensorC'] = df['SensorC'].fillna(df['SensorC'].mean())

    sensor_data = df[sensor_cols].values.astype('float32')

    # Scaling Features
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_X.fit_transform(sensor_data)

    # RUL Target Generation (Simple reverse count)
    total_samples = len(scaled_data)
    rul_data = np.arange(total_samples)[::-1] 
    
    # Scaling Target
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    rul_data_scaled = scaler_y.fit_transform(rul_data.reshape(-1, 1)).flatten()

    # Create sequences (X) and corresponding RUL target (y)
    X, y = [], []
    for i in range(len(scaled_data) - TIME_STEPS):
        X.append(scaled_data[i:(i + TIME_STEPS), :])
        y.append(rul_data_scaled[i + TIME_STEPS])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
    )

    return X_train, y_train, X_test, y_test, scaler_y


# --- 2. Hybrid LSTM Model Definition ---
def build_hybrid_lstm_model(input_shape):
    """Defines the Hybrid Conv1D + Bidirectional LSTM model architecture."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# --- 3. Alert System Logic ---
def apply_alert_thresholds(rul_predictions: np.ndarray) -> pd.Series:
    """Translates RUL predictions into maintenance alerts."""
    alerts = []
    for rul in rul_predictions.flatten():
        if rul < CRITICAL_RUL:
            alerts.append("CRITICAL")
        elif rul < WARNING_RUL:
            alerts.append("WARNING")
        else:
            alerts.append("NORMAL")
    return pd.Series(alerts, name="Alert_Level")


# --- 4. Plotting Functions (Milestone 3 & 4) ---

def plot_training_history(history):
    """Plot 1: Plots Training and Validation Loss/MAE (Milestone 3)."""
    plt.figure(figsize=(14, 5))

    # Loss (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Milestone 3: Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Milestone 3: Model MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True)
    
    plt.show()

def plot_rul_trajectory(y_test_original, y_pred_original):
    """Plot 2: Plots Predicted vs. Actual RUL Trajectory (Milestone 3)."""
    
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.arange(len(y_test_original)), 
        y_test_original.flatten(), 
        label='Actual RUL', 
        color='blue', 
        linewidth=2
    )
    plt.plot(
        np.arange(len(y_pred_original)), 
        y_pred_original.flatten(), 
        label='Predicted RUL', 
        color='red', 
        linestyle='--'
    )
    
    plt.axhline(CRITICAL_RUL, color='darkred', linestyle='-', linewidth=1, label=f'CRITICAL ({CRITICAL_RUL} days)')
    plt.axhline(WARNING_RUL, color='orange', linestyle='--', linewidth=1, label=f'WARNING ({WARNING_RUL} days)')
    
    plt.title('Milestone 3: Predicted vs. Actual RUL Trajectory')
    plt.xlabel('Time Step in Test Set (Sample Index)')
    plt.ylabel('RUL (Days/Cycles)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rul_scatter(y_test_original, y_pred_original):
    """Plot 3: Plots Predicted vs. Actual RUL Scatter Plot (Milestone 3)."""
    plt.figure(figsize=(8, 8))
    # Scatter plot
    plt.scatter(y_test_original.flatten(), y_pred_original.flatten(), alpha=0.6, s=15, color='darkgreen')
    
    # Ideal line (y=x)
    max_val = max(y_test_original.max(), y_pred_original.max())
    min_val = min(y_test_original.min(), y_pred_original.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction (y=x)')
    
    plt.title('Milestone 3: Predicted vs. Actual RUL (Scatter Plot)')
    plt.xlabel('Actual RUL (Days/Cycles)')
    plt.ylabel('Predicted RUL (Days/Cycles)')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_error_distribution(y_test_original, y_pred_original):
    """Plot 4: Plots Prediction Error Distribution (Histogram) (Milestone 3)."""
    errors = y_pred_original.flatten() - y_test_original.flatten()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True, color='skyblue')
    
    plt.axvline(errors.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean Error: {errors.mean():.2f}')
    
    plt.title('Milestone 3: Prediction Error Distribution')
    plt.xlabel('Prediction Error (Predicted RUL - Actual RUL)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def plot_alert_distribution(alert_series: pd.Series):
    """Plot 5: Generates and displays a bar chart of the maintenance alert distribution (Milestone 4)."""
    
    alert_counts = alert_series.value_counts().reindex(["NORMAL", "WARNING", "CRITICAL"], fill_value=0)
    alert_distribution = (alert_counts / len(alert_series)) * 100
    
    color_map = {"NORMAL": "g", "WARNING": "orange", "CRITICAL": "r"}
    
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=alert_distribution.index, 
        y=alert_distribution.values, 
        palette=[color_map[key] for key in alert_distribution.index]
    )
    
    plt.title("Milestone 4: Maintenance Alert Distribution", fontsize=14)
    plt.xlabel("Alert Level", fontsize=12)
    plt.ylabel("Percentage of Predictions (%)", fontsize=12)
    
    for i, p in enumerate(alert_distribution.values):
        plt.text(i, p + 1, f'{p:.1f}%', ha='center')
        
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--')
    plt.show()


# --- 5. Main Execution ---
if __name__ == "__main__":
    
    print("=" * 60)
    print("Executing Final Hybrid LSTM RUL Model with All 5 Plots")
    print("=" * 60)
    
    # 1. Load, Preprocess, and Split Data
    X_train, y_train, X_test, y_test, scaler_y = load_and_preprocess_data()
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"✓ Data loaded. Training samples: {len(X_train)}")
    
    # 2. Build and Train Model
    model = build_hybrid_lstm_model(input_shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("\nStarting Model Training (Patience=5)...")
    history = model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.1, 
        callbacks=[early_stopping],
        verbose=0 
    )
    print("✓ Training complete.")

    # 3. Generate Predictions and Inverse Transform
    y_pred_scaled = model.predict(X_test)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # --- Generate ALL Milestone 3 Plots ---
    print("\n--- Generating Milestone 3 Plots (Model Evaluation) ---")
    plot_training_history(history)
    plot_rul_trajectory(y_test_original, y_pred_original)
    plot_rul_scatter(y_test_original, y_pred_original)
    plot_error_distribution(y_test_original, y_pred_original)

    # --- Generate Milestone 4 Plot and Report ---
    alert_results = apply_alert_thresholds(y_pred_original) 
    
    alert_counts = alert_results.value_counts()
    alert_distribution = (alert_counts / len(alert_results) * 100).sort_index(ascending=False)
    
    print("\n" + "=" * 60)
    print("MILESTONE 4: MAINTENANCE ALERT SYSTEM RESULTS")
    print("=" * 60)
    print("ALERT DISTRIBUTION ON TEST SET:")
    for level, percentage in alert_distribution.items():
        print(f"{level: <10} {percentage:.1f}%")
    print("-" * 60)
        
    # Generate and display the final Milestone 4 plot
    plot_alert_distribution(alert_results)
    
    print("\nAll 5 plots generated successfully. Check the pop-up windows for visualizations.")