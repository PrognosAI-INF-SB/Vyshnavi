import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt # Included for plotting

# --- Project Constants (MINIMIZED CONFIGURATION) ---
N_TIMESTEPS = 50 
N_FEATURES = 14
N_SPLITS = 5
SEED = 42

# ⚡ FAST EXECUTION SETTINGS (CHANGE FOR FINAL RUN) ⚡
EPOCHS = 10         # Reduced from 150
TOTAL_SAMPLES = 1000  # Reduced from 10000+

# Set reproducibility seed
tf.random.set_seed(SEED)
np.random.seed(SEED)

def build_lstm_model(input_shape):
    """Constructs the Improved LSTM model for RUL prediction."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape, name='lstm_1'),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False, name='lstm_2'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', name='dense_1'),
        Dense(1, activation='linear', name='output_layer') 
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

# --- ⚠️ DUMMY DATA SETUP: REPLACE THIS BLOCK WITH YOUR ACTUAL DATA LOADING ⚠️ ---
print(f"Creating {TOTAL_SAMPLES} dummy samples and running {EPOCHS} epochs for quick test.")
# Load/Create your scaled feature data (X_scaled) and scaled target data (y_scaled)
X_scaled = np.random.rand(TOTAL_SAMPLES, N_TIMESTEPS, N_FEATURES).astype(np.float32)
y_scaled = np.random.rand(TOTAL_SAMPLES, 1).astype(np.float32)

# Load/Fit your real scalers here
scaler_X = MinMaxScaler().fit(np.random.rand(100, N_FEATURES)) 
scaler_y = MinMaxScaler().fit(np.random.rand(100, 1))
# --------------------------------------------------------------------------------


# --- CROSS-VALIDATION LOGIC ---
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_fold_results = []
all_fold_histories = []
fold_no = 1

for train_index, test_index in kf.split(X_scaled):
    print(f"\n--- Training Fold {fold_no}/{N_SPLITS} (Epochs: {EPOCHS}) ---")
    
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    
    model = build_lstm_model((N_TIMESTEPS, N_FEATURES))
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,           
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1 # Show progress
    )
    all_fold_histories.append(history)
    
    # Calculate key metrics
    y_pred_test = model.predict(X_test, verbose=0)
    y_pred_train = model.predict(X_train, verbose=0)
    
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    
    all_fold_results.append({
        'Fold': fold_no,
        'R2_Train': r2_train,
        'R2_Test': r2_test,
        'RMSE': rmse,
        'MAE': mae
    })
    fold_no += 1

# Save cross-validation results
df_results = pd.DataFrame(all_fold_results)
df_results.to_csv('crossval_results.csv', index=False)
print("\nCross-validation complete. Results saved to crossval_results.csv.") 
print(df_results)

# ------------------------------------------------
## VISUALIZATION (Generates PNG files and shows plots)
# ------------------------------------------------
print("\n--- Generating Visualization Charts ---")

# 1. Plotting Loss Curves for the first fold
history = all_fold_histories[0] 
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title(f'Fold 1: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('fold1_loss_curve.png') 
plt.show()

# 2. Plotting R2 Scores Across All Folds
plt.figure(figsize=(8, 5))
plt.plot(df_results['Fold'], df_results['R2_Train']*100, marker='o', label='Train R²')
plt.plot(df_results['Fold'], df_results['R2_Test']*100, marker='o', label='Test R²')
plt.title('R² Scores Across Folds')
plt.xlabel('Fold')
plt.ylabel('R² (%)')
plt.xticks(df_results['Fold'])
plt.grid(True)
plt.legend()
plt.savefig('crossval_r2_scores.png') 
plt.show()

# --- FINAL MODEL SAVING ---
print("\n--- Final Production Model Saving ---")
final_model = build_lstm_model((N_TIMESTEPS, N_FEATURES))

# For the ACTUAL final run, uncomment the training line below
# final_model.fit(X_scaled, y_scaled, epochs=150, batch_size=64, verbose=0) 

final_model.save("optimized_lstm_98plus.keras")
joblib.dump({'X': scaler_X, 'y': scaler_y}, "optimized_scalers.pkl")

print("\nSuccess: Final production model and scalers saved.")