import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_and_plot(model, history, X_val, y_val):
    # Create 'plots' folder if not exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Predictions
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"📈 MAE: {mae:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")

    # --- LOSS CURVE ---
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/loss_curve.png')
    print("✅ Saved: plots/loss_curve.png")
    plt.show()  # 🔹 Automatically open the plot

    # --- PREDICTED vs ACTUAL ---
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(y_val)), y_val, label='Actual RUL', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted RUL', alpha=0.6)
    plt.title('Predicted vs Actual RUL')
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/predicted_vs_actual.png')
    print("✅ Saved: plots/predicted_vs_actual.png")
    plt.show()  # 🔹 Automatically open the plot
