import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_preprocess():
    filepath = os.path.join("data", "train_FD001.txt")

    columns = [
        'unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
        'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
        'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
    ]

    try:
        data = pd.read_csv(filepath, sep=r'\s+', header=None, names=columns, engine='python')
    except Exception as e:
        print("❌ Error reading file:", e)
        return None, None

    if data.shape[0] == 0:
        print("🚨 ERROR: train_FD001.txt is empty!")
        return None, None

    # --- Rename for consistency ---
    data = data.rename(columns={'unit': 'unit_number'})

    # --- Calculate RUL ---
    data['RUL'] = data.groupby('unit_number')['time'].transform(max) - data['time']

    # --- Normalize sensor columns ---
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    scaler = MinMaxScaler()
    data[sensor_cols] = scaler.fit_transform(data[sensor_cols])

    print("✅ Preprocessing completed successfully!")
    print(data.head(5))
    return data, sensor_cols


if __name__ == "__main__":
    data, sensors = load_and_preprocess()

