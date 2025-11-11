# ============================================================
# 📊 AI PrognosAI - Final Integrated RUL Dashboard
# Author: Vyshnavi 💚
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import streamlit as st
import plotly.express as px

# ============================================================
# 🧩 1. Generate synthetic machine sensor data
# ============================================================

def generate_data(n_samples=500):
    np.random.seed(42)
    data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n_samples, freq="h"),
        "temperature": np.random.uniform(40, 100, n_samples),
        "pressure": np.random.uniform(1, 10, n_samples),
        "vibration": np.random.uniform(0.1, 1.0, n_samples),
        "voltage": np.random.uniform(200, 250, n_samples)
    })
    data["RUL"] = np.maximum(0, 120 - (0.4 * data["temperature"] +
                                       0.3 * data["pressure"] +
                                       0.2 * data["vibration"] +
                                       np.random.uniform(-2, 2, n_samples)))
    return data

# ============================================================
# ⚙️ 2. Preprocessing
# ============================================================

def preprocess_data(data):
    features = ["temperature", "pressure", "vibration", "voltage"]
    target = "RUL"
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[features])
    y = data[target].values
    return X_scaled, y, scaler

# ============================================================
# 🧠 3. Define and Train LSTM Model
# ============================================================

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ============================================================
# 📊 4. Streamlit Dashboard UI
# ============================================================

st.set_page_config(page_title="AI PrognosAI Dashboard", layout="wide")

st.title("⚙️ AI PrognosAI – Equipment Health Dashboard")
st.markdown("Monitor Remaining Useful Life (RUL) and detect early failures 🔍")

# Sidebar
st.sidebar.header("🔧 Dashboard Controls")
data_choice = st.sidebar.selectbox("Select Data Source", ["Demo Dataset", "Upload CSV"])

if data_choice == "Demo Dataset":
    data = generate_data()
else:
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a dataset to continue.")
        st.stop()

# ============================================================
# 🧮 5. Model Training
# ============================================================

X, y, scaler = preprocess_data(data)
X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

model = build_model((1, X.shape[1]))
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # fixed ✅

# ============================================================
# 🚨 6. Alerts and RUL Insights
# ============================================================

latest_rul = y_pred[-1][0]

if latest_rul < 20:
    status = "🔴 Critical"
    color = "red"
elif latest_rul < 50:
    status = "🟠 Warning"
    color = "orange"
else:
    status = "🟢 Healthy"
    color = "green"

col1, col2, col3 = st.columns(3)
col1.metric("📉 Mean Absolute Error", f"{mae:.3f}")
col2.metric("📈 R² Score", f"{r2:.3f}")
col3.metric("📊 RMSE", f"{rmse:.3f}")

st.markdown(f"### Machine Status: <span style='color:{color};font-weight:bold'>{status}</span>", unsafe_allow_html=True)

# ============================================================
# 🧾 7. Display Data Table
# ============================================================

st.subheader("📋 Sample of Sensor Data")
st.dataframe(data.head(15))  # Show first 15 rows

results = pd.DataFrame({
    "Actual RUL": y_test.flatten(),
    "Predicted RUL": y_pred.flatten()
})
st.subheader("🔍 Model Predictions")
st.dataframe(results.head(15))

# ============================================================
# 📈 8. Interactive Charts
# ============================================================

fig1 = px.line(data, x="timestamp", y="RUL", title="📉 RUL Trend Over Time", line_shape="spline", color_discrete_sequence=[color])
fig1.update_layout(hovermode="x unified", template="plotly_dark")

fig2 = px.scatter(data, x="temperature", y="RUL", color="RUL",
                  title="🌡️ Temperature vs RUL", color_continuous_scale="RdYlGn_r")

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# ✅ 9. Save & Export Results
# ============================================================

if st.sidebar.button("💾 Save Predictions"):
    results.to_csv("Final_RUL_Predictions.csv", index=False)
    st.sidebar.success("Predictions saved as Final_RUL_Predictions.csv ✅")

st.success("✅ Dashboard loaded successfully! Explore the data, tables, and charts above.")
