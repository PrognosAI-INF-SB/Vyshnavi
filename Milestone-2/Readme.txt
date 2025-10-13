🧩 Milestone 2: Model Development & Training
Project Name:PrognosAI: AI-Driven Predictive Maintenance System Using Time-Series Sensor Data
Dataset:NASA Turbofan Jet Engine Degradation Simulation Dataset (FD001)

🎯 Objective:
To develop and train an LSTM (Long Short-Term Memory) deep learning model that predicts the Remaining Useful Life (RUL) of turbofan engines using time-series sensor data.
This milestone focuses on data preprocessing, model training, evaluation, and visualization of performance metrics.

🧠 Modules Used and Their Purpose:
Library	Purpose
NumPy	Efficient numerical computations and array handling
Pandas	Data manipulation and cleaning
TensorFlow / Keras	Building, training, and saving the LSTM model
Sequential, LSTM, Dense, Dropout	Defining deep learning model architecture
EarlyStopping, ReduceLROnPlateau	Callbacks to control overfitting and optimize learning
Adam Optimizer	Optimizes the model weights during training
sklearn.preprocessing.MinMaxScaler	Scales input features for stable training
sklearn.metrics	Evaluates model performance (MAE, RMSE)
Matplotlib / Seaborn	Visualizes loss curves and predictions
joblib / pickle	Saves scalers for reuse in prediction/inference

Plotted:
Training vs Validation Loss (plots/loss_curve.png)
Predicted vs Actual RUL (plots/predicted_vs_actual.png)

Example results:
📈 MAE: 54.14
📉 RMSE: 65.36
✅ Model trained and saved successfully.

Model Saving
Saved the trained model in HDF5 (.h5) format inside the models/ folder.
Saved plots in the plots/ directory for visualization.
