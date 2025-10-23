Milestone 3 – Model Optimization & Cross-Validation

Objective:
Optimize the LSTM model from Milestone 2 using advanced layers, cross-validation, and hyperparameter tuning to achieve better RUL prediction accuracy.

Steps Implemented:
Constructed a deeper LSTM architecture with normalization and dropout.
Applied 5-fold Cross-Validation to evaluate generalization performance.
Recorded performance metrics (R², RMSE, MAE) across folds.
Visualized training and validation losses.
Saved optimized model (optimized_lstm_98plus.keras) and scalers (optimized_scalers.pkl).

Deliverables:
optimized_lstm_98plus.keras – final production model
optimized_scalers.pkl – saved normalization objects
crossval_results.csv – metrics for each fold
crossval_r2_scores.png – R² visualization across folds

Results:
The optimized model achieved >98% R² accuracy in cross-validation, confirming robustness and reliability for predictive maintenance.
