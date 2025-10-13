import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

from train_model import train_and_save_model
from evaluate_model import evaluate_and_plot

if __name__ == "__main__":
    model, history, (X_val, y_val) = train_and_save_model()
    evaluate_and_plot(model, history, X_val, y_val)
    print("✅ Model training and evaluation complete. Plots saved in /plots folder.")


