# PrognosAI: AI-Driven Predictive Maintenance System

**Milestone 1 – Data Preparation & Feature Engineering**

## 📌 Project Objective

The goal of PrognosAI is to design and develop an AI-based predictive maintenance system capable of estimating the **Remaining Useful Life (RUL)** of industrial machinery using **multivariate time-series sensor data**.

This milestone focuses on preparing and preprocessing the **NASA CMAPSS dataset**, generating rolling window sequences, and computing RUL targets for model training.

---

## 🛠️ Workflow (Milestone 1)

1. **Data Ingestion**

   * Load CMAPSS FD001 dataset into pandas DataFrame.
2. **Feature Engineering**

   * Compute Remaining Useful Life (RUL) for each engine cycle.
   * Create rolling window sequences for time-series modeling.
3. **Output**

   * Prepared dataset with sensor features and RUL labels.
   * Training-ready rolling sequences.

---

## 📊 Deliverables for Milestone 1

* ✅ Cleaned and preprocessed CMAPSS dataset.
* ✅ Python script for data loading and preprocessing.
* ✅ Rolling window sequence generation.
* ✅ Computed RUL targets for each cycle.

---

## ⚙️ Tech Stack

* **Python** – Core programming language
* **NumPy, Pandas** – Data processing
* **Matplotlib, Seaborn** – Visualization (optional for Milestone 1)
* **Scikit-learn** – Utilities & metrics
* **TensorFlow/Keras** – (For Milestone 2 onward: model training)

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/PrognosAI.git
   cd PrognosAI
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the milestone script:

   ```bash
   python milestone1.py
   ```





