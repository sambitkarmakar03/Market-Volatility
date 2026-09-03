# Hybrid GARCH-LSTM Market Volatility & Quantile Forecasting

A robust probabilistic financial forecasting pipeline that combines **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** volatility modeling with **Quantile Regression LSTMs** to forecast market volatility and establish reliable risk prediction intervals.

---

## 📈 Project Overview

Traditional neural networks (like standard LSTMs) often struggle with sudden market shocks because they rely on static uncertainty bounds and fail to account for volatility clustering. This project implements a **Hybrid Architecture** that feeds explicit GARCH volatility metrics into an LSTM trained via quantile loss. 

By predicting upper, lower, and median quantiles (e.g., an 80% prediction interval), the model dynamically widens or narrows its risk bounds based on real-time market turbulence, significantly outperforming baseline models.

### Key Highlights:
* **Error Reduction:** Cuts Mean Pinball Loss, MAE, and RMSE by **23% to 25%** compared to a baseline LSTM.
* **Superior Risk Calibration:** Achieves an **82.87% empirical coverage rate** for its 80% prediction interval (closely matching the theoretical target), whereas the baseline is severely under-calibrated at **75.09%**.
* **Statistical Validation:** Backed by the **Diebold-Mariano test**, confirming statistically significant performance gains ($p < 0.0001$) across asset tickers.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.11+
* **Deep Learning:** TensorFlow / Keras (`quantile_lstm_model.keras`, `lstm_baseline_model.keras`)
* **Financial Econometrics:** `arch` (for GARCH volatility feature extraction)
* **Data Processing & Analysis:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn (`side_by_side_prediction_comparison.png`, `calibration_curve.png`)

---

## 📂 Repository Structure

```text
├── GARCH.md                              # Documentation on GARCH feature engineering pipeline
├── LSTM.md                               # Deep dive into the LSTM quantile architecture
├── train.py                              # Main training script for the Hybrid model
├── train_pipeline.py                     # End-to-end data ingestion and pipeline script
├── evaluate.py                           # Model evaluation script (calculates coverage, pinball loss)
├── comparison.ipynb                      # Jupyter notebook for side-by-side comparative analysis
├── requirements.txt                      # Python package dependencies
├── side_by_side_prediction_comparison.png  # Visual proof of dynamic interval expansion vs baseline
├── calibration_curve.png                 # Probability calibration and reliability plot
├── dm_test_per_ticker.csv                # Diebold-Mariano statistical test results per ticker
└── *.keras                               # Serialized trained model weights
