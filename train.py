import numpy as np
import pandas as pd
import tensorflow as tf
from Quantile_LSTM import build_quantile_lstm  # Assuming your model is in Quantile_LSTM.py
from train_pipeline import (
    generate_multi_ticker_daily_pipeline,
    time_based_split,
)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
  print(f"\n[INFO] Found {len(gpus)} GPU(s): {gpus}")
  print("[INFO] Forcing execution onto Apple Silicon GPU (MPS)...")
  device_name = "/GPU:0"
else:
  print(
      "\n[WARN] No GPU found. Falling back to CPU (Check tensorflow-metal"
      " installation)."
  )
  device_name = "/CPU:0"

def create_sequences_by_ticker(
    df: pd.DataFrame,
    feature_cols=[
        "log_return",
        "rolling_vol",
        "volume_zscore",
        "garch_vol",
    ],
    target_col="target_future_vol",
    lookback: int = 30,
):
  """Slices a pooled multi-ticker dataframe into 3D LSTM arrays (samples,

  time_steps, features) ensuring sequences never cross ticker boundaries.
  """
  X_list, y_list, tickers_list = [], [], []

  for ticker, group in df.groupby("ticker"):
    group = group.sort_index()

    feature_data = group[feature_cols].values
    target_data = group[target_col].values

    for i in range(lookback, len(group)):
      X_seq = feature_data[i - lookback : i]
      y_val = target_data[i]

      X_list.append(X_seq)
      y_list.append(y_val)
      tickers_list.append(ticker)

  X = np.array(X_list, dtype=np.float32)
  y = np.array(y_list, dtype=np.float32)

  print(
      f"Generated sequences -> X: {X.shape}, y: {y.shape} across"
      f" {len(np.unique(tickers_list))} tickers."
  )
  return X, y


if __name__ == "__main__":
  print("Step 1: Loading multi-ticker dataset with GARCH feature fusion...")
  master_df = generate_multi_ticker_daily_pipeline(period="5y")

  print("\nStep 2: Splitting dataset by date to prevent market leakage...")
  # Adjust these date thresholds based on your timeline (e.g., 5y from 2026 goes back to 2021)
  # Let's use 2024-06-01 for validation start and 2025-01-01 for test start as an example
  train_df, val_df, test_df = time_based_split(
      master_df, val_start="2024-06-01", test_start="2025-01-01"
  )

  print("\nStep 3: Generating 3D tensor sequences for LSTM...")
  LOOKBACK = 30
  X_train, y_train = create_sequences_by_ticker(train_df, lookback=LOOKBACK)
  X_val, y_val = create_sequences_by_ticker(val_df, lookback=LOOKBACK)
  X_test, y_test = create_sequences_by_ticker(test_df, lookback=LOOKBACK)

  print("\nStep 4: Building Quantile LSTM Model...")
  model = build_quantile_lstm(
      lookback_window=LOOKBACK, num_features=4, quantiles=[0.10, 0.50, 0.90]
  )

  print("\nStep 5: Training Model...")
  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss", patience=5, restore_best_weights=True
  )

  history = model.fit(
      X_train,
      y_train,
      validation_data=(X_val, y_val),
      epochs=30,
      batch_size=64,
      callbacks=[early_stopping],
  )

  print("\nTraining complete! Model is ready for backtesting and evaluation.")
  # Save the trained model weights and architecture
  model.save("quantile_lstm_model.keras")
  print("\n[INFO] Model successfully saved to 'quantile_lstm_model.keras'")