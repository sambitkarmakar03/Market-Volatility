import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from Quantile_LSTM import PinballLoss  # Required for custom loss loading
from train import (
    create_sequences_by_ticker,
    generate_multi_ticker_daily_pipeline,
    time_based_split,
)

CACHE_PATH = "master_df_cache.parquet"
MODEL_PATH = "quantile_lstm_model.keras"
LOOKBACK = 30
QUANTILE_LABELS = ("q10", "q50", "q90")


def load_dataset(period: str = "5y") -> pd.DataFrame:
    """
    Loads the pooled multi-ticker dataset, using a parquet cache if present
    so evaluate.py doesn't re-fetch data and re-fit ~2,500 GARCH models from
    scratch on every run. Delete the cache file to force a refresh.
    """
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached dataset from {CACHE_PATH}...")
        return pd.read_parquet(CACHE_PATH)

    print("No cache found — generating dataset (this fits GARCH per ticker, expect a few minutes)...")
    master_df = generate_multi_ticker_daily_pipeline(period=period)
    master_df.to_parquet(CACHE_PATH)
    print(f"Cached dataset to {CACHE_PATH} for future runs.")
    return master_df


def get_sequence_metadata(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
):
    """
    Mirrors create_sequences_by_ticker's exact iteration order (grouped by
    ticker, sorted by date, same lookback offset) to produce parallel
    ticker/date arrays aligned with X/y — needed so predictions can be
    traced back to a specific ticker and date for plotting, without
    modifying train.py's function signature.
    """
    tickers_out, dates_out = [], []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_index()
        for i in range(lookback, len(group)):
            tickers_out.append(ticker)
            dates_out.append(group.index[i])
    return np.array(tickers_out), np.array(dates_out)


def check_quantile_crossing(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray):
    """
    Verifies q10 <= q50 <= q90 for every prediction. The originally-trained
    model used independent linear heads with no monotonicity constraint, so
    this is a real possibility, not a formality — check it before trusting
    any calibration numbers below.
    """
    crossed_low = q10 > q50
    crossed_high = q50 > q90
    n_crossed = np.sum(crossed_low | crossed_high)
    pct = 100 * n_crossed / len(q10)
    print(f"\nQuantile crossing check: {n_crossed}/{len(q10)} ({pct:.2f}%) predictions violate q10<=q50<=q90")
    if pct > 1.0:
        print("  [WARN] Non-trivial crossing rate — consider retraining with the "
              "monotonic-head architecture (enforce_monotonic=True in build_quantile_lstm) "
              "before trusting the calibration results below.")
    return n_crossed


def compute_pinball_loss(y_true: np.ndarray, preds: np.ndarray, quantiles=(0.10, 0.50, 0.90)) -> float:
    """Pinball loss on the test set, matching the training objective, as a single summary number."""
    y_true = y_true.reshape(-1, 1)
    q = np.array(quantiles).reshape(1, -1)
    error = y_true - preds
    losses = np.maximum(q * error, (q - 1.0) * error)
    return float(np.mean(np.sum(losses, axis=-1)))


def compute_calibration(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray, nominal: float = 0.80) -> float:
    """
    Empirical coverage: fraction of true outcomes falling inside [q10, q90].
    Should be close to `nominal` (0.80) if the model is well-calibrated.
    """
    inside = (y_true >= q10) & (y_true <= q90)
    coverage = float(np.mean(inside))
    print(f"\nCalibration check: nominal 80% interval covers {coverage*100:.2f}% of actual outcomes "
          f"(target: {nominal*100:.0f}%)")
    return coverage


def calibration_curve(y_true: np.ndarray, preds: np.ndarray, quantiles=(0.10, 0.50, 0.90)):
    """
    Builds a calibration curve using the available quantile heads as
    reference points: for each predicted quantile level, what fraction of
    true outcomes actually fall below that prediction? A perfectly
    calibrated model has empirical == nominal at every point (the diagonal).
    """
    nominal_levels = []
    empirical_levels = []
    for idx, q_level in enumerate(quantiles):
        below = np.mean(y_true <= preds[:, idx])
        nominal_levels.append(q_level)
        empirical_levels.append(below)
        print(f"  Nominal q{int(q_level*100)}: empirical fraction below = {below*100:.2f}%")
    return nominal_levels, empirical_levels


def plot_calibration_curve(nominal_levels, empirical_levels, save_path="calibration_curve.png"):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(nominal_levels, empirical_levels, marker="o", color="firebrick", label="Model")
    plt.xlabel("Nominal quantile level")
    plt.ylabel("Empirical fraction of outcomes below prediction")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved {save_path}")


def plot_fan_chart(
    dates: np.ndarray,
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    ticker: str,
    save_path: str = None,
):
    """Actual vs. predicted forecast band over time, for one ticker."""
    save_path = save_path or f"fan_chart_{ticker}.png"
    order = np.argsort(dates)
    dates, y_true, q10, q50, q90 = dates[order], y_true[order], q10[order], q50[order], q90[order]

    plt.figure(figsize=(14, 5))
    plt.fill_between(dates, q10, q90, alpha=0.25, color="steelblue", label="80% predicted interval")
    plt.plot(dates, q50, color="steelblue", label="Predicted median")
    plt.plot(dates, y_true, color="black", linewidth=1, label="Actual realized volatility")
    plt.title(f"{ticker} — Volatility Forecast Fan Chart")
    plt.xlabel("Date")
    plt.ylabel("Realized volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved {save_path}")


def run_evaluation():
    print("Step 1: Loading test dataset...")
    master_df = load_dataset(period="5y")
    _, _, test_df = time_based_split(
        master_df, val_start="2024-06-01", test_start="2025-01-01"
    )

    print("\nStep 2: Generating test sequences...")
    X_test, y_test = create_sequences_by_ticker(test_df, lookback=LOOKBACK)
    tickers_arr, dates_arr = get_sequence_metadata(test_df, lookback=LOOKBACK)
    assert len(tickers_arr) == len(X_test), (
        "Metadata length mismatch vs. sequences — check that get_sequence_metadata's "
        "grouping/lookback exactly matches create_sequences_by_ticker's."
    )

    print("\nStep 3: Loading saved Quantile LSTM model...")
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"PinballLoss": PinballLoss}
    )

    print("\nStep 4: Predicting risk quantiles on test data...")
    preds = model.predict(X_test)
    q10, q50, q90 = preds[:, 0], preds[:, 1], preds[:, 2]
    print(f"Generated predictions shape: {preds.shape}")

    print("\nStep 5: Validating predictions...")
    check_quantile_crossing(q10, q50, q90)

    pinball = compute_pinball_loss(y_test, preds)
    print(f"\nTest-set pinball loss: {pinball:.6f}")

    compute_calibration(y_test, q10, q90, nominal=0.80)

    print("\nCalibration curve (per-quantile):")
    nominal_levels, empirical_levels = calibration_curve(y_test, preds)
    plot_calibration_curve(nominal_levels, empirical_levels)

    print("\nStep 6: Plotting fan chart for a sample ticker...")
    sample_ticker = tickers_arr[0]  # change to a specific ticker if you want a particular one
    mask = tickers_arr == sample_ticker
    plot_fan_chart(
        dates_arr[mask], y_test[mask], q10[mask], q50[mask], q90[mask], ticker=sample_ticker
    )

    print("\nEvaluation complete.")
    return X_test, y_test, q10, q50, q90


if __name__ == "__main__":
    run_evaluation()