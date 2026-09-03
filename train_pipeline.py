import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model


def fit_garch_volatility(
    series: pd.Series,
    min_window: int = 252,
    refit_every: int = 10,
) -> pd.Series:
    """
    Fits GARCH(1,1) on an expanding window, refitting periodically (not every
    single day) to keep runtime reasonable across many tickers. No lookahead:
    every fit uses only data strictly before the day being forecast.

    refit_every: re-estimate GARCH parameters every N days; in between,
    reuse the last fitted model's rolling forecast. This cuts fit count by
    ~refit_every-x with negligible accuracy cost, since GARCH parameters
    change slowly day to day.
    """
    # CRITICAL FIX: drop the leading NaN (log_return's first entry is always
    # NaN) BEFORE slicing — otherwise every train_data slice contains it,
    # arch_model raises on every call, and the except branch silently
    # replaces the entire column with a plain rolling std, every time.
    clean_series = series.dropna()

    # Defensive: guard against stray inf values (e.g. a bad tick where Close
    # momentarily reads 0) which would trigger the same silent-failure pattern.
    clean_series = clean_series.replace([np.inf, -np.inf], np.nan).dropna()

    scaled_returns = clean_series * 100.0

    garch_vols = pd.Series(index=clean_series.index, dtype=float)
    fallback_count = 0
    last_model_fit = None
    last_fit_pos = -1

    for i in range(len(scaled_returns)):
        if i < min_window:
            garch_vols.iloc[i] = np.nan
            continue

        needs_refit = (last_model_fit is None) or (i - last_fit_pos >= refit_every)

        if needs_refit:
            train_data = scaled_returns.iloc[:i]
            try:
                model = arch_model(
                    train_data, vol="Garch", p=1, q=1, mean="Constant", rescale=False
                )
                last_model_fit = model.fit(disp="off", show_warning=False)
                last_fit_pos = i
            except Exception as e:
                if fallback_count == 0:
                    print(
                        f"  [WARN] GARCH fit failed at position {i}: {e}. "
                        f"Using rolling-std fallback (will not re-print for repeats)."
                    )
                fallback_count += 1
                garch_vols.iloc[i] = scaled_returns.iloc[max(0, i - 10):i].std() / 100.0
                continue

        try:
            forecast = last_model_fit.forecast(horizon=1, reindex=False)
            conditional_var = forecast.variance.iloc[-1].values[0]
            garch_vols.iloc[i] = np.sqrt(conditional_var) / 100.0
        except Exception:
            fallback_count += 1
            garch_vols.iloc[i] = scaled_returns.iloc[max(0, i - 10):i].std() / 100.0

    if fallback_count > 0:
        print(
            f"  [INFO] GARCH fallback triggered on {fallback_count}/{len(scaled_returns)} rows "
            f"— if this is a large fraction, investigate before trusting garch_vol."
        )

    # Reindex back to the ORIGINAL series index (restoring the dropped-NaN /
    # dropped-inf positions as NaN) so it aligns cleanly with the dataframe.
    return garch_vols.reindex(series.index)


def generate_multi_ticker_daily_pipeline(
    tickers=None,
    period="5y",
    garch_min_window: int = 252,
    garch_refit_every: int = 10,
):
    """
    Pulls daily OHLCV for multiple tickers and engineers volatility-forecasting
    features, including a GARCH(1,1) conditional volatility feature (fixed —
    see fit_garch_volatility for the leading-NaN fix and periodic-refit speedup).

    Daily bars are NOT subject to yfinance's 60-day intraday cap, so `period`
    can safely span several years to maximize data volume.
    """
    if tickers is None:
        # A broader, more varied set than a handful of correlated megacap tech
        # names — mixes large stable caps, historically higher-vol names, and
        # market-wide ETFs so the pooled dataset spans different vol regimes.
        tickers = [
            "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
            "TSLA", "AMD", "NFLX", "META", "JPM",
            "XOM", "PFE", "KO", "WMT", "DIS",
            "GME", "COIN", "PLTR", "SPY", "QQQ",
        ]

    all_dfs = []

    for ticker in tickers:
        print(f"\nFetching historical daily bars for {ticker} over a {period} window...")
        df = yf.download(ticker, period=period, interval="1d", progress=False)

        if df.empty:
            print(f"  -> No data returned for {ticker}, skipping.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 1. Base Feature: Log Return
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_return"] = df["log_return"].replace([np.inf, -np.inf], np.nan)

        # 2. Feature: Rolling Realized Volatility (trailing 10-day window)
        window_trailing = 10
        df["rolling_vol"] = df["log_return"].rolling(window=window_trailing).std()

        # 3. Feature: Volume Z-Score (20-day window)
        vol_mean = df["Volume"].rolling(window=20).mean()
        vol_std = df["Volume"].rolling(window=20).std()
        df["volume_zscore"] = (df["Volume"] - vol_mean) / (vol_std + 1e-8)

        # 4. Feature: GARCH(1,1) Conditional Volatility (fixed hybrid baseline)
        print(f"  -> Fitting expanding GARCH(1,1) conditional volatility for {ticker}...")
        df["garch_vol"] = fit_garch_volatility(
            df["log_return"],
            min_window=garch_min_window,
            refit_every=garch_refit_every,
        )

        # 5. Target Variable: Future Realized Volatility (no lookahead —
        # rolling(10).std() at row t+10 covers rows [t+1..t+10]; shifting the
        # whole series back by -10 aligns that strictly-future value onto row t)
        window_forward = 10
        df["target_future_vol"] = (
            df["log_return"].rolling(window=window_forward).std().shift(-window_forward)
        )

        # Clean NaNs (includes GARCH warm-up period and shift offsets)
        df.dropna(inplace=True)
        df["ticker"] = ticker

        all_dfs.append(df)

    master_df = pd.concat(all_dfs, axis=0)
    print(
        f"\nMulti-ticker GARCH-fusion pipeline complete! {len(all_dfs)} tickers, "
        f"master dataset shape: {master_df.shape}"
    )
    return master_df


def time_based_split(df: pd.DataFrame, val_start: str, test_start: str):
    """
    Splits the pooled multi-ticker dataset by DATE, not randomly — this avoids
    leaking market-wide volatility events across train/val/test when multiple
    tickers share overlapping date ranges.

    val_start / test_start: date strings, e.g. "2024-06-01", "2024-10-01"
    """
    train = df[df.index < val_start]
    val = df[(df.index >= val_start) & (df.index < test_start)]
    test = df[df.index >= test_start]
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    return train, val, test


if __name__ == "__main__":
    dataset = generate_multi_ticker_daily_pipeline(period="5y")

    print("\nMaster dataset tail preview:")
    print(
        dataset[[
            "ticker",
            "Close",
            "rolling_vol",
            "garch_vol",
            "volume_zscore",
            "target_future_vol",
        ]].tail(3)
    )

    print("\nRows per ticker:")
    print(dataset.groupby("ticker").size())

    # Sanity check suggested earlier: garch_vol should NOT be near-identical
    # to rolling_vol. A correlation near 1.0 here would suggest the GARCH
    # fallback is still triggering too often — investigate before trusting it.
    corr = dataset[["rolling_vol", "garch_vol"]].corr().iloc[0, 1]
    print(f"\nCorrelation between rolling_vol and garch_vol: {corr:.4f}")
    print("(Expect a meaningfully positive but NOT near-1.0 correlation — "
          "near-1.0 suggests GARCH is degenerating into the rolling-std fallback.)")