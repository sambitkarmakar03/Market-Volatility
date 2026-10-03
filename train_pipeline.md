## Overview of the Volatility Forecasting Pipeline

The purpose of this feature engineering pipeline is to transform raw, noisy financial market data into clean, stationary, and predictive inputs for deep learning models, avoiding the severe pitfalls of feeding raw stock prices directly into a neural network.

---

### 1. Data Ingestion & Log Returns (Solving Stationarity)

* **The Problem:** Raw stock prices are non-stationary—their mean and variance change over time, and a $2 price shift means something entirely different for a low-value stock versus a high-value stock. Neural networks struggle to learn effectively from non-stationary inputs.
* **The Solution:** The pipeline downloads 5-minute historical bars and converts raw prices into **log returns** ($ln(P_t / P_{t-1})$).
* **The Benefit:** This normalizes the data into stationary percentage changes centered around zero, giving the model a universal scale to learn from regardless of the asset's absolute price tag.

---

### 2. Rolling Realized Volatility (Capturing Market Regime)

* **The Concept:** Volatility measures how aggressively a stock is shaking rather than which direction it is moving.
* **The Implementation:** By computing the standard deviation of log returns over a trailing 12-bar window (representing a 1-hour block), the pipeline creates a baseline feature for recent market behavior.
* **The Benefit:** It tells the model whether the preceding hours were calm or volatile, establishing the prevailing market regime.

---

### 3. Volume Z-Score (Automated Stress & Outlier Detection)

* **The Concept:** Price movements are heavily influenced by market participation and trading pressure. However, raw volume numbers fluctuate wildly and lack context.
* **The Implementation:** The pipeline calculates a **Z-score** over a 48-bar window, evaluating how much current volume deviates from its historical moving average and standard deviation.
* **The Benefit:** A Z-score close to `0` represents normal activity, while a high positive Z-score acts as an automated outlier detector for abnormal volume spikes—serving as an early warning indicator for impending market volatility shocks.

---

### 4. Target Future Volatility (Establishing the Prediction Objective)

* **The Concept:** A supervised forecasting model requires a clear "answer key" that looks forward in time rather than reacting to the present.
* **The Implementation:** The pipeline shifts the calculation window forward by 12 bars to measure **future realized volatility** over the upcoming hour.
* **The Benefit:** This maintains strict temporal alignment and eliminates look-ahead bias, ensuring that the LSTM learns to genuinely forecast upcoming risk based purely on historical patterns.



## Hybrid GARCH-LSTM Feature Engineering Pipeline

1. Multi-Feature Engineering Suite

- Log Returns (log_return): Transforms raw closing prices into stationary returns ($\log(P_t / P_{t-1})$).
- Rolling Realized Volatility (rolling_vol): Measures trailing standard deviation (10-day window) to capture local momentum.
- Volume Z-Score (volume_zscore): Normalizes trading volume (20-day rolling window) to flag liquidity spikes and market stress anomalies.
- Econometric Hybrid Feature (garch_vol): Injects structural financial intelligence to model baseline persistence.
- Target Variable (target_future_vol): The forward-looking 10-day realized volatility window shifted strictly onto row $t$ to prevent lookahead bias.

2. What is GARCH(1,1)?

Standard statistics assumes volatility is constant, but financial markets exhibit volatility clustering (calm periods group together, and large shocks are followed by prolonged choppy trading)
- Core Principle: GARCH(1,1) calculates tomorrow's risk based on a combination of yesterday's unexpected price shock (the ARCH term, $\alpha$) and yesterday's forecasted risk level (the GARCH persistence term, $\beta$)
- Why it matters: It provides an econometric foundation that models how market panic or calm naturally lingers over time.
 
3. Optimization Techniques in GARCH

- Maximum Likelihood Estimation (MLE): The primary optimization engine. It iteratively adjusts the model's parameters ($\omega, \alpha, \beta$) to find the exact combination that maximizes the statistical likelihood of having witnessed the historical price shocks that actually occurred.
- Numerical Solvers: Behind the scenes, gradient-based numerical optimization algorithms (such as Sequential Least Squares Programming) search the parameter space to find the optimal econometric fit.