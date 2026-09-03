# Comprehensive Evaluation Report: LSTM vs. LSTM + GARCH Hybrid Quantile Forecasting Framework

## 1. Executive Summary

This report evaluates the predictive performance of two volatility forecasting architectures:

1. **Baseline LSTM:** Uses market features without GARCH volatility.
2. **LSTM + GARCH Hybrid:** Extends the baseline by incorporating GARCH-derived conditional volatility as an additional input feature.

The models are evaluated across a multi-asset universe of **20 tickers**, with approximately **377–378 test observations per ticker**.

The evaluation focuses on both point and probabilistic forecasting performance using:

- Mean Pinball Loss
- Median Absolute Error (MAE)
- Median Root Mean Squared Error (RMSE)
- 80% empirical coverage
- Quantile crossing rate
- Diebold-Mariano (DM) test with HAC/Newey-West correction

The results indicate that the **LSTM + GARCH Hybrid model achieves substantially lower forecasting errors and better interval calibration at the aggregate level**. The DM test further provides statistical evidence that the difference in forecasting performance is not simply attributable to random variation.

---

# 2. Evaluation Metrics

## 2.1 Pinball Loss

Because the model produces probabilistic forecasts at the 10th, 50th, and 90th quantiles, conventional mean squared error alone is not sufficient for evaluation.

The **Pinball Loss** evaluates the accuracy of an individual quantile forecast.

For quantile $\tau$:

$$
\mathcal{L}_{\tau}(y_t,\hat{y}_t)
=
\max
\left[
\tau(y_t-\hat{y}_t),
(\tau-1)(\hat{y}_t-y_t)
\right]
$$

where:

- $y_t$ is the actual future volatility
- $\hat{y}_t$ is the predicted volatility at quantile $\tau$
- $\tau \in \{0.10,0.50,0.90\}$

The model therefore produces three forecasts:

$$
Q_{10},\quad Q_{50},\quad Q_{90}
$$

The per-observation loss can be aggregated across the three quantiles to obtain an overall probabilistic forecasting loss.

Lower Pinball Loss indicates better quantile forecasting performance.

---

## 2.2 Median Absolute Error (MAE)

MAE evaluates the accuracy of the median forecast ($Q_{50}$):

$$
MAE =
\frac{1}{T}
\sum_{t=1}^{T}
|y_t-\hat{y}_{50,t}|
$$

A lower MAE indicates that the median volatility forecast is closer to the actual future volatility.

---

## 2.3 Median RMSE

RMSE gives greater weight to larger forecasting errors:

$$
RMSE =
\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
(y_t-\hat{y}_{50,t})^2
}
$$

This is particularly useful for identifying whether a model produces large forecasting errors during difficult market conditions.

---

## 2.4 80% Coverage

The model produces an interval between the 10th and 90th quantiles:

$$
[Q_{10},Q_{90}]
$$

This is intended to represent an approximately **80% prediction interval**.

The empirical coverage is:

$$
Coverage =
\frac{
\text{Number of actual observations inside }[Q_{10},Q_{90}]
}{
\text{Total observations}
}
\times100
$$

An ideally calibrated 80% interval should have coverage close to:

$$
80\%
$$

---

## 2.5 Quantile Crossing Rate

Quantile forecasts should preserve their natural ordering:

$$
Q_{10}\leq Q_{50}\leq Q_{90}
$$

A **quantile crossing** occurs when this ordering is violated.

The crossing rate measures the proportion of observations where such an inversion occurs.

A crossing rate of:

$$
0.00\%
$$

means that the predicted quantiles maintained the correct ordering throughout the evaluated observations.

---

# 3. Diebold-Mariano Test

## 3.1 Why the DM Test Is Used

The aggregate metrics tell us that the Hybrid model has lower forecasting error.

However, a lower average loss alone does not establish whether the improvement is statistically significant.

The **Diebold-Mariano (DM) test** is designed specifically to compare the predictive accuracy of two competing forecasting models. It tests whether their expected forecast losses are equal. :contentReference[oaicite:1]{index=1}

For this project, the comparison is:

```text
Baseline LSTM
       vs.
LSTM + GARCH Hybrid