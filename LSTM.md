# Quantile LSTM for Future Volatility Forecasting

## 1. Why Use a Quantile LSTM?

The GARCH model provides an estimate of the **current conditional volatility** based primarily on the historical behavior of returns.

However, GARCH is an econometric volatility model. It does not directly learn complex nonlinear relationships among several different market variables.

The Quantile LSTM is therefore used as a second-stage forecasting model.

The overall idea is:

```text
Historical Prices
       ↓
   Returns
       ↓
 ┌───────────────┐
 │ Feature       │
 │ Engineering   │
 └───────────────┘
       ↓
 ┌─────────────────────────────────┐
 │ log_return                      │
 │ rolling_vol                     │
 │ volume_zscore                   │
 │ garch_vol                       │
 └─────────────────────────────────┘
       ↓
 Previous 30 Trading Days
       ↓
     LSTM
       ↓
 ┌─────────────────────────────────┐
 │ Q10   Q50   Q90                  │
 │ Low   Median  High-risk scenario │
 └─────────────────────────────────┘
       ↓
 Future Volatility Forecast
```

The important distinction is:

> **GARCH estimates the current volatility state, while the Quantile LSTM learns from the recent sequence of market information to forecast future volatility.**

---

# 2. Features Used by the LSTM

The model uses four input features:

```python
feature_cols = [
    "log_return",
    "rolling_vol",
    "volume_zscore",
    "garch_vol",
]
```

These features provide different information about the market.

### 2.1 Log Return

The log return represents the daily movement of the asset:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

where:

* \(P_t\) = current price
* \(P_{t-1}\) = previous price

It tells the LSTM whether the asset moved up or down and by approximately how much.

---

### 2.2 Rolling Volatility

Rolling volatility measures the recent amount of price fluctuation.

For a window of \(n\) observations:

$$
\sigma_t^{rolling}
=
Std(r_{t-n+1},...,r_t)
$$

It provides the LSTM with a direct measure of recent market turbulence.

---

### 2.3 Volume Z-Score

The volume z-score measures how unusual the current trading volume is compared with its recent history.

A general form is:

$$
Z_t =
\frac{V_t-\mu_V}{\sigma_V}
$$

where:

* \(V_t\) = current volume
* \(\mu_V\) = historical/rolling mean volume
* \(\sigma_V\) = historical/rolling standard deviation of volume

A large positive z-score indicates unusually high trading activity.

---

### 2.4 GARCH Volatility

The GARCH model produces a conditional variance forecast:

$$
\sigma_t^2
$$

which is converted to volatility using:

$$
\sigma_t = \sqrt{\sigma_t^2}
$$

This value becomes the `garch_vol` feature.

Therefore, GARCH is **not the final forecasting model** here.

Instead:

$$
\boxed{\text{GARCH volatility} \rightarrow \text{LSTM input feature}}
$$

The LSTM can then learn whether the GARCH volatility estimate contains useful information about future volatility when combined with the other market features.

---

# 3. Why Add GARCH to the LSTM?

An LSTM could theoretically be trained without GARCH.

For example, we could use:

```text
log_return
rolling_vol
volume_zscore
```

and let the LSTM learn the volatility dynamics itself.

So why include GARCH?

Because GARCH provides an explicitly modeled representation of **time-varying volatility**.

The two models approach the problem differently:

| Model           | Main role                                                        |
| --------------- | ---------------------------------------------------------------- |
| GARCH           | Models volatility dynamics from historical returns               |
| LSTM            | Learns temporal and nonlinear relationships                      |
| Quantile output | Represents different parts of the future volatility distribution |

The intuition is:

> **GARCH tells the LSTM what the current volatility regime looks like, while the LSTM determines how that information, together with other market signals, relates to future volatility.**

However, this does **not** automatically prove that GARCH improves performance.

The architecture gives us a reason to test GARCH as an additional feature. Whether it actually helps must be determined empirically by comparing models with and without `garch_vol`.

---

# 4. Creating the LSTM Sequences

The model uses a:

$$
\boxed{30\text{-day lookback window}}
$$

For every prediction, the LSTM receives the previous 30 trading days.

Each day contains four features:

$$
4 \text{ features/day}
$$

Therefore, each input sequence has the shape:

$$
\boxed{30 \times 4}
$$

Conceptually:

```text
Day 1   → [return, rolling_vol, volume_zscore, garch_vol]
Day 2   → [return, rolling_vol, volume_zscore, garch_vol]
Day 3   → [return, rolling_vol, volume_zscore, garch_vol]
...
Day 30  → [return, rolling_vol, volume_zscore, garch_vol]
                         ↓
                       LSTM
                         ↓
                 Future volatility
```

The target is:

```python
target_col = "target_future_vol"
```

So the model does not attempt to predict the next stock price.

It predicts **future volatility**.

---

# 5. What Does the LSTM Actually Learn?

The LSTM processes the sequence of the previous 30 days.

For example, it can potentially learn patterns such as:

```text
Large negative returns
        +
Increasing trading volume
        +
Increasing rolling volatility
        +
Increasing GARCH volatility
        ↓
Possible future volatility expansion
```

The advantage of the LSTM is that it does not have to rely only on the most recent observation.

It can learn temporal patterns across the entire 30-day sequence.

For example:

```text
Week 1 → calm market
Week 2 → volatility begins increasing
Week 3 → negative returns appear
Week 4 → volume increases sharply
        ↓
LSTM
        ↓
Higher predicted future volatility
```

This is the central role of the LSTM in the architecture.

---

# 6. Why Quantile Forecasting?

A normal regression model might produce a single prediction:

$$
\hat{y}=2.5\%
$$

But financial volatility is uncertain.

Instead of asking:

> "What will future volatility be?"

the Quantile LSTM asks:

> "What could different points of the future volatility distribution look like?"

The model therefore predicts three quantiles:

$$
\boxed{Q_{10}, Q_{50}, Q_{90}}
$$

using:

```python
quantiles = [0.10, 0.50, 0.90]
```

---

# 7. Meaning of Q10, Q50 and Q90

These values are **not** predictions of `0.10`, `0.50`, and `0.90`.

They represent different conditional quantiles of the predicted future volatility.

For example, suppose the model predicts:

$$
Q_{10}=1.2\%
$$

$$
Q_{50}=2.1\%
$$

$$
Q_{90}=4.8\%
$$

The interpretation is approximately:

| Quantile | Interpretation                      |
| -------- | ----------------------------------- |
| Q10      | Lower-volatility scenario           |
| Q50      | Median or typical scenario          |
| Q90      | Upper-tail/high-volatility scenario |

Therefore:

```text
Q10 = 1.2%     → lower volatility scenario
Q50 = 2.1%     → central/typical forecast
Q90 = 4.8%     → high-volatility scenario
```

---

# 8. Why Q90 Is Important for Risk Detection

If the objective is to identify a possible **high-volatility event in advance**, the upper quantile is particularly useful.

Q90 focuses on the upper portion of the conditional volatility distribution.

Therefore, instead of using only:

$$
Q_{50}
$$

we can monitor:

$$
\boxed{Q_{90}}
$$

because it represents a high-volatility scenario predicted by the model.

The logic is:

```text
Q50
 ↓
"What is the typical expected volatility?"

Q90
 ↓
"What could volatility look like in a high-risk scenario?"
```

This makes Q90 useful for an **early-warning system**.

---

# 9. Q90 Does Not Automatically Mean "High Volatility"

This distinction is extremely important.

A Q90 prediction of 2% might be normal in one asset but extremely high in another.

Therefore, we should not simply say:

$$
Q_{90} > 0
$$

and call it high volatility.

Instead, Q90 should be compared against an appropriate benchmark.

For example, suppose historical volatility has a 95th-percentile threshold of:

$$
V_{95}=4.0\%
$$

and the model predicts:

$$
Q_{90}=4.8\%
$$

Then:

$$
Q_{90}>V_{95}
$$

This can be interpreted as:

> The model's upper-tail forecast is above a historically extreme volatility level.

That provides a more defensible basis for a high-volatility warning.

---

# 10. Identifying a Bearish Market Regime

If the goal is specifically to detect **potential high volatility during a bearish market**, Q90 can be combined with a bearish-market indicator.

A bearish regime could be defined using a market condition such as:

$$
P_t < MA_{50}
$$

or a stronger trend condition such as:

$$
MA_{50}<MA_{200}
$$

The exact definition should be selected and validated as part of the project.

The important point is that **bearishness and high volatility are separate concepts**.

A market can be:

```text
Bearish + Low volatility
```

or:

```text
Bearish + High volatility
```

Therefore, a negative return alone should not automatically trigger a high-volatility warning.

---

# 11. Combining Bearish Regime with Q90

A simple warning framework can therefore be written as:

$$
\boxed{
\text{High Volatility Warning}
=
\text{Bearish Regime}
\land
(Q_{90} > V_{95})
}
$$

For example:

```text
Is the market bearish?
        ↓
       YES
        ↓
Is predicted Q90 above
the historical high-volatility threshold?
        ↓
       YES
        ↓
🚨 Potential High-Volatility Regime
```

The interpretation is:

> The market is currently in a bearish regime, and the model's upper-tail forecast indicates that future volatility may reach an unusually high level.

---

# 12. Comparing Q90 With Current GARCH Volatility

Another useful approach is to compare the predicted Q90 with the current volatility state provided by GARCH.

Suppose:

$$
garch\_vol_t=2.5\%
$$

while:

$$
Q_{90,t+1}=5.0\%
$$

The model is indicating that future volatility could become substantially larger than the current GARCH volatility estimate.

We can define a volatility expansion ratio:

$$
\boxed{
VR_t =
\frac{Q_{90,t+1}}
{garch\_vol_t}
}
$$

In this example:

$$
VR_t=
\frac{5.0}{2.5}
=2
$$

So the upper-tail forecast is approximately **2 times the current GARCH volatility**.

This can be interpreted as a potential volatility-expansion signal.

---

# 13. A Combined Early-Warning Framework

A stronger framework can combine:

1. Market direction
2. Current volatility
3. Predicted upper-tail volatility

For example:

$$
\boxed{
\text{Warning}_t =
\begin{cases}
1,
&
\text{if Bearish Regime}
\land
(Q_{90,t+1}>V_{95})
\land
\left(
\frac{Q_{90,t+1}}
{garch\_vol_t}>k
\right)
\\
0,
&
\text{otherwise}
\end{cases}
}
$$

where \(k\) is a threshold determined using the validation data.

Conceptually:

```text
                 Market Direction
                       ↓
                Bearish Regime?
                  /          \
                NO            YES
                ↓              ↓
             No flag      Check Q90
                               ↓
                    Q90 unusually high?
                          /       \
                        NO         YES
                        ↓           ↓
                     No flag    Compare with
                                GARCH volatility
                                    ↓
                          Strong volatility expansion?
                              /             \
                            NO               YES
                            ↓                 ↓
                         No flag       🚨 Warning
```

---

# 14. What the Three Quantiles Can Be Used For

You do not necessarily have to choose only one quantile.

All three can provide useful information.

### Q10

Represents the lower-volatility side of the forecast.

Useful for identifying potentially calm conditions.

### Q50

Represents the median forecast.

Useful as the central estimate of future volatility.

### Q90

Represents the upper-tail forecast.

Useful for risk monitoring and potential high-volatility warnings.

You can also examine the spread:

$$
\boxed{
Q_{90}-Q_{10}
}
$$

A large spread indicates that the model's predicted future volatility distribution is wide, meaning greater uncertainty around the forecast.

For example:

$$
Q_{10}=1.0\%,\quad
Q_{50}=2.0\%,\quad
Q_{90}=5.0\%
$$

gives:

$$
Q_{90}-Q_{10}=4.0\%
$$

which indicates a much wider predicted range than:

$$
Q_{10}=1.8\%,\quad
Q_{50}=2.1\%,\quad
Q_{90}=2.5\%
$$

where:

$$
Q_{90}-Q_{10}=0.7\%
$$

---

# 15. Pinball Loss

Quantile regression requires a loss function that is appropriate for each quantile.

The Quantile LSTM uses **Pinball Loss**.

For quantile \(q\):

$$
L_q(y,\hat{y})=
\begin{cases}
q(y-\hat{y}), & y\geq\hat{y}\\
(1-q)(\hat{y}-y), & y<\hat{y}
\end{cases}
$$

where:

* \(y\) = actual future volatility
* \(\hat{y}\) = predicted quantile
* \(q\) = target quantile

For example, for Q90:

$$
q=0.90
$$

If the model underpredicts the actual volatility, the error receives a larger penalty.

This encourages the Q90 prediction to sit appropriately high in the conditional distribution.

---

# 16. Why Different Quantiles Have Different Penalties

Consider Q90.

If:

$$
y>\hat{y}
$$

the model has underestimated volatility.

The loss is:

$$
0.90(y-\hat{y})
$$

This is a relatively large penalty.

If:

$$
y<\hat{y}
$$

the model has overestimated volatility.

The loss becomes:

$$
0.10(\hat{y}-y)
$$

So Q90 strongly discourages underestimating high volatility.

This is exactly why Q90 is useful when the objective is **risk-sensitive forecasting**.

---

# 17. Current Architecture

The training pipeline follows:

```text
                    Stock Price Data
                           ↓
                    Feature Engineering
                           ↓
        ┌──────────────────────────────────┐
        │ log_return                       │
        │ rolling_vol                      │
        │ volume_zscore                    │
        │ garch_vol                        │
        └──────────────────────────────────┘
                           ↓
                     30-Day Window
                           ↓
                   Quantile LSTM
                           ↓
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
            Q10           Q50           Q90
             ↓             ↓             ↓
       Low-volatility   Median       High-volatility
         scenario      scenario         scenario
```

The model therefore combines:

$$
\boxed{
\text{Traditional volatility modeling}
+
\text{Deep temporal modeling}
+
\text{Distributional forecasting}
}
$$

---

# 18. Training Configuration

The implementation uses:

```python
LOOKBACK = 30
```

and:

```python
model = build_quantile_lstm(
    lookback_window=LOOKBACK,
    num_features=4,
    quantiles=[0.10, 0.50, 0.90]
)
```

The model is trained using:

```python
epochs = 30
batch_size = 64
```

with validation data and early stopping.

The data is divided chronologically into:

```text
Training Data
      ↓
Validation Data
      ↓
Test Data
```

with the implementation using:

```python
val_start = "2024-06-01"
test_start = "2025-01-01"
```

This preserves the temporal ordering required for forecasting.

---

# 19. The Role of GARCH in the Final Architecture

The complete logic can be summarized as:

### Step 1: GARCH

Use historical returns to estimate the current conditional volatility:

$$
\text{Historical Returns}
\rightarrow
\text{GARCH}
\rightarrow
garch\_vol_t
$$

### Step 2: Feature Combination

Combine GARCH volatility with other market information:

$$
X_t=
[
log\_return_t,
rolling\_vol_t,
volume\_zscore_t,
garch\_vol_t
]
$$

### Step 3: Temporal Sequence

Give the LSTM the previous 30 days:

$$
X_{t-30:t}
$$

### Step 4: Quantile Forecasting

The LSTM produces:

$$
\boxed{
[Q_{10,t+1},Q_{50,t+1},Q_{90,t+1}]
}
$$

### Step 5: Risk Interpretation

Q90 can then be compared with historical volatility thresholds and the current volatility regime to identify potential high-volatility episodes.

---

# 20. The Key Conceptual Distinction

The easiest way to explain the architecture during a presentation is:

> **GARCH answers: "What is the current volatility state based on historical shocks?"**

> **LSTM answers: "Given the recent sequence of market information, what might future volatility look like?"**

> **Quantile forecasting answers: "What do the lower, median, and upper parts of that future volatility distribution look like?"**

And for the bearish-market use case:

> **Q90 can act as an early-warning indicator when its predicted upper-tail volatility is unusually high, particularly when the market is simultaneously in a bearish regime and the predicted volatility represents a substantial expansion over the current volatility state.**

---

# 21. What We Can and Cannot Claim

The architecture gives us a **theoretical reason** to include GARCH:

$$
\text{GARCH}
\rightarrow
\text{explicit volatility-state information}
\rightarrow
\text{LSTM}
$$

But we cannot claim that:

> "GARCH definitely improves the LSTM."

The LSTM could potentially learn much of the same volatility information from `log_return` and `rolling_vol`.

Therefore, the correct interpretation is:

> **GARCH is introduced as an additional volatility-aware feature. Its incremental predictive value must be established empirically by comparing the LSTM with and without the GARCH feature.**

That distinction keeps the project scientifically honest and prevents the rather common academic maneuver of declaring victory before checking whether the extra model actually did anything.

---

# 22. Final End-to-End Interpretation

The complete forecasting system can therefore be described as:

```text
Stock Prices
     ↓
Returns + Volume
     ↓
Feature Engineering
     ↓
 ┌──────────────────────────────────┐
 │ log_return                       │
 │ rolling_vol                      │
 │ volume_zscore                    │
 │ GARCH conditional volatility     │
 └──────────────────────────────────┘
     ↓
Previous 30 Trading Days
     ↓
Quantile LSTM
     ↓
 ┌────────┬────────┬────────┐
 │  Q10   │  Q50   │  Q90   │
 └────────┴────────┴────────┘
     ↓
Future Volatility Distribution
     ↓
Risk Interpretation
     ↓
Q90 + Historical Threshold
     +
Bearish Market Regime
     +
Current GARCH Volatility
     ↓
Potential High-Volatility
Early Warning
```

The fundamental idea is therefore not simply:

$$
\boxed{\text{GARCH} \rightarrow \text{LSTM}}
$$

It is:

$$
\boxed{
\text{GARCH provides volatility-state information}
\rightarrow
\text{LSTM learns temporal relationships}
\rightarrow
\text{Quantile outputs represent uncertainty}
\rightarrow
\text{Q90 can support risk-event detection}
}
$$
