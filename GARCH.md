# GARCH Volatility Forecasting

## 1. What Are We Trying to Achieve?

The goal of this implementation is to **predict how volatile a stock is likely to be tomorrow**.

We are **not trying to predict tomorrow's stock price**.

For example, suppose a stock is currently trading at ₹100.

We are not asking:

> "Will the stock be ₹105 tomorrow?"

Instead, we are asking:

> "How much movement should we expect from this stock tomorrow?"

This is where **GARCH** comes in.

GARCH is a statistical model used to estimate and forecast **volatility**.

---

# 2. What Is Volatility?

Volatility tells us how much the price of a stock is moving around.

Consider two stocks.

### Stock A

```text
₹100 → ₹101 → ₹100 → ₹102 → ₹101
```

The price is moving only a little.

This means:

**Low volatility**

### Stock B

```text
₹100 → ₹110 → ₹92 → ₹115 → ₹88
```

The price is moving dramatically.

This means:

**High volatility**

Therefore:

> **Volatility = How much and how rapidly the returns of an asset fluctuate.**

---

# 3. Why Do We Need GARCH?

Financial markets have an interesting characteristic called **volatility clustering**.

Large price movements tend to be followed by more large price movements.

Similarly, calm periods tend to be followed by calm periods.

For example:

```text
Day        Return
Day 1       0.2%
Day 2      -0.3%
Day 3       0.4%
Day 4       5.5%   ← Large movement
Day 5      -4.2%   ← Large movement
Day 6       3.8%   ← Large movement
Day 7      -2.9%   ← Large movement
```

Notice that after Day 4, the market remains unstable for several days.

GARCH is designed to capture this behavior.

It essentially learns:

> "When the market experiences large shocks, volatility tends to remain high for some time."

---

# 4. What Does GARCH Predict?

The most important thing to understand is:

**GARCH predicts volatility, not price.**

For example:

```text
Today's information
        ↓
      GARCH
        ↓
Tomorrow's predicted volatility
```

The output might look like:

| Date  | Return | GARCH Volatility |
| ----- | -----: | ---------------: |
| Dec 1 |   0.5% |             1.2% |
| Dec 2 |  -0.8% |             1.4% |
| Dec 3 |   4.5% |             2.7% |
| Dec 4 |  -3.8% |             3.1% |
| Dec 5 |   2.9% |             3.3% |

The model is telling us that after several large movements, the expected volatility has increased.

So:

```text
1.2% volatility → Relatively calm
3.3% volatility → Much more unstable
```

---

# 5. Overall GARCH Pipeline

The complete process can be thought of as:

```text
              STOCK PRICE DATA
                     ↓
              Calculate Returns
                     ↓
          Need at least 252 observations?
                /              \
              NO                YES
              ↓                  ↓
             NaN          Fit GARCH Model
                                ↓
                    Estimate Volatility
                                ↓
                   Forecast Next Period
                                ↓
                         GARCH Volatility
```

The process is repeated as we move forward through time.

---

# 6. Step 1: Obtain Stock Price Data

The first step is to obtain historical stock prices.

Typical data contains:

```text
Date
Open
High
Low
Close
Volume
```

For GARCH, the most important information is the **price series**, because we use it to calculate returns.

For example:

```python
import yfinance as yf

data = yf.download(
    "AAPL",
    start="2018-01-01",
    end="2025-01-01"
)
```

The data might look like:

```text
Date          Open    High    Low     Close
2018-01-02    42.54   43.08   42.31   43.06
2018-01-03    43.13   43.64   42.99   43.06
2018-01-04    43.13   43.37   43.02   43.26
```

---

# 7. Step 2: Calculate Returns

GARCH does not normally work directly with stock prices.

Instead, we calculate **returns**.

A simple return is:

```text
Return = (Today's Price - Yesterday's Price) / Yesterday's Price
```

For example:

```text
Yesterday = ₹100
Today     = ₹105

Return = (105 - 100) / 100
       = 0.05
       = 5%
```

In Python:

```python
data["Return"] = data["Close"].pct_change()
```

The dataset now contains:

```text
Date        Close    Return
Day 1       100      NaN
Day 2       105      5.0%
Day 3       103     -1.9%
Day 4       110      6.8%
```

---

# 8. Why Do We Use Returns Instead of Prices?

Consider a stock that goes:

```text
₹100 → ₹110
```

The increase is ₹10.

Now consider:

```text
₹1,000 → ₹1,010
```

The increase is also ₹10.

But these two movements clearly don't represent the same thing.

In percentage terms:

```text
₹100 → ₹110

Return = 10%
```

while:

```text
₹1,000 → ₹1,010

Return = 1%
```

Returns therefore give us a more meaningful measure of price movement.

---

# 9. Step 3: Convert Returns to the Appropriate Scale

Many GARCH implementations use returns expressed in percentages rather than decimals.

For example:

```text
0.01 → 1%
0.02 → 2%
-0.03 → -3%
```

Therefore:

```python
returns = data["Return"].dropna() * 100
```

Now GARCH receives:

```text
1.2
-0.8
2.3
-4.1
...
```

instead of:

```text
0.012
-0.008
0.023
-0.041
...
```

This scaling is often useful for numerical stability and easier interpretation.

---

# 10. Step 4: The Warm-Up Period

The implementation uses:

```python
min_window = 252
```

Why 252?

There are approximately **252 trading days in a year**.

The model is saying:

> "Before I start forecasting volatility, I want roughly one year of historical observations."

Therefore:

```text
Days 1 → 251
        ↓
Not enough history
        ↓
GARCH forecast = NaN
```

Once we reach Day 252:

```text
Days 1 → 252
        ↓
Enough history
        ↓
GARCH can begin
```

This does **not** mean the first 251 observations are useless.

They are simply not used to produce a GARCH forecast until enough history has accumulated.

---

# 11. Step 5: The Expanding Window

The implementation uses an **expanding window**.

This means that as we move forward through time, we keep all the historical observations and add the newest one.

For example:

```text
Day 252
Use: Days 1 → 252

Day 253
Use: Days 1 → 253

Day 254
Use: Days 1 → 254

Day 255
Use: Days 1 → 255
```

Visually:

```text
Day 252: █████████████████
Day 253: ██████████████████
Day 254: ███████████████████
Day 255: ████████████████████
```

The historical window keeps expanding.

This is different from a rolling window.

### Expanding Window

```text
████████████
█████████████
██████████████
███████████████
```

### Rolling Window

```text
        ████████████
         ████████████
          ████████████
```

With an expanding window, the model remembers the entire available history.

---

# 12. Step 6: Fit the GARCH Model

Once we have enough historical data, we fit a GARCH model.

A typical implementation using the `arch` library looks like:

```python
from arch import arch_model

model = arch_model(
    returns,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1
)

result = model.fit(disp="off")
```

The important part is:

```python
vol="GARCH"
```

This tells Python:

> "Use a GARCH volatility model."

---

# 13. What Does GARCH(1,1) Mean?

You will often see:

```text
GARCH(1,1)
```

Don't let the notation scare you.

The two numbers describe how the model uses previous information.

Conceptually, GARCH considers:

```text
Current volatility
       ↓
depends on
       ↓
Recent shock + Previous volatility
```

More specifically, the model considers:

### 1. Recent shocks

Did the stock experience a large unexpected movement recently?

### 2. Previous volatility

Was the market already volatile?

The model combines these pieces of information to estimate today's volatility and forecast future volatility.

---

# 14. The GARCH Equation

For a basic GARCH(1,1) model:

$$
\sigma_t^2 =
\omega +
\alpha\epsilon_{t-1}^2 +
\beta\sigma_{t-1}^2
$$

Don't worry about memorizing this immediately.

The intuition is:

```text
Current volatility
        =
Baseline volatility
+
Recent shock
+
Previous volatility
```

Where:

* `σ²` = variance/volatility
* `ε²` = size of the previous shock
* `α` = how strongly recent shocks affect volatility
* `β` = how persistent volatility is
* `ω` = baseline component

The important idea is that **volatility has memory**.

---

# 15. Step 7: Forecast Tomorrow's Volatility

After fitting the model, we ask:

> "Based on everything I've seen so far, what should volatility be tomorrow?"

For example:

```python
forecast = result.forecast(horizon=1)
```

The forecast gives us the predicted variance.

We can convert variance into volatility using the square root:

```python
variance = forecast.variance.iloc[-1, 0]

volatility = variance ** 0.5
```

This becomes our:

```text
garch_vol
```

---

# 16. Why Do We Refit Every 10 Days?

The implementation uses:

```python
refit_every = 10
```

This means we don't completely rebuild the GARCH model every single day.

Why?

Because fitting a GARCH model involves mathematical optimization and can be computationally expensive.

Imagine we have:

```text
5 years of data
20 stocks
~1,250 trading days
```

Re-estimating the entire model every day can become expensive.

Instead:

```text
Day 252
    ↓
FIT GARCH
    ↓
Get parameters
    ↓
Forecast

Day 253
    ↓
Use existing model

Day 254
    ↓
Use existing model

...

Day 261
    ↓
Use existing model

Day 262
    ↓
FIT AGAIN
```

So:

> **Refitting = relearning the GARCH parameters.**

> **Forecasting = using the existing parameters to produce the volatility estimate.**

---

# 17. Why Doesn't the Model Need to Be Refit Every Day?

The underlying behavior of volatility usually doesn't change dramatically from one day to the next.

For example:

```text
Monday:
Volatility behavior = moderately persistent

Tuesday:
Probably still moderately persistent

Wednesday:
Probably still moderately persistent
```

Therefore, recalculating the parameters every day may provide little additional benefit relative to the computational cost.

Refitting every 10 days provides a practical compromise.

---

# 18. Important: There Is No Lookahead Bias

One of the most important requirements in forecasting is:

> **Never use future information to predict the past.**

Suppose we're predicting volatility for:

```text
January 20
```

We should only use information that was available **before January 20**.

We cannot use:

```text
January 21
January 22
January 23
```

because those events haven't happened yet.

The implementation moves through the dataset chronologically:

```text
Past data
   ↓
Fit model
   ↓
Forecast tomorrow
   ↓
Move forward
   ↓
New past data
   ↓
Fit/update
   ↓
Forecast again
```

This makes the process much closer to what would happen in a real trading or risk-management environment.

---

# 19. The Complete Loop

The implementation can be understood as this loop:

```text
START
  ↓
Do we have 252 observations?
  ↓
  NO ─────────────→ Store NaN
  │
  YES
  ↓
Is it time to refit?
  ↓
 YES                NO
  ↓                  ↓
Fit GARCH       Keep existing parameters
  │                  │
  └────────┬─────────┘
           ↓
   Forecast volatility
           ↓
     Store garch_vol
           ↓
      Move to next day
           ↓
          REPEAT
```

---

# 20. Example Implementation

A simplified version of the entire process is:

```python
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model


# -----------------------------------------
# 1. Download data
# -----------------------------------------

data = yf.download(
    "AAPL",
    start="2018-01-01",
    end="2025-01-01",
    auto_adjust=True
)


# -----------------------------------------
# 2. Calculate returns
# -----------------------------------------

data["Return"] = data["Close"].pct_change() * 100


# -----------------------------------------
# 3. Parameters
# -----------------------------------------

min_window = 252
refit_every = 10

data["garch_vol"] = np.nan

last_model = None


# -----------------------------------------
# 4. Walk forward through time
# -----------------------------------------

for i in range(min_window, len(data)):

    # Historical data available up to today
    historical_returns = (
        data["Return"]
        .iloc[:i]
        .dropna()
    )

    try:

        # ---------------------------------
        # Refit every 10 observations
        # ---------------------------------

        if last_model is None or (i - min_window) % refit_every == 0:

            model = arch_model(
                historical_returns,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1
            )

            last_model = model.fit(disp="off")


        # ---------------------------------
        # Forecast next-period variance
        # ---------------------------------

        forecast = last_model.forecast(horizon=1)

        variance = forecast.variance.iloc[-1, 0]

        # Convert variance to volatility
        volatility = np.sqrt(variance)

        data.iloc[i, data.columns.get_loc("garch_vol")] = volatility


    except Exception:

        # ---------------------------------
        # Safety fallback
        # ---------------------------------

        fallback_vol = (
            historical_returns
            .rolling(30)
            .std()
            .iloc[-1]
        )

        data.iloc[i, data.columns.get_loc("garch_vol")] = fallback_vol
```

---

# 21. What Does Each Important Variable Mean?

| Variable             | Meaning                                             |
| -------------------- | --------------------------------------------------- |
| `min_window`         | Minimum amount of history needed before forecasting |
| `refit_every`        | Number of observations between GARCH refits         |
| `historical_returns` | Returns available up to the current point           |
| `model`              | GARCH model specification                           |
| `last_model`         | Most recently fitted GARCH model                    |
| `forecast`           | Forecast produced by the model                      |
| `variance`           | Predicted future variance                           |
| `garch_vol`          | Predicted future volatility                         |
| `fallback_vol`       | Backup volatility estimate if GARCH fails           |

---

# 22. What Happens When GARCH Fails?

Statistical models don't always behave nicely.

Sometimes optimization can fail because of:

* unusual market movements
* extreme outliers
* insufficient variation
* numerical problems
* convergence issues

Instead of allowing the entire pipeline to crash, the implementation uses:

```python
try:
    ...
except Exception:
    ...
```

If GARCH fails, the code calculates a simpler volatility measure:

```python
historical_returns.rolling(30).std()
```

This calculates the **30-day rolling standard deviation**.

So the logic is:

```text
Try GARCH
   ↓
Successful?
 /       \
YES       NO
 ↓         ↓
GARCH    Rolling
vol      Std Dev
```

This is a **fallback mechanism**.

---

# 23. Why Use Rolling Standard Deviation as a Fallback?

Standard deviation is a simple measure of volatility.

Suppose the last 30 returns are:

```text
0.5%
-0.8%
1.2%
-0.4%
...
```

We calculate how much those returns vary.

That gives us a simple volatility estimate.

It isn't as sophisticated as GARCH, but it allows the pipeline to continue.

---

# 24. What Does the Final Dataset Look Like?

Eventually, you might have:

| Date  | Close | Return | GARCH Volatility |
| ----- | ----: | -----: | ---------------: |
| Jan 1 |   100 |    NaN |              NaN |
| Jan 2 |   101 |   1.0% |              NaN |
| ...   |   ... |    ... |              ... |
| Dec 1 |   105 |   2.1% |             1.8% |
| Dec 2 |   101 |  -3.8% |             2.5% |
| Dec 3 |   106 |   5.0% |             3.2% |
| Dec 4 |   107 |   0.9% |             3.0% |

The first 252 observations don't have a GARCH forecast because we deliberately waited for enough historical information.

After that, every observation receives a volatility estimate.

---

# 25. How Should We Interpret `garch_vol`?

Suppose:

```text
garch_vol = 1.5%
```

This means the model estimates relatively moderate volatility.

Suppose later:

```text
garch_vol = 5.0%
```

This means the model believes the stock is currently in a much more volatile regime.

The exact interpretation depends on the model specification and whether the volatility is annualized, daily, etc., so the unit must be checked before interpreting the number as a probability or expected price move.

---

# 26. GARCH in One Picture

The whole idea can be summarized as:

```text
                    HISTORICAL RETURNS
                           ↓
                ┌─────────────────────┐
                │                     │
                │   Were movements    │
                │   recently large?   │
                │                     │
                └─────────────────────┘
                           ↓
                 Previous Volatility
                           ↓
                      GARCH(1,1)
                           ↓
             ┌─────────────────────────┐
             │                         │
             │  Expected Volatility    │
             │       Tomorrow          │
             │                         │
             └─────────────────────────┘
                           ↓
                     garch_vol
```

---

# 27. The Most Important Intuition

Think of GARCH as a **volatility memory system**.

It remembers two things:

```text
1. How big were recent shocks?

2. How volatile was the market already?
```

Then it combines those pieces of information to estimate future volatility.

For example:

```text
Calm market
    ↓
Small shocks
    ↓
Low volatility forecast


Big market shock
    ↓
Large recent return
    ↓
Volatility increases


Several large shocks
    ↓
Volatility remains elevated
    ↓
Higher volatility forecast
```

This is the key reason GARCH is useful.

---

# 28. GARCH vs Normal Volatility

A simple rolling standard deviation might say:

> "The last 30 days had this much variation."

GARCH goes one step further:

> "Recent shocks and previous volatility have a relationship, and I can use that relationship to forecast future volatility."

So:

| Method             | Main Idea                                                            |
| ------------------ | -------------------------------------------------------------------- |
| Standard Deviation | How spread out were recent returns?                                  |
| Rolling Volatility | How spread out were the last N returns?                              |
| GARCH              | How does volatility evolve over time, and what might it be tomorrow? |

---

# 29. What Are We Ultimately Getting From GARCH?

At the end of the implementation, we have a new feature:

```text
garch_vol
```

This feature tells us about the **expected volatility/risk regime** of the stock.

It can then be used for:

* Risk management
* Portfolio allocation
* Volatility forecasting
* Option pricing
* Trading strategies
* Financial forecasting models
* Machine-learning features

For example, if you are building a larger forecasting model:

```text
                    STOCK DATA
                       ↓
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
     Price          Volume           Returns
       ↓               ↓                ↓
       └───────────────┼────────────────┘
                       ↓
                  GARCH Model
                       ↓
                GARCH Volatility
                       ↓
              Machine Learning Model
                       ↓
                  Final Prediction
```

GARCH therefore provides the model with information about **how risky the current market environment is**.

---

# 30. Final Summary

The entire implementation can be reduced to six ideas:

### 1. Get stock prices

```text
Price data
```

### 2. Convert prices into returns

```text
Prices → Returns
```

because GARCH models the behavior of returns rather than raw prices.

### 3. Wait until we have enough history

```text
252 trading days
```

before making forecasts.

### 4. Fit GARCH

The model learns how volatility behaves based on:

```text
Recent shocks
+
Previous volatility
```

### 5. Forecast tomorrow's volatility

```text
Historical information
        ↓
      GARCH
        ↓
Tomorrow's volatility
```

### 6. Repeat through time

The model walks forward through the dataset while avoiding future information.

---

# The One Sentence to Remember

> **GARCH is a statistical model that uses the history of returns, especially recent shocks and past volatility, to estimate how volatile a stock is likely to be in the next period.**

And your implementation is essentially doing:

```text
Stock Prices
     ↓
Returns
     ↓
252-day warm-up
     ↓
GARCH(1,1)
     ↓
Forecast future volatility
     ↓
Store as `garch_vol`
     ↓
Repeat through time
```

That is what the entire GARCH implementation is trying to accomplish.

---

# 31. GARCH Formula Summary

The complete mathematical flow of the implementation can be summarized as follows.

## 1. Calculate Returns

First, convert stock prices into returns:

$$
r_t = \frac{P_t-P_{t-1}}{P_{t-1}}
$$

Where:

* $P_t$ = current price
* $P_{t-1}$ = previous price
* $r_t$ = return

---

## 2. Convert Returns to Percentage

The returns are multiplied by 100:

$$
R_t = r_t \times 100
$$

For example:

$$
0.05 \rightarrow 5\%
$$

---

## 3. Calculate the Shock

The shock represents the unexpected part of the return:

$$
\epsilon_t = R_t-\mu_t
$$

Where:

* $R_t$ = actual return
* $\mu_t$ = expected/mean return
* $\epsilon_t$ = unexpected return or shock

The `arch` library estimates these shocks internally.

---

## 4. Square the Shock

GARCH uses the squared shock:

$$
\epsilon_t^2
$$

This measures the **size of the shock** without caring whether the movement was positive or negative.

For example:

$$
(+5)^2 = 25
$$

$$
(-5)^2 = 25
$$

---

## 5. GARCH(1,1) Variance Equation

The main GARCH equation is:

$$
\boxed{
\sigma_t^2 =
\omega+
\alpha\epsilon_{t-1}^2+
\beta\sigma_{t-1}^2
}
$$

In simple terms:

$$
\boxed{
\text{Current Variance}
=
\text{Baseline}
+
\text{Recent Shock}^2
+
\text{Previous Variance}
}
$$

Where:

* $\sigma_t^2$ = current conditional variance
* $\omega$ = baseline variance
* $\alpha$ = effect of recent shocks
* $\epsilon_{t-1}^2$ = previous squared shock
* $\beta$ = volatility persistence
* $\sigma_{t-1}^2$ = previous conditional variance

---

## 6. Convert Variance to Volatility

GARCH produces a variance forecast first.

We convert it to volatility by taking the square root:

$$
\boxed{
\sigma_t=\sqrt{\sigma_t^2}
}
$$

This is the value stored as:

```python
garch_vol
```

---

## 7. Rolling Standard Deviation Fallback

If GARCH fails, the implementation uses the 30-day rolling standard deviation:

$$
\boxed{
s_t =
\sqrt{
\frac{1}{n-1}
\sum_{i=1}^{n}
(R_i-\bar R)^2
}
}
$$

where:

$$
n=30
$$

This provides a simpler volatility estimate.

---

# Complete Formula Flow

The entire GARCH implementation can therefore be represented as:

$$
\boxed{
P_t
\rightarrow
r_t
\rightarrow
R_t
\rightarrow
\epsilon_t
\rightarrow
\epsilon_t^2
\rightarrow
\sigma_t^2
\rightarrow
\sigma_t
\rightarrow
garch\_vol
}
$$

Or in plain English:

```text
Stock Price
     ↓
Calculate Return
     ↓
Convert to Percentage Return
     ↓
Find Unexpected Return (Shock)
     ↓
Square the Shock
     ↓
GARCH combines:
    • Baseline variance
    • Recent squared shock
    • Previous variance
     ↓
Forecast Variance
     ↓
Take √(Variance)
     ↓
GARCH Volatility
     ↓
Store as garch_vol
```

### The Core GARCH Formula

If there is only **one formula** to remember, remember this:

$$
\boxed{
\sigma_t^2 =
\omega+
\alpha\epsilon_{t-1}^2+
\beta\sigma_{t-1}^2
}
$$

It means:

> **Today's expected variance depends on a baseline level, yesterday's shock, and yesterday's variance.**

Then:

$$
\boxed{
\text{Volatility}=\sqrt{\text{Variance}}
}
$$

So the fundamental idea behind the entire implementation is:

> **Recent large shocks increase volatility, and high volatility tends to persist.**

