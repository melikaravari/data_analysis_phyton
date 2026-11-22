# Project 9: SARIMA forecasting of German electricity prices with Diebold-Mariano tests

**Data:** daily average of the hourly German prices, 2019-01-01 to 2026-07-22 (2,760 days). Training 2019-2024 (2,192 days),
test 2025-01-01 to 2026-07-22 (568 days).

## Questions
1. What do unit-root tests and ACF/PACF say about the series?
2. Which SARIMA model is selected, and are its residuals white noise?
3. Does SARIMA beat naive benchmarks out of sample, and is the difference statistically significant?

## Method
ADF and KPSS tests, ACF/PACF plots, a grid of 15 SARIMA specifications (weekly seasonality, period 7) compared by AIC on the training
data, Ljung-Box residual test, one-day-ahead forecasts on the test period with parameters fixed at their training estimates (the
filter is updated with each new observation), and Diebold-Mariano tests (squared error loss, Newey-West variance) against three
benchmarks.

## Results
- Weekday means: about 101-107 EUR/MWh Monday to Thursday, 98 on Friday, 81 on Saturday and 69 on Sunday. The series is persistent:
  the ADF test rejects a unit root only marginally (p = 0.048) while KPSS rejects stationarity (p = 0.01), so both tests point to
  non-stationarity in levels. After a 7-day seasonal difference the unit root is clearly rejected.
- **Selected model:** SARIMA(1,1,2)(0,1,1)7 (AIC 21,039). The seasonal MA coefficient is -0.93, close to a unit root,
  which is what a strong, stable weekly pattern looks like after seasonal differencing.
- **Residuals:** Ljung-Box does not reject at lags 7 and 14 (p = 0.59, 0.21) but rejects at lag 28 (p = 0.004), so the model leaves some
  autocorrelation. Residuals are heavy-tailed (excess kurtosis 7.6): price spikes.
- **Forecast accuracy (568 test days):**

| Model | MAE | RMSE |
|---|---|---|
| SARIMA | 18.66 | 25.28 |
| Naive (yesterday) | 22.12 | 29.72 |
| Mean of last 7 days | 23.45 | 30.59 |
| Seasonal naive (same day last week) | 28.91 | 39.11 |

  Diebold-Mariano tests reject equal accuracy against all three benchmarks (statistics between -5.1 and -7.8, p < 0.001):
  SARIMA's MAE is 16% below the naive forecast.

## Limitations
- Fixed parameters and one-day-ahead forecasts only; no rolling re-estimation and no multi-day horizons.
- The grid of orders is small, and the remaining residual autocorrelation suggests the model is not complete.
- No exogenous variables (load, wind, solar, gas prices); squared-error loss is dominated by price spikes.
- The DM test treats the forecasts as stationary; the test period has a different price level from the training period.

## Possible extensions
SARIMAX with exogenous variables, multiple seasonalities (weekly and annual), rolling re-estimation, comparison with the hourly models
(see project 3, section on the daily comparison), prediction intervals.
