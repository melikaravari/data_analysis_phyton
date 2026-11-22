# Project 3: German hourly electricity prices

**Data:** hourly prices in EUR/MWh, 2019-01-01 to 2026-07-22 (66,232 hours; 8 single missing hours, consistent
with one spring daylight-saving change per year, filled by interpolation).

## Questions
1. How do prices vary by hour, year and how often are they negative?
2. Can calendar effects plus lagged prices beat naive day-ahead forecasts?
3. Does gradient boosting add value over the linear model?

## Method
Descriptive statistics by year and hour. Forecast target: the price in hour t, using only information that would
be available the day before (lags of 24, 48 and 168 hours, previous-day mean/min/max, hour, weekday, month and an
hour x weekend interaction). Training period 2019-2024, test period 2025-01 to 2026-07 (13,632 hours).
Models: two naive benchmarks, ridge regression, histogram gradient boosting. A classical benchmark was added at the daily
level: SARIMA(1,1,2)(0,1,1)7 on the daily average price (order chosen by AIC in project 09, parameters fixed after training).

## Results
- **Price regimes differ strongly by year:** mean price 38 EUR/MWh in 2019, 235 in 2022 (energy crisis), 79-99
  afterwards.
- **Negative prices are becoming more common:** 2.4% of hours in 2019, 6.5% in 2025 and 7.0% in 2026 so far
  (573 hours in 2025). They concentrate around midday: 12% of hours at 13:00 to 14:00 are negative against under
  1% in the evening, which points to solar generation (not tested directly: no solar data in this project).
- **Forecasting (test period):**

| Model | MAE | RMSE |
|---|---|---|
| Same hour yesterday | 27.28 | 43.06 |
| Same hour last week | 34.16 | 52.41 |
| Ridge, calendar + lags | 24.83 | 37.02 |
| Gradient boosting | 24.56 | 37.02 |

  The regression models reduce MAE by about 28% relative to the weekly naive forecast and by about 9% relative to
  the best naive one. Gradient boosting is only marginally better than the linear model.

- **Daily level, SARIMA vs the hourly models (568 test days):** SARIMA has MAE 18.67 and RMSE 25.28; the hourly
  gradient boosting model averaged to daily gets 18.66 and 25.19, the ridge model 19.15 and 25.40, and the naive
  "yesterday" forecast 22.12 and 29.72. A seasonal ARIMA on one daily series is as accurate as the more complex hourly
  models, and I did not test whether these small differences are significant (see project 09 for Diebold-Mariano tests).

## Limitations
- No fundamentals (load, wind, solar, gas and CO2 prices), so price spikes and weather-driven negative prices are
  not anticipated. The distance between the models is small compared to the remaining error.
- The training period contains the 2021-22 crisis, whose price level is very different from 2025-26.
- One train/test split, no rolling re-estimation, no probabilistic forecasts.

## Possible extensions
Add weather/load/gas price inputs, rolling re-estimation, quantile regression for prediction intervals.
