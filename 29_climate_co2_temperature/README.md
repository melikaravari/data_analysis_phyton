# Project 29: Climate time series - CO2 growth, global warming trends and spurious regression

**Data:** monthly atmospheric CO2 at Mauna Loa, March 1958 to September 2026 (NOAA; via the Datahub "co2-ppm" repository; the file header does not match its seven data fields, so columns are
read by position), and monthly global temperature anomalies from NASA GISTEMP (1880-2025) and NOAA GCAG (1850-2025; used as a robustness check). Download with `download_public_data.py`.

## Questions
1. How fast is CO2 rising, is the growth accelerating, and how does the seasonal cycle behave?
2. How large is the warming trend, how much does ignoring autocorrelation understate its uncertainty, and is there acceleration?
3. How do temperature and CO2 relate, and why are regressions in levels misleading?
4. Which simple model forecasts CO2 best out of sample?

## Method
STL decomposition; quadratic-trend regression with HAC errors; trend estimation with naive, HAC and AR(1)-GLS standard errors; a one-knot piecewise-linear model; ADF and KPSS tests;
levels versus first-difference regressions and an Engle-Granger cointegration test; forecasts of CO2 from 2010 onward from models trained to 2009 (linear and quadratic trend with month effects,
SARIMA, damped Holt-Winters).

## Results
- **CO2 rose from 316 ppm (1958) to 426 ppm (2026), and its growth is accelerating:** annual growth 0.8 ppm/yr in the 1960s, 1.6 in the 1980s, 2.4 in the 2010s and 2.65 in the 2020s so far. The
  quadratic trend implies growth rising from 0.75 to 2.59 ppm/yr (a factor of 3.5; HAC t = 39). The seasonal cycle peaks in May and its amplitude grew from 5.9 to 6.8 ppm (+0.15 ppm per decade).
- **Warming trend:** 0.207 C per decade since 1975 (HAC 95% CI 0.180-0.233), identical for NASA and NOAA; 0.083 C per decade over 1880-2025. Residuals are strongly autocorrelated, so the naive
  standard error understates the uncertainty by about a factor of 2.5 over the full period (0.004 vs 0.010 HAC).
- **Acceleration:** a piecewise model with its knot at 1975 has slopes of 0.036 C per decade before and 0.205 after; the quadratic curvature term is positive (HAC t = 8.8). The knot is chosen from
  the data, so p-values for it are not valid.
- **Unit-root tests:** ADF does not reject a unit root (p = 0.98) and KPSS rejects trend-stationarity (p = 0.01): the series is non-stationary even around a linear trend, consistent with a
  nonlinear or stochastic trend.
- **Temperature vs CO2:** the levels regression gives 2.7 C per doubling of CO2 with R-squared 0.93; in first differences the R-squared is only 0.13 and the coefficient is imprecise. The Engle-Granger
  test rejects no-cointegration (p < 0.001). This is a statistical exercise, not an estimate of climate sensitivity (aerosols, solar, volcanoes and ocean lags are omitted).
- **Forecasts (2010-2026, trained to 2009):** quadratic trend RMSE 2.1 ppm; SARIMA 5.8; damped Holt-Winters 13.1; linear trend 14.3. Models without acceleration underpredict increasingly.

## Limitations
- Statistical description only; no physical model and no attribution of causes; one measurement site for CO2; the most recent values may be provisional; the knot of the piecewise model is data-chosen.

## Possible extensions
Regression with ENSO, volcanic and solar covariates, structural time series (local linear trend with Kalman filter, project 5), regional temperature data, a Bayesian change-point model.
