# Project 2: Volatility modelling for the S&P 500 and the VIX

**Data:** daily index levels 2000-01-03 to 2019-08-28 (4,945 days) for SPX, DJI, NDX, RUT, and the VIX.
The models use SPX returns.

## Questions
1. Do GARCH-type models capture volatility clustering and the leverage effect?
2. Does an asymmetric model with fat-tailed errors beat a plain GARCH(1,1)?
3. How closely does model volatility track the VIX?
4. Out of sample, does GARCH beat simple benchmarks?

## Method
Daily log returns (%); GARCH(1,1) with normal and Student-t errors; GJR-GARCH(1,1,1) with Student-t errors;
comparison by AIC/BIC. Out of sample: parameters estimated on data before 2018, then one-day-ahead variance
forecasts for 2018-01 to 2019-08 compared with a RiskMetrics EWMA (lambda = 0.94) and a 250-day rolling variance,
using MSE and QLIKE against squared returns.

## Results
- Returns show excess kurtosis of 8.5 and a lag-1 autocorrelation of absolute returns of 0.25: fat tails and clear
  volatility clustering.
- **Model ranking (AIC):** GARCH normal 13,495 > GARCH-t 13,279 > GJR-GARCH-t 13,105. Student-t errors and the
  asymmetry term both improve the fit a lot. Estimated degrees of freedom: about 7.2.
- **Leverage effect:** the GJR asymmetry parameter is 0.194 (t = 8.4). In this fit the symmetric ARCH term is about
  zero, so almost all of the reaction to shocks comes from negative returns. Persistence is about 0.987.
- **VIX:** the correlation between GJR-GARCH annualised volatility and the VIX is 0.90. Mean model volatility is
  16.7% versus a mean VIX of 19.6%, consistent with the VIX containing a volatility risk premium
  (I did not test this explicitly).
- **Out of sample (2018-01 to 2019-08, about 420 days):** QLIKE 0.75 for GJR-GARCH versus 0.96 for EWMA and
  1.32 for the rolling variance, so GARCH is best; MSE differences are small (4.68 vs 4.97 vs 5.31).

## Limitations
- One out-of-sample window with no major crisis; results may differ in turbulent periods. I did not run a formal
  forecast comparison test (e.g. Diebold-Mariano).
- Squared returns are a noisy proxy for true variance. Realized volatility from intraday data would be better.
- Parameters are fixed after 2017 rather than re-estimated on a rolling basis.

## Possible extensions
EGARCH, realized-volatility models (HAR), VIX as an exogenous regressor, Diebold-Mariano tests.
