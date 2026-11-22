# Project 6: Value-at-Risk and Expected Shortfall backtesting

**Data:** daily SPX, DJI, NDX, RUT, 2000-2019 (4,944 returns). First forecast after 500 days of history, leaving
4,444 forecasts per index.

## Questions
1. Which method produces VaR forecasts with the right number of violations?
2. Do violations cluster in time?
3. Is the Expected Shortfall (ES) of the best model consistent with realised losses?

## Method
1-day VaR at 99% and 97.5% from three methods, all using only past data: historical simulation (500-day window);
a normal distribution with RiskMetrics EWMA volatility; GJR-GARCH(1,1,1) with Student-t errors, re-estimated every
250 days. Backtests: Kupiec test (right number of violations), Christoffersen independence test (no clustering) and
the combined conditional coverage test, all at the 5% level.

## Results
- **The normal EWMA model is rejected in all 8 index/level combinations by both tests.** Its violation rates are about
  twice the target at 99% (1.96% to 2.18%) and 3.5% to 3.8% at 97.5%: normal tails are too thin.
- **Historical simulation** is rejected in 7 of 8 Kupiec tests and 8 of 8 conditional coverage tests. It reacts too slowly, and violations cluster
  (independence rejected for most cases).
- **GJR-GARCH-t is clearly best but not perfect:** 5 of 8 Kupiec rejections and 3 of 8 conditional coverage rejections.
  Violation rates are still above target at 99% (1.28% to 1.42%) but close to target at 97.5% (2.72% to 3.13%);
  NDX and RUT pass most tests, SPX and DJI do not.
- **ES check:** on violation days the average realised return of the GARCH model was close to its average forecast ES,
  for example SPX -2.68% vs -2.54% at 99%, NDX -3.07% vs -2.81%, so ES is slightly too optimistic. This is a rough
  comparison, not a formal ES backtest.

## Limitations
- The sample contains the 2008 crisis, which produces many violations in clusters; results for calmer periods may differ.
- Violation counts are small (44 expected at 99%), so the tests have limited power.
- Only one model of each type was tried; no filtered historical simulation, no skewed-t, no formal ES test (e.g. Acerbi-Szekely).
- Indices only, not a portfolio with changing weights.

## Possible extensions
Filtered historical simulation, skewed-t errors, a formal ES backtest, dynamic quantile test, regulatory traffic-light analysis.
