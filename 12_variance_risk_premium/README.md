# Project 12: The variance risk premium - does the VIX help to predict returns?

**Data:** daily S&P 500 and VIX, 2000-01-03 to 2019-08-28 (4,945 days).

## Questions
1. How large is the variance risk premium (VRP) and when is it negative?
2. Do VRP and the VIX level predict returns at horizons of 1, 5, 21 and 63 days (in sample)?
3. Out of sample, does a VRP-based forecast beat the historical mean?

## Method
VRP = VIX squared minus 21-day realised variance (annualised percent squared). Predictive regressions of the future h-day return
on VRP, on the VIX level, and on VRP + VIX + past return, with HAC standard errors. Out of sample: expanding window starting 2010,
forecasts every fifth day, training only on observations whose target is already known; evaluation by the Campbell-Thompson
out-of-sample R-squared against the historical mean and the Clark-West test for nested models.

## Results
- **The premium exists on average:** mean VRP 95 percent squared, mean VIX 19.6% vs mean 21-day realised volatility 16.1%.
  VRP is negative on 15% of days, typically in crises.
- **In sample, VRP predicts returns; the VIX level does not.** The VRP coefficient has t-statistics of 2.5-2.8 at horizons 1, 5 and 21
  days (R-squared 0.6%, 1.6%, 2.7%) and t = 6.4 at 63 days (R-squared 5.0%). The VIX level alone has |t| < 1 and an R-squared at or below 0.06%
  at every horizon. The 63-day t-statistic is likely too high because of the heavy overlap in the returns.
- **Out of sample (2010-2019), the evidence is weaker:** VRP gives a positive R2_OS at 1 day (1.26%, Clark-West p = 0.045), 5 days
  (1.56%, p = 0.012) and 21 days (2.59%, p = 0.052), but a negative R2_OS at 63 days (-1.03%). The VIX level never helps (all
  R2_OS <= 0.1%, and negative at 21 and 63 days).

## Limitations
- Out-of-sample R-squared that exceeds the in-sample R-squared (1-day horizon) is unusual and suggests that the result depends on
  the 2010-2019 period; I would not treat it as a stable relation.
- Overlapping returns, short effective samples at long horizons and forecasts every fifth day make all p-values approximate.
- Only one definition of realised variance, no trading rule, no transaction costs, no tests after 2019.
- Return predictability results in the literature are fragile; this project documents one pattern in one sample.

## Possible extensions
Other realised variance measures (e.g. intraday), a trading strategy with costs, other markets, the sample 2020-2026 as a
second out-of-sample test.
