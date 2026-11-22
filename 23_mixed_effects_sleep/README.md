# Project 23: Mixed-effects models for repeated measurements - sleep deprivation and reaction time

**Data:** `sleepstudy` (Belenky et al. 2003; from the R package lme4): 18 drivers restricted to 3 hours of sleep per night for 9 nights; average reaction time (ms)
measured daily on days 0-9 (180 observations). Download with `download_public_data.py`.

## Questions
1. How much slower do reaction times get per day, and how much does this vary between people?
2. Why is pooled OLS misleading, and what do random intercepts and slopes change?
3. Is the random slope needed? Which approach predicts later days best?

## Method
Pooled OLS (naive and cluster-robust), a two-stage analysis of individual slopes, linear mixed models with random intercepts and with random intercepts and slopes
(statsmodels, REML), likelihood-ratio tests on ML fits with boundary-corrected p-values, shrinkage of individual slopes, and an out-of-sample check predicting days 5-9
from days 0-4.

## Results
- **Average effect: +10.5 ms per day** without sleep (reaction time rises from 257 ms at day 0 to 351 ms at day 9). The mixed-model results (intercept 251.4, slope 10.47, SD of
  intercepts 24.7, SD of slopes 5.9, residual SD 25.6, correlation 0.07) agree with the published lme4 values.
- **Pooled OLS understates the uncertainty:** SE 1.24 (CI 8.0-12.9) naive vs 1.55 (CI 7.4-13.5) in the mixed model, cluster-robust OLS, and the mean of the 18 individual slopes.
  The point estimate is identical because the design is balanced.
- **People differ a lot:** individual slopes range from -2.9 to 21.8 ms/day; the random slope improves the fit strongly (LR = 42.1, p < 0.001; AIC 1764 vs 1802 for random intercepts only and
  1904 for OLS).
- **Shrinkage:** the mixed model pulls individual slopes toward the mean (SD of slopes 6.6 to 5.5).
- **Prediction of days 5-9 from days 0-4:** RMSE 51.1 ms for the mixed model, 57.4 for separate lines per person and 58.8 for the pooled line, so partial pooling predicts best.

## Limitations
- 18 professional drivers; a linear trend over days is assumed; constant residual variance and no autocorrelation within persons; normal random effects.
- The residuals are not normal (Shapiro-Wilk p < 0.001) and the spread of reaction times grows over the days; statsmodels offers no small-sample degrees-of-freedom
  correction, so intervals are somewhat too narrow.

## Possible extensions
Heteroskedastic or autoregressive residuals, nonlinear time trends, Bayesian multilevel model, bootstrap inference, comparison with `lme4` and `nlme`.
