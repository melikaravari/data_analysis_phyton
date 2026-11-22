# Project 11: Market betas, their stability and a Fama-MacBeth test (COVID year)

**Data:** daily prices for 491 stocks (cleaned as in project 04), 2019-01-03 to 2020-11-12. Betas are estimated on 2019
(255 days) against an equally weighted market proxy; tests use 2020 (223 days).

## Questions
1. How stable are betas from a calm year to the crash year?
2. Did high-beta stocks lose more in the crash and gain more in the rebound?
3. Fama-MacBeth: was beta rewarded in the 2020 cross-section?

## Method
OLS betas on the equally weighted market; regression of 2020 betas on 2019 betas; quintile portfolios formed on 2019 beta (equally
weighted, returns aggregated from daily log returns); daily Fama-MacBeth regressions of returns on beta with Newey-West
t-statistics (5 lags), for the whole year and for three phases (crash, rebound, rest).

## Results
- **Betas are only moderately stable:** correlation between 2019 and 2020 betas 0.46 (R-squared 0.22), slope 0.37, so betas
  regress strongly toward 1 (the average beta is 1.00 by construction). 49% of the stocks' betas moved by more than 0.3.
- **Beta sorts behave as theory predicts within each phase:** in the crash (2020-02-19 to 2020-03-23) the lowest-beta quintile fell
  33.6% and the highest 48.6%; in the rebound (2020-03-24 to 2020-06-08) the lowest rose 38.1% and the highest 71.4%. Over
  the whole of 2020 the pattern is not monotone (Q1 -0.7%, Q5 -9.7%), so the high-beta stocks lost more over the full year.
- **Fama-MacBeth:** the average daily slope on beta over 2020 is -0.018 (t = -0.20): no evidence of a priced beta over the whole year.
  The slope follows the market's direction: -0.74 per day in the crash (t = -1.85), +0.29 in the rebound (t = 1.43), but neither is
  significant at 5%.

## Limitations
- The market proxy is an equally weighted average of the surviving stocks, not a value-weighted index; betas depend on that choice.
- One episode (2020), 223 days, survivorship bias, no size/value/momentum controls.
- Daily Fama-MacBeth slopes with only 24 crash days are imprecise; the 2019 betas are estimated with error (errors-in-variables),
  which biases the slopes toward zero. I did not use portfolio-level betas (Fama-MacBeth 1973) to reduce this.
- Cumulative portfolio returns are computed from averaged daily log returns, which is an approximation of daily rebalanced returns.

## Possible extensions
Portfolio-level betas, a value-weighted market and size/value factors (Fama-French data from WRDS), Scholes-Williams or Dimson betas for
non-synchronous trading, rolling betas.
