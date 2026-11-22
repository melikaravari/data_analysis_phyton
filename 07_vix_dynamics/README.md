# Project 7: VIX dynamics - mean reversion, regimes and forecasting

**Data:** daily VIX 1990-01-02 to 2020-10-30 (7,767 days); S&P 500 index levels 2000-01-03 to 2019-08-28 for the
return regression.

## Questions
1. How persistent is the VIX; what is the half-life of a shock?
2. Are there distinct low- and high-volatility regimes?
3. How does the VIX react to S&P 500 returns, and is the reaction asymmetric?
4. Does a HAR-type model beat an AR(1) and a random walk out of sample?

## Method
ADF tests and an AR(1) on log VIX; a two-regime Markov-switching model (own mean and variance per regime) on weekly
log VIX; a daily regression of VIX changes on positive and negative S&P 500 returns with HAC standard errors;
forecasting at 1 and 22 days with a random walk, AR(1) and a HAR model (daily, weekly and monthly averages), estimated
up to 2009 and evaluated from 2010 on.

## Results
- **Mean reversion:** AR(1) coefficient on log VIX 0.983, implying a long-run VIX near 18 and a half-life of about 40
  trading days (roughly two months). ADF rejects a unit root in both log VIX and VIX (p < 0.001), although the process
  is highly persistent.
- **Regimes:** two regimes, a calm one (VIX around 14) and a stressed one (around 24), each lasting over a year on average
  (about 73 and 67 weeks). The stressed regime is active about 46% of the time; it covers 2008-09 and March-April 2020, and is
  inactive in 2017. (My first attempt, a Markov-switching AR(1), produced short-lived regimes that mostly separated noisy from
  quiet weeks, so I switched to this specification. The choice is a judgement call; a third regime was not identified.)
- **Asymmetric reaction:** R-squared 0.54. A 1% S&P 500 fall raises the VIX by about 4.8% on average, a 1% rise lowers it
  by about 3.7%; the difference is significant (p = 0.0002). The coefficients are for the log change in VIX.
- **Forecasting (2010-2020):** at 1 day all three models are practically equal (RMSE 0.0787, 0.0784, 0.0782 in log VIX).
  At 22 days the HAR model is best: RMSE 0.253 vs 0.256 (AR(1)) and 0.270 (random walk), about 6% better than the random walk.

## Limitations
- The return regression covers 2000-2019 only; the VIX series extends to 2020.
- Parameters are estimated once, not re-estimated on a rolling basis; no formal test of equal forecast accuracy
  (e.g. Diebold-Mariano), so the 22-day improvement has no p-value.
- Markov-switching estimates depend on the specification and the starting values.
- The VIX is a risk-neutral measure; forecasting it is not forecasting realised volatility.

## Possible extensions
Diebold-Mariano tests, rolling estimation, VIX futures term structure, adding realised volatility to the HAR model.
