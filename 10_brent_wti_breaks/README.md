# Project 10: The Brent-WTI spread - structural breaks and a VECM

**Data:** daily Brent and WTI, 2010-01-04 to 2026-07-20 (4,106 days with both prices; one day with negative WTI removed).

## Questions
1. Is the log spread stationary, with and without allowing for a structural break?
2. When does the level of the spread break, and does that match known events?
3. Which price adjusts to the equilibrium, and does the VECM differ before and after the break?

## Method
Spread = 100 x (log Brent - log WTI). ADF and Zivot-Andrews tests, a scan of Chow F statistics for a single break in the mean
(15% trimming), an OLS-CUSUM stability test, and VECMs (lag order by AIC, Johansen trace rank) on the full sample and on the sub-samples
before and after the estimated break.

## Results
- **The spread is stationary** (ADF stat -4.33, p < 0.001), but its mean has shifted: yearly averages go from 0.1% in 2010 to 16-17% in
  2011-12 (Brent trading well above WTI), then down to 1-9% in most later years.
- **Break date:** Zivot-Andrews estimates 2013-03-06 (p = 0.002), the Chow scan 2013-04-30 (F = 590). The mean spread is 11.6% before
  and 6.3% after. The OLS-CUSUM test rejects parameter stability (p < 0.001). The 2013 date is consistent with the narrowing of the WTI
  discount in 2013 as US pipeline and rail capacity grew (the economic story is my interpretation and I did not test it);
  the 2015 export ban removal is not the main break in this data.
- **VECM, full sample:** rank 2 is selected, which cannot be right for a two-variable system whose spread is stationary
  unless both log prices are stationary (they are not clearly so); the Johansen test is sensitive here. With rank 1 imposed, the
  cointegrating vector is Brent = 1.00, WTI = -1.09, and WTI adjusts significantly (alpha +0.021, t = 2.5) while Brent does not
  (alpha -0.008, t = -1.1).
- **After the break (from 2013-04-30)** (rank 1 imposed again, although Johansen selects rank 2): same pattern with stronger adjustment: vector Brent 1.00, WTI -1.03; Brent alpha -0.025
  (t = -1.9), WTI alpha +0.031 (t = 2.2). WTI adjusts to Brent, which fits Brent being the international benchmark.
- **Before the break:** only 831 days and no cointegration found: before 2013 the two prices were not tied together at the 5% level.

## Limitations
- Only one break is searched for; the spread probably has several regimes (2011-14 discount, 2020, 2022). I did not use a formal
  multiple-break procedure (Bai-Perron) and the Chow scan gives a date, not a valid p-value.
- The pre-break sample is short, so "no cointegration" may be a power problem.
- Johansen rank decisions depend on the lag length (AIC chose 10 lags on the large samples).
- The restriction that the cointegrating vector is a pure spread (1, -1) was not tested.

## Possible extensions
Bai-Perron multiple-break tests, a threshold or regime-switching VECM, testing the (1, -1) restriction, a time-varying
cointegrating relation estimated with a Kalman filter (see project 5).
