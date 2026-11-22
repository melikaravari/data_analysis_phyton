# Project 1: Energy price dynamics (Brent, WTI, Henry Hub, jet fuel)

**Data:** daily prices, 2010-2026 (4,317 rows; 4,099 days with all four prices observed).

## Questions
1. Are energy prices non-stationary while returns are stationary?
2. Do Brent, WTI and jet fuel share a long-run equilibrium?
3. How do shocks transmit across markets in the short run?

## Method
Log prices and log returns; ADF tests; Johansen trace test and pairwise Engle-Granger tests;
a VECM (rank 2, 5 lagged differences); VAR on returns (lag length by BIC); Granger causality; orthogonalised impulse responses;
forecast error variance decomposition.

## Results
- **Returns are clearly stationary** (ADF p < 0.001 for all). Log prices are borderline: Brent and WTI reject the
  unit root at 5% only just (p = 0.047 and 0.035), jet fuel does not (p = 0.13).
- **Strong long-run co-movement.** Engle-Granger rejects "no cointegration" for Brent~WTI (p = 0.0001),
  Brent~jet fuel (p = 0.008) and WTI~jet fuel (p = 0.003). The Johansen cointegrating vector is close to
  Brent = 1.11 x WTI (jet fuel gets almost no weight).
- **The full-sample Johansen test selects rank 3 out of 3**, which would mean all three log-price series are
  stationary. That conflicts with the borderline ADF results, so I did not rely on it. Splitting the sample gives
  rank 2 in both 2010-2019 and 2020-2026. The honest conclusion is: strong co-movement, but the exact rank is
  sensitive to the sample, probably because of structural breaks (2014-16 price collapse, 2020, 2022).
- **VECM (rank 2):** the two cointegrating vectors link Brent and WTI to jet fuel. WTI adjusts significantly
  (alpha -0.031 on the WTI-jet fuel vector, t = -3.6; +0.021 on the Brent-jet fuel vector, t = 2.3), Brent adjusts weakly
  (-0.017, t = -2.0 on the Brent-jet fuel vector) and jet fuel does not adjust significantly on either vector
  (|t| < 1.6). Statistically, the crude benchmarks move toward the equilibrium and jet fuel is the "anchor"; this describes
  the adjustment pattern and is not a causal claim. Rank 2 follows the subperiod tests, not the full-sample test.
- **Short-run transmission is fast.** In the VAR(4), Brent, WTI and jet fuel returns each Granger-cause the other
  markets (p < 0.001). Henry Hub gas does not (p = 0.97): US gas is not driven by oil returns in this system.
  Impulse responses show shocks passing through within the same day and dying out after 1-2 days.
- 10 days ahead, 42% of the variance in jet fuel returns is explained by Brent shocks and 7% by WTI.

## Limitations
- WTI was negative on 2020-04-20; logs are undefined, so that day is treated as missing.
- Only days where all four markets trade are used; holidays differ between markets.
- The impulse responses depend on the variable ordering (Brent first); a different ordering changes the
  same-day effects. No structural breaks are modelled.
- Jet fuel is quoted in USD per gallon, crude in USD per barrel; logs make this irrelevant for returns.

## Possible extensions
Vector error correction model for the cointegrated block; regime-specific (Markov-switching) models; adding the VIX.
