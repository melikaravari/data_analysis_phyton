# Project 8: Jet fuel and Brent - error correction and asymmetric pass-through

**Data:** daily Brent (USD/barrel) and jet fuel (USD/gallon), 2010-01-04 to 2026-07-20, 4,104 days with both prices.

## Questions
1. How large and persistent is the jet fuel crack spread over Brent?
2. How fast does jet fuel return to its long-run relation with Brent?
3. Is pass-through asymmetric ("rockets and feathers": faster for price increases than decreases)?

## Method
Jet fuel converted to USD/barrel (x 42) to define the crack spread. Engle-Granger two-step approach: long-run
regression of log jet fuel on log Brent, then an error-correction model for daily jet fuel returns with 0-5 lags of Brent
returns, 2 lags of own returns and HAC standard errors. Asymmetry tested by splitting Brent returns into rises and falls
(Wald test on cumulative pass-through) and by splitting the equilibrium error into positive and negative parts. Results
repeated before and after 2020.

## Results
- **Crack spread:** mean 16.5 USD/bbl, very persistent (autocorrelation 0.98 at lag 1, 0.62 at lag 60). It was
  low in 2020 (4.5), spiked in 2022 (mean 40.5, max 107) and was high again in 2026 (mean 42.1 so far).
- **Long-run relation:** the elasticity of jet fuel to Brent is 1.00 (R-squared 0.92).
- **Error correction:** jet fuel closes about 1.8% of the gap to equilibrium per day (t = -3.0), a half-life of about 37 days.
- **Short-run pass-through:** a 1% Brent move is followed by a 0.62% same-day jet fuel move and 0.79% cumulatively
  over six days.
- **No evidence of asymmetry:** cumulative pass-through is 0.73 for Brent rises and 0.85 for falls (Wald p = 0.34), and the speed of
  adjustment is -0.015 above and -0.026 below equilibrium (p = 0.54). In the subperiods the numbers differ
  (e.g. adjustment speed -0.031 in 2010-19 vs -0.020 in 2020-26), but I did not test those differences formally.

## Limitations
- A null result for asymmetry is not proof of symmetry: the test may have low power. With daily data, timing differences
  between the two price quotes matter.
- The long-run relation is estimated once on the full sample, so subperiod results use full-sample information; breaks in
  the crack spread (2020, 2022) argue for a time-varying long-run relation.
- Brent is a crude benchmark, not the price airlines pay; no other cost drivers (taxes, FX, refining margins) are modelled.
- One lag structure; no sensitivity analysis.

## Possible extensions
Threshold ECM, rolling estimation of the long-run relation, adding refinery margins or the USD/EUR exchange rate.
