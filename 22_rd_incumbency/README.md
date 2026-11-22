# Project 22: Regression discontinuity - the incumbency advantage in US Senate elections

**Data:** US Senate elections 1914-2010 (1,390 races; 1,297 with a next-election result), from the `rdrobust` package (Lee 2008; Cattaneo, Idrobo and Titiunik 2020).
Running variable: Democratic margin of victory in election t. Outcome: Democratic vote share in the next election for the same seat, six years later.

## Question
Does winning a Senate seat raise a party's vote share in the next election for that seat? In close races the winner is nearly random, so the jump at margin = 0
measures the incumbency advantage.

## Method
Local linear regression with a triangular kernel and MSE-optimal bandwidth, conventional and robust bias-corrected inference (`rdrobust`); the density test for
manipulation of the running variable (`rddensity`); covariate balance and placebo-cutoff tests; sensitivity to bandwidth, polynomial order, kernel, a donut hole and the
bandwidth selector.

## Results
- **Incumbency advantage: 7.4 percentage points** (SE 1.46); bias-corrected 7.5, robust 95% CI [4.1, 10.9], p < 0.0001; bandwidth 17.8 points (360 / 323 races on each
  side). This matches Lee (2008). Winning also raises the probability of winning the next election by 33 percentage points.
- **Robust to specification:** estimates between 7.1 and 8.4 points for bandwidths from half to double the optimal one, a local quadratic (8.05), a uniform kernel
  (7.2), a donut hole (7.1) and the CER-optimal bandwidth (7.6).
- **No evidence of manipulation:** density test p = 0.38; of 102 races decided by under 2 points, the Democrat won 52.
- **Two warning signs:** one of six predetermined covariates jumps at the cutoff (open seat, p = 0.02), and one of four placebo cutoffs (-10) shows a spurious jump of
  -23 points (p = 0.005). With ten tests, some chance rejections are expected, but I do not treat the design as perfectly clean; I did not investigate the cause.

## Limitations
- A local estimate for close races, for the party (not the person); the next election is six years later; the sample near the cutoff is about 700 races.
- The covariate imbalance and the placebo result above; results for the House or other periods may differ.

## Possible extensions
Covariate-adjusted RD, comparison with the House of Representatives, fuzzy RD, heterogeneity over time (incumbency advantage by decade), randomisation-inference RD
in a narrow window.
