# Project 14: Monte Carlo study - when can you trust standard errors and confidence intervals?

**Data:** none; all data are simulated (20,000 replications per setting). Model: y = 1 + 0.5 x + e. The null hypothesis H0: slope = 0.5
is true, so the rejection rate is the actual size of the nominal 5% test.

## Questions
1. When do classical, heteroskedasticity-robust (HC3) and Newey-West (HAC) standard errors reach the nominal size?
2. What does robustness cost in confidence-interval width and power?
3. How precise is the simulation itself?

## Method
Four data-generating processes: (A) iid normal errors, (B) heteroskedastic errors, sd = 0.25 + x squared, (C) AR(1) errors and
regressor, both with coefficient 0.7, (D) heavy-tailed Student-t errors with 3 degrees of freedom. Sample sizes 25, 50 and 200.
Vectorised numpy implementation of the three standard errors; Student-t critical values. The Monte Carlo standard error of a rejection
rate near 5% is about 0.15 percentage points, so rates between about 4.7% and 5.3% are consistent with 5%.

## Results (actual size of the nominal 5% test, in %)
| DGP | n | Classical | HC3 | Newey-West |
|---|---|---|---|---|
| A iid | 25 / 50 / 200 | 5.1 / 4.9 / 4.9 | 5.5 / 4.9 / 5.0 | 11.2 / 8.2 / 6.1 |
| B heteroskedastic | 25 / 50 / 200 | 33.8 / 34.8 / 36.1 | 7.0 / 6.2 / 5.3 | 16.1 / 11.4 / 6.6 |
| C autocorrelated | 25 / 50 / 200 | 20.7 / 23.4 / 24.8 | 21.1 / 24.0 / 25.2 | 23.2 / 18.8 / 11.9 |
| D heavy tails | 25 / 50 / 200 | 5.0 / 5.3 / 5.0 | 4.1 / 4.4 / 4.6 | 9.6 / 7.9 / 5.6 |

- **Heteroskedasticity makes classical inference badly wrong and does not go away with more data:** about 35% rejection instead of 5%
  at every n. HC3 fixes it (5.3% at n = 200) but is still slightly oversized in small samples (7.0% at n = 25).
- **Robustness has a price in width and power.** In DGP B at n = 200 the classical interval is 0.52 wide but the true sampling standard
  deviation is 0.29, so a correct interval is about 1.1 wide; HC3 gets 1.09. At n = 50 under DGP B, HC3 has only 22% power to
  detect a slope that is 0.5 too high, against 92% for the classical test under iid errors in DGP A (not comparable: different error
  variances). The honest message is that heteroskedastic data contain less information about the slope.
- **Autocorrelation is the hardest case:** classical and HC3 reject about 21-25% of the time. Newey-West improves with n (23% at n = 25,
  19% at 50, 12% at 200) but is still more than twice the nominal size at n = 200 with a rule-of-thumb lag of 4.
  Under iid errors Newey-West is oversized in small samples (11.2% at n = 25), a price paid for protection that is not needed there.
- **Heavy tails alone (D) do not hurt the t-test** here: sizes are within about 1 point of 5% for the classical and HC3 methods.

## Limitations
- One regressor and four stylised processes; the lag rule for Newey-West is fixed, not data-driven; no bootstrap or fixed-b critical
  values (which improve small-sample HAC inference); results for other designs (several regressors, other forms of
  heteroskedasticity or stronger leverage) can differ.
- Power comparisons across DGPs are not like-for-like because the error variances differ.

## Possible extensions
Wild bootstrap and fixed-b HAC, HC1-HC3 comparison with high-leverage designs, cluster-robust standard errors with few clusters,
data-driven bandwidths (Andrews, Newey-West 1994).
