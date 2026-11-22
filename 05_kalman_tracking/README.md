# Project 5: Kalman filter and smoother for tracking a moving object

**Data:** `x_sim.csv` (6 columns, 69 time steps) and `y_sim.csv` (2 columns, 69 time steps), simulated, no headers.
**My reading of the files** (inferred from the numbers, not documented): `x_sim` is the true state
`[vx, x, vy, y, ax, ay]` with constant accelerations and a time step of dt = 0.1; `y_sim` is a noisy measurement of the
position `[x, y]`. Checks in the script support this: position increments equal velocity x 0.1, and both acceleration
columns are constant. Source of the data: TODO (author).

## Questions
1. How much better is a Kalman filter than the raw noisy measurements?
2. Can the noise parameters be estimated from the measurements alone?
3. What does smoothing add, and what is lost with a wrong model?

## Method
Linear Gaussian state-space model, written from scratch in numpy: a constant-acceleration model (acceleration follows
a random walk) and a deliberately misspecified constant-velocity model. Filter, Rauch-Tung-Striebel smoother, and
maximum likelihood estimation of the process-noise intensity and measurement variance from the innovations, which
uses only the measurements. The true path is used only for evaluation.

## Results
Position / velocity RMSE against the true path (first 10 steps excluded in the "after warm-up" columns, because the filter
starts without knowing the velocity):

| Estimator | Position RMSE, after warm-up | Velocity RMSE, after warm-up |
|---|---|---|
| Raw measurements | 0.94 | not observed |
| Kalman filter, constant acceleration | 0.47 | 1.01 |
| Kalman smoother, constant acceleration | 0.14 | 0.21 |
| Kalman filter, constant velocity (wrong) | 0.64 | 2.23 |
| Kalman smoother, constant velocity (wrong) | 0.32 | 0.84 |

- The filter halves the position error and recovers velocities that are never measured. Using future observations
  (smoother) reduces the position error by another factor of three.
- The estimated measurement noise std is 0.92; the actual error std in the data is 0.82 (x) and 1.00 (y).
- The maximum likelihood process noise is essentially zero: the simulated acceleration is exactly constant, so the
  likelihood prefers a model with no randomness in the acceleration.
- The misspecified model fits worse (log-likelihood -226.9 vs -217.2) and gives clearly larger errors.
- Innovation diagnostics for the constant-acceleration model look reasonable: mean normalised innovation squared
  is 2.06 (target 2), standardised innovation std 0.90 and 1.12; lag-1 autocorrelation of the x innovations is 0.17,
  slightly high but with only 64 observations I would not read much into it.

## Limitations
- Simulated data with Gaussian noise and exactly known dynamics; real tracking data is messier.
- Only 69 time steps, so the estimates of q and r are imprecise and I did not compute standard errors.
- The filter is initialised at the first measurement with large uncertainty; a different initialisation changes the early steps.

## Possible extensions
Extended Kalman filter for non-linear dynamics, missing measurements, standard errors for the ML estimates, a Kalman
filter for a time-varying regression coefficient (e.g. a time-varying beta).
