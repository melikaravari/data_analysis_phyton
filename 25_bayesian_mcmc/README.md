# Project 25: Bayesian modelling with MCMC written from scratch

Two analyses in one script; the samplers are implemented in numpy (no black-box library).

## Part A: Hierarchical model for eight schools
**Data:** the eight-schools data (Rubin 1981; Gelman et al., *Bayesian Data Analysis*): estimated SAT coaching effects y = 28, 8, -3, 7, -1, 1, 18, 12 points with known standard
errors 15, 10, 16, 11, 9, 11, 10, 18. **Model:** y_j ~ N(theta_j, s_j squared), theta_j ~ N(mu, tau squared). **Sampler:** Gibbs sampler for theta and mu, Metropolis step for tau;
4 chains x 25,000 draws after 5,000 warm-up; split R-hat and effective sample size (Geyer) written from scratch.

- **Convergence:** R-hat at most 1.003; effective sample size above 1,000 for all checked parameters (tau: about 1,060).
- **Partial pooling:** the raw estimate of school 1 (28 points) shrinks to a posterior mean of 9.2 (95% interval -2 to 23); mu has a posterior mean of 7.7 (95% interval -1.0 to 16.6);
  tau has a median of 2.8 (interval 0.1 to 11.8). P(school 1 has the largest effect) = 0.20.
- **The prior on tau matters** because eight schools say little about it: the posterior median of tau ranges from 0.6 (inverse-gamma(0.001) on tau squared) to 5.2 (uniform(0, 30)), and the
  posterior mean of theta_1 from 8.3 to 11.2. The inverse-gamma prior, often called "non-informative", gives the strongest shrinkage here.

## Part B: Bayesian regression for log wages (Card's data)
Gibbs sampler for the regression coefficients and error variance (6,000 draws, 1,000 warm-up).
- With weakly informative priors, posterior means and credible intervals **coincide with OLS** (education 0.074, 95% credible interval 0.067-0.081).
- A very strong prior on the education coefficient (0.05 +- 0.005) gives a posterior mean of 0.066, a compromise between the prior and the data (0.074): prior sensitivity in action.
- Posterior predictive checks: the model reproduces the minimum and the spread of log wages but **not the skewness** (observed -0.16, replicated -0.03, p = 0.998): wages
  have a heavier left tail than the normal model implies.

## Limitations
- Eight observations carry almost no information about tau; a Gibbs sampler mixes slowly near tau = 0, and Hamiltonian Monte Carlo with a non-centred parametrisation would be better.
- Part B is a normal linear model with constant variance; priors are subjective; the regression is descriptive (education is endogenous, see project 21).

## Possible extensions
Compare with PyMC/Stan, leave-one-out cross-validation (LOO-CV) for model comparison, a hierarchical model for the sleepstudy data (project 23), robust (Student-t) errors for Part B.
