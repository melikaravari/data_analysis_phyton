# Applied Statistics and Econometrics Portfolio (Python)

Twenty-nine self-directed projects in Python on econometrics, finance, energy, psychology, chemistry, social science,
text and climate data, written alongside my Master's studies in Econometrics (TU Dortmund). Each folder has a runnable script
(`analysis.py`), the saved output (`results.txt`), figures, and a README that states the question, method, results and limitations.
The READMEs report weak, mixed or negative results as such.

| # | Project | Methods |
|---|---------|---------|
| 1 | [Energy price dynamics](01_energy_prices) | ADF tests, Johansen and Engle-Granger cointegration, VAR, Granger causality, impulse responses |
| 2 | [Volatility modelling](02_volatility) | GARCH, GJR-GARCH, Student-t errors, comparison with VIX, out-of-sample QLIKE evaluation |
| 3 | [German electricity prices](03_german_power) | Descriptive analysis, negative prices, day-ahead forecasting (ridge regression, gradient boosting) |
| 4 | [S&P 500 panel](04_sp500_panel) | Data cleaning, PCA factor structure, Ledoit-Wolf covariance, minimum-variance portfolio |
| 5 | [Kalman filter tracking](05_kalman_tracking) | State-space models, Kalman filter and smoother from scratch, maximum likelihood, innovation diagnostics |
| 6 | [VaR and ES backtesting](06_var_backtesting) | Historical, EWMA-normal and GARCH-t VaR, Kupiec and Christoffersen tests, ES check |
| 7 | [VIX dynamics](07_vix_dynamics) | Mean reversion, Markov-switching regimes, asymmetric leverage regression, HAR forecasting |
| 8 | [Jet fuel pass-through](08_jetfuel_passthrough) | Cointegration, error-correction model, asymmetric pass-through, Wald tests |
| 9 | [SARIMA forecasting of power prices](09_sarima_power) | Box-Jenkins identification, SARIMA, Ljung-Box, Diebold-Mariano tests |
| 10 | [Brent-WTI spread](10_brent_wti_breaks) | Zivot-Andrews and CUSUM break tests, Chow scan, VECM before/after the break |
| 11 | [Betas and Fama-MacBeth](11_beta_fama_macbeth) | CAPM betas, portfolio sorts, Fama-MacBeth with Newey-West errors |
| 12 | [Variance risk premium](12_variance_risk_premium) | Predictive regressions with HAC errors, out-of-sample R2, Clark-West test |
| 13 | [Cancer diagnosis, machine learning](13_cancer_diagnosis_ml) | Nested cross-validation, calibration, cost-sensitive thresholds, permutation importance |
| 14 | [Monte Carlo study of inference](14_monte_carlo_inference) | Simulation design, size and power, robust and HAC standard errors, Monte Carlo error |
| 15 | [Survey analysis: vote choice](15_survey_vote_choice) | Logit and ordered logit, marginal effects, likelihood-ratio tests, cross-validation |
| 16 | [Chemistry: kinetics and chemometrics](16_chemistry_kinetics_chemometrics) | Nonlinear regression (Michaelis-Menten), bootstrap CIs, PCA, PLS-DA, outlier diagnostics |
| 17 | [Psychology: Big Five measurement](17_psychology_big_five) | Reliability (alpha), factor analysis, parallel analysis, effect sizes, multiple testing |
| 18 | [Social science: education and vocabulary](18_social_education_vocabulary) | Regression with interactions, robust SEs, binomial GLM robustness, missing-data check |
| 19 | [Corporate leverage: panel data (Compustat)](19_capital_structure_panel) | Fixed effects, clustered standard errors, Mundlak test, robustness; WRDS data not included |
| 20 | [Difference-in-differences](20_did_incinerator) | Causal inference, DiD, randomisation inference, composition checks |
| 21 | [Instrumental variables](21_iv_returns_to_education) | 2SLS, weak-instrument tests, Anderson-Rubin set, over-identification, instrument validity |
| 22 | [Regression discontinuity](22_rd_incumbency) | Local linear RD, density test, covariate balance, placebo cutoffs, sensitivity |
| 23 | [Mixed-effects models](23_mixed_effects_sleep) | Random intercepts and slopes, shrinkage, likelihood-ratio tests, prediction |
| 24 | [Survival analysis](24_survival_recidivism) | Kaplan-Meier, Cox model, proportional-hazards tests, AFT model, randomised experiment |
| 25 | [Bayesian MCMC from scratch](25_bayesian_mcmc) | Gibbs and Metropolis samplers, hierarchical model, R-hat/ESS, prior sensitivity, posterior predictive checks |
| 26 | [Factorial experiment (ANOVA)](26_doe_anova_npk) | Design of experiments, blocking, confounding, power, permutation tests |
| 27 | [Spatial econometrics](27_spatial_cigarettes) | Moran's I, LISA, SAR/SEM/SLX models, effect decomposition, maps |
| 28 | [Text analysis](28_text_inaugural) | Readability trends, weighted log-odds, NMF topics, ridge regression on text |
| 29 | [Climate time series](29_climate_co2_temperature) | STL, trend inference with HAC/AR(1), unit roots, cointegration, spurious regression, forecasting |

## Reproduce
```bash
pip install -r requirements.txt
python download_public_data.py        # once: small public data sets for projects 16-18 and 20-29
python 23_mixed_effects_sleep/analysis.py   # run any project the same way
```
| Projects | Data |
|---|---|
| 13, 14, 15 | loaded from scikit-learn / statsmodels, or simulated: nothing to download |
| 16-18, 20-29 | public data sets, fetched by `download_public_data.py` (some come with Python packages) |
| 1-12 | financial, energy and simulation data files that are **not included**; see [DATA.md](DATA.md) for the expected files and formats |
| 19 | Compustat data from WRDS, **not included** (licensed); see the project README for the variables to download |

Scripts read from a `data/` folder, which is excluded from the repository (`.gitignore`) so that no licensed data is published.
