# Project 24: Survival analysis - does financial aid after prison delay re-arrest? (Rossi experiment)

**Data:** `rossi` from the lifelines package (Rossi, Berk and Lenihan 1980): 432 men released from Maryland state prisons and followed for 52 weeks; half were randomly
assigned financial aid (USD 60 per week for 52 weeks). 114 men (26%) were re-arrested; 74% are censored at 52 weeks. Covariates: age, race, prior work experience, married,
on parole, number of prior convictions. Because the aid was randomly assigned, the group comparison estimates the causal effect of the offer of aid (intention to treat).

## Questions
1. How does the aid change the time to re-arrest?
2. What is the hazard ratio, and does the proportional-hazards assumption hold?
3. Does the effect differ by subgroup, and how well does the model predict?

## Method
Kaplan-Meier curves and log-rank test; restricted mean survival time with a bootstrap interval; Cox proportional-hazards models with and without covariates; Schoenfeld-residual tests
of proportional hazards; interaction models; a Weibull accelerated-failure-time model; 5-fold cross-validated concordance (lifelines).

## Results
- **Randomisation worked:** covariates are balanced between groups.
- **Aid delays re-arrest, but the evidence is borderline:** 77.8% of the aid group and 69.4% of the control group were not re-arrested at 52 weeks (+8.3 points); log-rank p = 0.050;
  restricted mean time without re-arrest +2.0 weeks (bootstrap CI -0.3 to +4.6).
- **Cox hazard ratio 0.69** (95% CI 0.48-1.00, p = 0.052) without covariates and 0.68 (0.47-0.996, p = 0.047) with covariates: similar, as expected under randomisation.
  Each prior conviction raises the hazard by 10% (p = 0.001); each year of age lowers it by 6% (p = 0.009).
- **Weibull model:** the aid multiplies the time to re-arrest by 1.31 (95% CI 1.00-1.72).
- **Proportional hazards:** no sign of violation for the treatment (p = 0.90), but evidence against it for age (p = 0.001) and work experience (p = 0.007), so those hazard ratios
  are averages over time.
- **Subgroups:** interaction tests give p = 0.049 for age and 0.068 for parole; with four exploratory tests I would not report these as findings.
- **Prediction is weak:** cross-validated concordance is 0.60 (in sample 0.64).

## Limitations
- Intention-to-treat effect; one year of follow-up with 74% censoring; re-arrest measures detected crime; the original weekly employment data (a mechanism) are not in this version.
- Borderline significance: the data are consistent with a moderate benefit but also with none.

## Possible extensions
Stratified Cox model or time-varying coefficients for age and work experience, competing risks, multiple imputation, mediation analysis with the employment data, a Bayesian survival model.
