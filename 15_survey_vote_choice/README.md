# Project 15: US vote choice 1996 - logistic regression, marginal effects and the limits of a survey

**Data:** 1996 American National Election Study extract, 944 respondents who voted for Clinton (58%) or Dole (42%), distributed with the
statsmodels package (`statsmodels.datasets.anes96`; not stored in this repository). Variables: vote, ideology (self-placement 1-7),
party identification (0 = strong Democrat ... 6 = strong Republican), age, education (1-7), income (1-24 percentile groups), days of TV
news per week, log of population of the place of residence.

## Questions
1. How well do demographics, ideology and party identification explain the vote, and what does each add?
2. How large are the associations in interpretable units?
3. Is linearity in ideology reasonable? Do the models predict out of sample?
4. What explains party identification itself?

All results are associations from an observational survey, not causal effects.

## Method
Three nested logit models (demographics, + ideology, + party ID) with robust (HC1) standard errors; likelihood-ratio tests;
average marginal effects; a linearity check (ideology as 7 categories vs linear); 10-fold cross-validated AUC and accuracy;
calibration plot; ordered logit for party identification.

## Results
- **Demographics explain very little; ideology and party ID explain a lot.** McFadden pseudo R-squared: 0.052 (demographics),
  0.346 (+ ideology), 0.617 (+ party ID). Cross-validated AUC: 0.634, 0.867, 0.952; accuracy 0.628, 0.797, 0.905 versus 0.584 from always
  predicting the majority class.
- **Party ID absorbs much of the ideology association:** the average marginal effect of one ideology point on the probability of
  voting Dole falls from 0.176 (without party ID) to 0.046 (with it). One step on the party ID scale is associated with 0.080 higher
  probability (odds ratio 2.85). Both are very significant.
- **Demographics matter little once politics is included:** age, education and TV news are not significant. Income (marginal effect 0.003 per
  group, p = 0.047) and population size of the residence (-0.007 per log unit, p = 0.016) are borderline-to-modest; with several
  coefficients tested I would not make much of them.
- **Linearity in ideology is acceptable:** LR test of 7 categories vs a linear term, chi-square(5) = 7.2, p = 0.21.
- **Party identification:** in an ordered logit it is strongly related to ideology (odds ratio 2.79 per point) and, less strongly, to
  education (1.20) and income (1.05); age and TV news are not significant. Proportional odds was not tested.

## Limitations
- The extract has no survey weights or design information, so standard errors assume simple random sampling and are probably too small.
- Ideology, party ID and vote are measured at the same time and are probably determined jointly; the coefficients are not causal.
  Party ID predicts the vote so well partly because it is close to a restatement of it.
- Only one election, only two-candidate voters, income as coarse groups, no interactions, no test of the proportional-odds assumption.

## Possible extensions
Survey-weighted analysis with design-based standard errors (e.g. on the full ANES files), multinomial models including a third
candidate or non-voting, interactions (education x ideology), multiple elections, multilevel models by state.
