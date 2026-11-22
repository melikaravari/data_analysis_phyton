# Project 21: Instrumental variables - the return to education (Card 1995)

**Data:** `card` (David Card's analysis of the National Longitudinal Survey of Young Men; in Wooldridge's textbook data): 3,010 men, wages in 1976 (log hourly wage),
years of education, experience, region, race, parental education, test scores, and indicators for growing up near a 2-year or 4-year college. Download with
`download_public_data.py`.

## Question
What is the causal effect of one more year of schooling on wages? OLS is biased if ability or family background affect both schooling and wages, or if schooling is
measured with error. Instrument: growing up near a 4-year college, which lowers the cost of schooling but should not affect wages directly.

## Method
OLS; first stage and the instrument's strength; reduced form; 2SLS with different instrument sets (`linearmodels`); Wu-Hausman endogeneity test; Sargan
over-identification tests; Anderson-Rubin confidence set (robust to weak instruments); balance checks of the instrument against predetermined characteristics.

## Results
- **OLS: 7.5% higher wages per year of schooling** (95% CI 6.8-8.2%).
- **First stage:** growing up near a 4-year college goes with 0.32 more years of schooling (SE 0.09); F = 14.1, above the rule of thumb of 10 but below the Stock-Yogo
  critical value of 16.4 for 10% size distortion. Reduced form: +4.2% wages (p = 0.016).
- **2SLS with college proximity: 13.2% per year** (SE 5.4; 95% CI 2.6-23.7%); with both college-proximity indicators 15.7% (SE 5.2); with parental education as
  instruments 10.2% (SE 1.3, n = 2,220). The Anderson-Rubin set is [0.03, 0.28], similar to the 2SLS interval.
- IV is larger than OLS but much less precise. Possible reasons: measurement error in schooling, a local effect for men whose schooling responds to college
  proximity (LATE), or an invalid instrument; the data cannot separate these. The Wu-Hausman test does not reject OLS (p = 0.28) because IV is imprecise.
- **Over-identification tests do not reject** (p = 0.26 and 0.35), but they only help if at least one instrument is valid.
- **Threat to validity:** men who grew up near a 4-year college had fathers with 1.1 more years of schooling and +2.6 IQ points (p = 0.001) before conditioning; after
  controlling for region and urban location the IQ difference is 0.67 points (p = 0.32). Parental education is the least credible instrument (family background affects
  wages directly), though its estimate is similar.

## Limitations
- The exclusion restriction cannot be tested directly; 1976 data on young men only; IV estimates a local effect; confidence intervals are wide.
- The instrument is only moderately strong (F = 14).

## Possible extensions
Heterogeneous returns (marginal treatment effects), limited-information ML or other weak-instrument-robust estimators, a sensitivity analysis for instrument invalidity
(Conley-Hansen-Rossi), the same design with more recent data.
