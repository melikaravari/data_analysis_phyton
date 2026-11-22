# Project 18: Education, origin and vocabulary in the US General Social Survey, 1978-2016

**Data:** `GSSvocab` from the R package `carData`: 28,867 adult respondents in 20 waves of the General Social Survey (NORC), with a 10-item vocabulary
test (WORDSUM: words correct, 0-10; mean 6.0, SD 2.1), years of education, age, gender and US-born status. Vocabulary is missing for
1,348 respondents (4.7%); the analysis sample is 27,360. Download with `download_public_data.py`.

## Questions
1. How strongly is vocabulary related to education, and is it linear?
2. How large are the gaps by gender and by place of birth?
3. Did vocabulary change over 1978-2016, and did the education gradient change?
4. Does the result depend on how the bounded outcome is modelled? 5. Is the missingness related to the covariates?

## Method
OLS with HC3 robust standard errors in five nested models (education; + age, age squared, gender, foreign-born; + survey year; + education x year;
+ education x foreign-born), an F-test of linearity in education, a binomial logit robustness check treating the score as the number of correct answers
out of 10, and a logit model of the missingness.

## Results
- **Education:** one more year is associated with 0.33 more words correct (0.16 SD; R-squared 0.23). Mean scores rise from 4.5 words
  (under 12 years) to 7.7 (more than 16 years). The relationship is not exactly linear (F-test p < 0.001 with 20 education categories), but the
  departure is small (R-squared 0.229 linear vs 0.237 categorical).
- **Foreign-born gap:** about -0.70 words (-0.33 SD) at 12 years of education, and the education gradient is flatter for the foreign-born
  (interaction -0.107 words per year), so the gap grows with education.
- **Gender:** women score 0.15 words higher (0.07 SD), a very small difference.
- **Time:** scores fall slightly, by 0.08 words per decade (0.04 SD), and the education gradient for US-born respondents weakens
  from about 0.42 words per year in 1978 to about 0.31 in 2016 (interaction -0.031 per decade, p < 0.001).
- **Age:** a hump-shaped profile peaking around age 71 (cross-sectional: age, cohort and period are confounded).
- **Robustness:** the binomial logit gives the same signs and significance for every term (e.g. odds ratio 1.17 per year of education, 0.71 for foreign-born).
- **Missingness:** missing vocabulary scores are more common among the foreign-born (odds ratio 1.94), older respondents and less educated ones,
  so complete-case results may be somewhat biased.

## Limitations
- Survey weights, strata and clusters are not in this extract: standard errors assume simple random sampling and are too small.
- The 10-item test has floor and ceiling effects; vocabulary is an imperfect measure of cognitive skill and depends on language background.
- The education coefficient is not the causal effect of schooling; ability, family background and schooling are related. Foreign-born status ignores
  time in the US and first language. Statistical significance is easy to reach with n = 27,000, so I focus on effect sizes.

## Possible extensions
Survey-weighted and design-based analysis with the full GSS files, multilevel models with survey year as a grouping level, age-period-cohort models
with constraints, multiple imputation for the missing scores, comparison of men and women within education groups.
