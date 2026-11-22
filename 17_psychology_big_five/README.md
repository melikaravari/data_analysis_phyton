# Project 17: Measuring personality - reliability and factor structure of a Big Five questionnaire

**Data:** `bfi` from the R package `psych` (Revelle): 2,800 volunteers, 25 items on a 1-6 scale (5 each for Agreeableness A,
Conscientiousness C, Extraversion E, Neuroticism N, Openness O), plus gender (919 men, 1,881 women), education (1-5) and age.
2,436 respondents (87%) answered all items. Download with `download_public_data.py`.

## Questions
1. Are the scales reliable, and do the reverse-keyed items behave as documented?
2. How many factors does the data support, and does the intended five-factor structure appear?
3. How do the scale scores differ by gender and relate to age and education, and how large are the effects?

## Method
Reverse-key detection from the data (sign of each item on the first principal component of its scale) compared with the documented
list; Cronbach's alpha; parallel analysis (Horn, 200 random data sets) and BIC of maximum-likelihood factor models; five-factor
solution with varimax rotation; scale scores; Welch t-tests with Cohen's d and confidence intervals, Holm correction; Spearman
correlations.

## Results
- **Reverse-keyed items are exactly the documented ones** (A1, C4, C5, E1, E2, O2, O5); the data-driven check agrees with the documentation.
- **Reliability** (Cronbach's alpha): A 0.70, C 0.73, E 0.76, N 0.81, O 0.60. Acceptable for research, but Openness is weak (mean inter-item
  correlation 0.24) and dropping item A1 would raise Agreeableness to 0.72.
- **Dimensionality:** parallel analysis supports five factors (eigenvalues 5.13, 2.75, 2.14, 1.85, 1.55, then 1.07, which falls short of
  the random-data threshold). BIC keeps decreasing up to the eight factors I tried, but the gains after five or six are small; with N = 2,436,
  BIC tends to favour more factors.
- **Structure:** in the five-factor solution every one of the 25 items has its largest loading on the factor of its intended scale (25 of 25), and
  the five scales use five different factors. A few items cross-load above 0.30 (E3, E4, E5 and N4); the five factors explain 42% of the variance.
- **Scales are correlated:** for example Agreeableness-Extraversion r = 0.46 and Neuroticism negatively related to all others (-0.08 to -0.23),
  which is why an oblique rotation would be more natural than varimax.
- **Gender differences (women minus men):** Agreeableness d = 0.45 (95% CI 0.37-0.53), Neuroticism 0.27, Extraversion 0.23, Conscientiousness 0.20
  and Openness -0.12; all remain significant after Holm correction (p < 0.003), but apart from Agreeableness they are small effects.
- **Age and education:** Spearman correlations with age are small (Agreeableness 0.20, Conscientiousness 0.15, Neuroticism -0.10);
  education is almost unrelated (all |r| <= 0.05, except Openness 0.11).

## Limitations
- A volunteer online sample, not a representative one. Five respondents report an age under 12 (a data-entry problem), which I did not remove
  because age plays a minor role.
- Items are ordinal but treated as continuous (polychoric correlations would be better); varimax forces uncorrelated factors; no confirmatory
  factor analysis; no test of measurement invariance, so mean gender differences assume the scales measure the same thing in both groups.
- With a sample this large, almost everything is statistically significant, so effect sizes matter more than p-values. Cross-sectional ages do not show development.

## Possible extensions
Oblique rotation (oblimin) and polychoric correlations, confirmatory factor analysis and measurement invariance (e.g. with `semopy`), item
response theory (graded response model), omega reliability, a second data set for replication.
