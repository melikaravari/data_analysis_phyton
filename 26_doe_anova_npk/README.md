# Project 26: Design and analysis of a factorial experiment in blocks (ANOVA)

**Data:** `npk` (Yates 1937; R's datasets package): yield of green peas (pounds per plot) from a 2^3 factorial experiment with nitrogen (N), phosphate (P) and potassium (K), 24 plots in
6 blocks of 4. Each block contains half of the 8 combinations: the three-factor interaction N:P:K is **confounded with blocks** by design. Download with `download_public_data.py`.
The same methods (factorial designs, blocking, ANOVA, effect plots) are used in chemistry, engineering and process optimisation.

## Questions
1. How large are the main effects and two-factor interactions, and which are distinguishable from noise?
2. How much did blocking help; are the model assumptions reasonable?
3. What effect size could the experiment detect, and do permutation tests agree?

## Method
Verification of the design (balance, confounding); ANOVA with blocks, main effects and two-factor interactions; effect estimates with confidence intervals; blocking efficiency; a mixed model with
random blocks; residual diagnostics; power analysis with the noncentral t distribution; permutation tests within blocks; interaction and normal plots of effects.

## Results
- **Nitrogen raises the yield by 5.6 lb per plot** (95% CI 2.1-9.1, p = 0.004); **potassium lowers it by 4.0 lb** (CI -7.5 to -0.5, p = 0.029); phosphate (-1.2, p = 0.48) and all two-factor
  interactions are indistinguishable from noise. These agree with the standard published analysis of this data set.
- **Blocking helped a lot:** block effect F = 4.45 (p = 0.016); without blocks the error variance would be twice as large (31.1 instead of 15.4).
- **Assumptions look fine:** Shapiro-Wilk p = 0.64, Levene p = 0.94; the mixed model with random blocks gives the same effects.
- **Limited power:** with a residual SD of 3.9 lb and 12 plots per level, 80% power needs an effect of about 4.9 lb. A non-significant effect (P, the interactions) is not evidence of no effect.
- **The potassium result is less secure than the ANOVA suggests:** the within-block permutation test gives p = 0.068 for K (0.008 for N, 0.62 for P).

## Limitations
- Only 24 plots; the three-factor interaction cannot be estimated; six effects are tested without correction for multiple testing; 1930s agricultural data.

## Possible extensions
Fractional factorial and response surface designs, design-of-experiments software for new designs (for example chemical process optimisation), analysis of a real laboratory data set with
replicates, Box-Cox transformation.
