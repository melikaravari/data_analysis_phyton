# Project 16: Two typical chemistry data analyses - enzyme kinetics and chemometrics

Two small analyses of the kind found in chemistry and biochemistry labs, in one script (Part A and Part B).

## Part A: Michaelis-Menten enzyme kinetics
**Data:** R's `Puromycin` data (Treloar 1974): reaction velocity (counts/min squared) at six substrate concentrations (0.02-1.1 ppm) in
cells treated with puromycin and untreated cells; 23 observations (12 treated, 11 untreated). Download with `download_public_data.py`.

**Questions:** estimate Vmax and Km with honest uncertainty; does puromycin change Vmax, Km or both; is the Lineweaver-Burk
linearisation as good as nonlinear least squares?

**Method:** nonlinear least squares (scipy `curve_fit`) per group with asymptotic and residual-bootstrap (2,000 draws) confidence
intervals; four nested joint models (no effect, Vmax differs, Km differs, both differ) compared by F-tests and AIC;
comparison with double-reciprocal regression.

**Results**
| | Vmax (counts/min squared) | Km (ppm) |
|---|---|---|
| Treated | 212.7, 95% CI (bootstrap) 202-227 | 0.064, CI 0.050-0.079 |
| Untreated | 160.3, CI 150-173 | 0.048, CI 0.034-0.061 |

- A model with a different Vmax is strongly preferred over no effect (F = 44.9, p < 0.0001). Allowing Km to differ as well adds nothing
  (F = 1.72, p = 0.21; AIC 111.3 for both models), but dropping the Vmax difference is clearly worse (F = 25.5, p = 0.0001).
  So the data support puromycin changing Vmax while there is no evidence of a change in Km.
- The Lineweaver-Burk estimates are biased relative to direct nonlinear least squares: Vmax about 8-10% lower, Km 25-35% lower.
  The transformation magnifies errors at low concentrations, so nonlinear least squares is preferred.

## Part B: Chemometrics of wine
**Data:** the classic wine data (Forina et al.): 178 wines from three cultivars (anonymised as class_0-2) in one Italian region, 13 chemical
variables; loaded from scikit-learn.

**Method:** autoscaling; PCA; Hotelling's T-squared (95% F-limit) and Q residual (empirical 95th percentile) outlier diagnostics
from a 3-component model; k-means and Ward clustering (adjusted Rand index); LDA, PLS-DA (implemented as a scikit-learn estimator),
SVM and kNN with 5-fold cross-validation repeated 10 times and scaling inside the pipeline; PLS-DA VIP scores.

**Results**
- The first two principal components explain 55% of the variance (three: 67%, five: 80%). PC1 is dominated by flavanoids, total phenols and
  OD280/OD315; PC2 by colour intensity, alcohol and proline. Unscaled variances differ by a factor of about 6 million, so autoscaling is essential.
- Unsupervised clustering recovers the cultivars well: adjusted Rand index 0.90 (k-means) and 0.79 (Ward).
- Cross-validated classification accuracy: LDA 98.7%, SVM 98.1%, PLS-DA 97.9%, kNN 96.2%; kNN without scaling only 70.6%.
- Outliers: 5 wines exceed the T-squared limit and 9 the Q limit (14 either; about 5% each by construction). The most extreme by
  T-squared are rows 121 and 59 (both cultivar class_1).
- PLS-DA VIP scores above 1 for 6 of 13 variables: alcohol (1.44), colour intensity (1.38), proline (1.37), OD280/OD315, flavanoids, hue.

## Limitations
- Part A has only 23 observations and no replicates; a variance that grows with the rate is plausible and was not modelled; bootstrap and
  asymptotic intervals can be inaccurate at this size.
- Part B: all wines come from one region, so accuracy will not transfer to other settings; the Q limit is empirical and the T-squared
  limit approximate; PLS components were not tuned; VIP is a heuristic.

## Possible extensions
Weighted nonlinear regression, global fits with shared parameters, profile-likelihood intervals, inhibition models (competitive vs
non-competitive), spectral data with preprocessing (SNV, derivatives) and PLS regression for concentration prediction.
