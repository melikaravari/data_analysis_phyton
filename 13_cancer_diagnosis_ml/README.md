# Project 13: Breast cancer diagnosis - supervised learning with honest validation

**Data:** Wisconsin Diagnostic Breast Cancer data: 569 patients, 30 numeric image features, 212 malignant and 357 benign
(Wolberg, Street and Mangasarian, 1995; loaded from scikit-learn, not stored in this repository).

## Questions
1. Which model class performs best when hyper-parameters are tuned inside the validation?
2. Are the predicted probabilities calibrated?
3. If a missed cancer is much worse than a false alarm, what threshold is best?
4. Which features matter, and why must importance be read with care?

## Method
Four model classes (L2 logistic regression, RBF-kernel SVM, random forest, gradient boosting) in pipelines with scaling inside the
cross-validation (no leakage). **Nested cross-validation:** 5 outer folds x 2 repeats estimate performance, 3 inner folds tune
hyper-parameters. Out-of-fold predictions for ROC and calibration curves and Brier scores. A cost-sensitive threshold (missed cancer
costs 10 times a false alarm; this ratio is an assumption). Permutation importance on held-out data.

## Results
- **All four models are excellent and nearly equal:** ROC AUC 0.992-0.995 (SVM 0.995, logistic 0.994, gradient boosting 0.994,
  random forest 0.992) with fold-to-fold standard deviation about 0.005. With 569 patients I would not rank the top models.
  Regularised logistic regression is as good as the complex models; the SVM has the best Brier score (0.017).
- **At the default threshold 0.5**, sensitivity (share of malignant cases found) is 93.6-96.2% and specificity 97.5-98.9%.
- **Cost-sensitive threshold:** lowering the threshold to 0.20-0.28 cuts missed cancers sharply. For the SVM: 9 missed and 5 false
  alarms (cost 95) at 0.5 versus 4 missed and 11 false alarms (cost 51) at 0.28. For logistic regression: 12 missed and 5 false alarms
  (cost 125) versus 5 missed and 20 false alarms (cost 70). The threshold is chosen on one set of out-of-fold predictions and evaluated
  on another repetition of the same patients, so the improvement is slightly optimistic.
- **Features:** 21 of 435 feature pairs have |correlation| above 0.9 (radius, perimeter, area). Permutation importance is therefore small
  and spread out (largest: worst texture, 0.006 AUC drop); individual importances are unreliable under correlation.

## Limitations
- One small, clean, curated data set from one source: no external validation, so the performance will be lower on patients from other
  settings. This is a methodological exercise, not a diagnostic tool.
- The cost ratio is assumed; small hyper-parameter grids; no fairness or subgroup analysis; no uncertainty intervals beyond
  the fold standard deviations (folds are not independent).

## Possible extensions
Calibration methods (Platt, isotonic) and their effect on the Brier score, SHAP values, learning curves, a test on a second
public data set, confidence intervals via repeated cross-validation corrected for dependence.
