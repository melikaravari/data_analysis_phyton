"""
Project 13 - Breast cancer diagnosis: supervised learning with honest validation

Data: Wisconsin Diagnostic Breast Cancer data (569 patients, 30 numeric features computed from images of fine-needle
aspirates; 212 malignant, 357 benign). Wolberg, Street & Mangasarian (1995), distributed with scikit-learn. Not stored in
this repository: it is loaded from scikit-learn.

Questions
  1. Which model class (regularised logistic regression, SVM, random forest, gradient boosting) classifies best, when
     hyper-parameters are tuned inside the validation (nested cross-validation, so the performance estimate is not optimistic)?
  2. Are the predicted probabilities calibrated?
  3. If a missed cancer is much worse than a false alarm, which decision threshold is best, and what does it cost?
  4. Which features matter, and why must importance be read with care when features are correlated?

Run:  python analysis.py      (needs numpy pandas scikit-learn matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")
SEED = 0


def log(*a):
    print(*a)
    print(*a, file=out)


data = load_breast_cancer(as_frame=True)
X = data.data
y = (data.target == 0).astype(int)  # 1 = malignant (the class we must not miss); scikit-learn codes malignant as 0
log(f"{X.shape[0]} patients, {X.shape[1]} features; malignant {y.sum()} ({y.mean() * 100:.0f}%), benign {(1 - y).sum()}")
corr = X.corr().abs().values[np.triu_indices(X.shape[1], 1)]
log(f"Features are strongly correlated: {(corr > 0.9).sum()} of {len(corr)} feature pairs have |correlation| > 0.9 "
    "(e.g. radius, perimeter and area measure almost the same thing)\n")

# ----------------------------------------------------------------------------
# Models and hyper-parameter grids (scaling is inside the pipeline: no information from the test fold leaks in)
# ----------------------------------------------------------------------------
models = {
    "Logistic regression (L2)": (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "SVM (RBF kernel)": (
        make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]}),
    "Random forest": (
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        {"max_depth": [None, 5], "min_samples_leaf": [1, 3]}),
    "Gradient boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"learning_rate": [0.03, 0.1], "max_depth": [2, 4]}),
}

# ----------------------------------------------------------------------------
# Nested cross-validation: outer 5 folds x 2 repeats estimate performance; inner 3 folds tune the hyper-parameters
# ----------------------------------------------------------------------------
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
oof = {m: np.zeros((2, len(y))) for m in models}  # out-of-fold probability, one row per repeat
rows = []
for name, (est, grid) in models.items():
    aucs, sens, spec, briers = [], [], [], []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        gs = GridSearchCV(est, grid, cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="roc_auc", n_jobs=-1)
        gs.fit(X.iloc[tr], y.iloc[tr])
        p = gs.predict_proba(X.iloc[te])[:, 1]
        oof[name][k // 5, te] = p
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y.iloc[te], pred).ravel()
        aucs.append(roc_auc_score(y.iloc[te], p))
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
        briers.append(brier_score_loss(y.iloc[te], p))
    rows.append({"model": name, "ROC AUC": np.mean(aucs), "AUC std (folds)": np.std(aucs), "sensitivity": np.mean(sens),
                 "specificity": np.mean(spec), "Brier score": np.mean(briers)})
res = pd.DataFrame(rows).set_index("model").round(3)
log("Nested cross-validation (5 folds x 2 repeats; threshold 0.5; sensitivity = share of malignant cases found)")
log(res.to_string(), "\n")
log("Differences between the best models are small relative to the fold-to-fold spread, so with 569 patients the ranking among the")
log("top models should not be over-interpreted; the folds are not independent, so the standard deviations are descriptive only.\n")

# ----------------------------------------------------------------------------
# Calibration and ROC curves from the out-of-fold predictions (repeat 1)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name in models:
    fpr, tpr, _ = roc_curve(y, oof[name][0])
    axes[0].plot(fpr, tpr, label=f"{name}")
    pt, pp = calibration_curve(y, oof[name][0], n_bins=8, strategy="quantile")
    axes[1].plot(pp, pt, "o-", ms=3, label=name)
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[0].set_xlabel("false positive rate")
axes[0].set_ylabel("true positive rate")
axes[0].set_title("ROC curves (out-of-fold)")
axes[0].legend(fontsize=7)
axes[1].set_xlabel("predicted probability")
axes[1].set_ylabel("observed share malignant")
axes[1].set_title("Calibration (out-of-fold)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "roc_and_calibration.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# Cost-sensitive decision: a missed cancer (false negative) is 10 times as costly as a false alarm (false positive).
# The cost ratio is an assumption for illustration, not a clinical estimate.
# The threshold is chosen on repeat 1 out-of-fold predictions and evaluated on repeat 2 (different splits, but the SAME patients,
# so this evaluation is only partly independent and slightly optimistic; the threshold is a single parameter, so the bias is small).
# ----------------------------------------------------------------------------
COST_FN, COST_FP = 10, 1
best_name = res["ROC AUC"].idxmax()
grid_t = np.linspace(0.02, 0.98, 49)


def total_cost(p, t):
    pred = p >= t
    return COST_FN * ((~pred) & (y == 1)).sum() + COST_FP * (pred & (y == 0)).sum()


rows = []
for name in models:
    t_star = grid_t[np.argmin([total_cost(oof[name][0], t) for t in grid_t])]
    for label, t in [("threshold 0.5", 0.5), (f"cost-optimal threshold {t_star:.2f}", t_star)]:
        p = oof[name][1]
        pred = p >= t
        fn = int(((~pred) & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        rows.append({"model": name, "rule": label, "missed cancers": fn, "false alarms": fp, "total cost": COST_FN * fn + COST_FP * fp})
log(f"Cost-sensitive thresholding (a missed cancer costs {COST_FN}x a false alarm; threshold tuned on repeat 1, evaluated on repeat 2 with the same patients, so slightly optimistic)")
log(pd.DataFrame(rows).to_string(index=False), "\n")

# ----------------------------------------------------------------------------
# Feature importance of the best model: permutation importance on held-out data
# ----------------------------------------------------------------------------
est, grid = models[best_name]
tr_idx, te_idx = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
gs = GridSearchCV(est, grid, cv=3, scoring="roc_auc", n_jobs=-1).fit(X.iloc[tr_idx], y.iloc[tr_idx])
pi = permutation_importance(gs.best_estimator_, X.iloc[te_idx], y.iloc[te_idx], scoring="roc_auc", n_repeats=50, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
log(f"Permutation importance (drop in ROC AUC when a feature is shuffled), {best_name}, one held-out fold of {len(te_idx)} patients")
log(imp.head(8).round(4).to_string())
log("Note: because of the strong correlations, shuffling one feature is partly compensated by its correlated partners, so individual")
log("importances are small and unstable even when the feature group matters a lot. Importance is not causal.\n")

fig, ax = plt.subplots(figsize=(6, 4))
imp.head(10)[::-1].plot.barh(ax=ax)
ax.set_title("Permutation importance (held-out fold)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "permutation_importance.png", dpi=150)
plt.close(fig)

log("Caveats: one small, clean, curated data set from a single source; no external validation on patients from other hospitals;")
log("the cost ratio is assumed; hyper-parameter grids are small; in practice calibration, fairness across patient groups and clinical")
log("workflow would also need checking before any real use. This project is a methodological exercise, not a diagnostic tool.")
out.close()
print("\nDone. See results.txt and figures/.")
