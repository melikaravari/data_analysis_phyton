"""
Project 15 - US presidential vote choice 1996: logistic regression, marginal effects and what a survey can (not) tell us

Data: 1996 American National Election Study (ANES) extract, 944 respondents, distributed with the statsmodels package
(statsmodels.datasets.anes96). Not stored in this repository: loaded from statsmodels.
Variables used: vote (0 = Clinton, 1 = Dole), selfLR (respondent's liberal-conservative self-placement, 1-7), PID (party
identification, 0 = strong Democrat ... 6 = strong Republican), age, educ (1-7), income (1-24 percentile groups),
TVnews (days per week watching TV news), logpopul (log of population of the place of residence).

Questions
  1. How well do demographics, ideology and party identification explain the vote, and what does each add?
  2. How large are the associations in interpretable units (average marginal effects)?
  3. Is the linearity assumption for ideology reasonable? Do the models predict out of sample and are they calibrated?
  4. What explains party identification itself (ordered logit)?

Important: this is an observational survey; all results describe associations, not causal effects.

Run:  python analysis.py      (needs numpy pandas scipy statsmodels scikit-learn matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


df = sm.datasets.anes96.load_pandas().data
df = df[["vote", "selfLR", "PID", "age", "educ", "income", "TVnews", "logpopul"]].dropna()
log(f"{len(df)} respondents; share voting Dole: {df.vote.mean() * 100:.0f}%, Clinton: {(1 - df.vote.mean()) * 100:.0f}%")
log("Vote by party identification (share Dole, %): " + ", ".join(f"PID {int(k)}: {v * 100:.0f}" for k, v in df.groupby("PID").vote.mean().items()))
log("Vote by ideology (share Dole, %): " + ", ".join(f"{int(k)}: {v * 100:.0f}" for k, v in df.groupby("selfLR").vote.mean().items()) + "\n")

# ----------------------------------------------------------------------------
# 1. Three nested logit models (robust HC1 standard errors)
# ----------------------------------------------------------------------------
forms = {
    "M1 demographics": "vote ~ age + educ + income + TVnews + logpopul",
    "M2 + ideology": "vote ~ age + educ + income + TVnews + logpopul + selfLR",
    "M3 + party ID": "vote ~ age + educ + income + TVnews + logpopul + selfLR + PID",
}
fits = {k: smf.logit(f, df).fit(disp=False, cov_type="HC1") for k, f in forms.items()}
rows = []
for k, m in fits.items():
    rows.append({"model": k, "pseudo R2 (McFadden)": round(m.prsquared, 3), "log-likelihood": round(m.llf, 1), "AIC": round(m.aic, 1), "BIC": round(m.bic, 1)})
log("Model comparison")
log(pd.DataFrame(rows).to_string(index=False), "\n")
lr = 2 * (fits["M3 + party ID"].llf - fits["M2 + ideology"].llf)
log(f"Likelihood-ratio test M3 vs M2 (adding party ID): chi2(1) = {lr:.1f}, p = {1 - stats.chi2.cdf(lr, 1):.2g}\n")

m3 = fits["M3 + party ID"]
tab = pd.DataFrame({"coef": m3.params, "robust se": m3.bse, "odds ratio": np.exp(m3.params), "p-value": m3.pvalues}).round(3)
log("Model M3: logit coefficients and odds ratios (outcome = voted Dole)")
log(tab.to_string(), "\n")

me = m3.get_margeff(at="overall").summary_frame().round(3)
log("Average marginal effects in M3 (change in the probability of voting Dole per one-unit increase)")
log(me[["dy/dx", "Std. Err.", "Pr(>|z|)"]].to_string(), "\n")
me2 = fits["M2 + ideology"].get_margeff(at="overall").summary_frame().round(3)
log(f"Without party ID (M2), the marginal effect of ideology is {me2.loc['selfLR', 'dy/dx']:.3f} per point, "
    f"versus {me.loc['selfLR', 'dy/dx']:.3f} once party ID is included: party ID absorbs much of the ideology association.\n")

# ----------------------------------------------------------------------------
# 2. Is ideology linear on the logit scale?
# ----------------------------------------------------------------------------
m_cat = smf.logit("vote ~ age + educ + income + TVnews + logpopul + C(selfLR) + PID", df).fit(disp=False)
m_lin = smf.logit(forms["M3 + party ID"], df).fit(disp=False)
lr_lin = 2 * (m_cat.llf - m_lin.llf)
dfree = int(m_cat.df_model - m_lin.df_model)
log(f"Linearity check: ideology as 7 categories vs one linear term: LR chi2({dfree}) = {lr_lin:.1f}, p = {1 - stats.chi2.cdf(lr_lin, dfree):.3f}\n")

# ----------------------------------------------------------------------------
# 3. Out-of-sample performance (10-fold cross-validation, pooled out-of-fold predictions)
# ----------------------------------------------------------------------------
feats = {"M1 demographics": ["age", "educ", "income", "TVnews", "logpopul"],
         "M2 + ideology": ["age", "educ", "income", "TVnews", "logpopul", "selfLR"],
         "M3 + party ID": ["age", "educ", "income", "TVnews", "logpopul", "selfLR", "PID"]}
cv = StratifiedKFold(10, shuffle=True, random_state=0)
oof, rows = {}, []
for k, cols in feats.items():
    Xs = (df[cols] - df[cols].mean()) / df[cols].std()
    p = cross_val_predict(LogisticRegression(max_iter=2000), Xs, df.vote, cv=cv, method="predict_proba")[:, 1]
    oof[k] = p
    rows.append({"model": k, "cross-validated AUC": round(roc_auc_score(df.vote, p), 3), "accuracy": round(((p > 0.5) == df.vote).mean(), 3)})
base = max(df.vote.mean(), 1 - df.vote.mean())
log(f"Out-of-sample performance (10-fold CV); always predicting the majority class gives accuracy {base:.3f}")
log(pd.DataFrame(rows).to_string(index=False), "\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
g = df.groupby("selfLR").vote.agg(["mean", "size"])
axes[0].plot(g.index, g["mean"], "o-", label="observed share Dole")
grid = pd.DataFrame({"selfLR": np.arange(1, 8)})
for c in ["age", "educ", "income", "TVnews", "logpopul"]:
    grid[c] = df[c].mean()
grid["PID"] = df.PID.mean()
axes[0].plot(grid.selfLR, m_lin.predict(grid), "s--", label="M3 prediction (other covariates at means)")
axes[0].set_xlabel("ideology (1 = extremely liberal, 7 = extremely conservative)")
axes[0].set_ylabel("probability of voting Dole")
axes[0].legend(fontsize=7)
axes[0].set_title("Ideology and vote")
pt, pp = calibration_curve(df.vote, oof["M3 + party ID"], n_bins=8, strategy="quantile")
axes[1].plot(pp, pt, "o-")
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].set_xlabel("predicted probability")
axes[1].set_ylabel("observed share")
axes[1].set_title("Calibration of M3 (out-of-fold)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "ideology_and_calibration.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4. Ordered logit for party identification (0-6)
# ----------------------------------------------------------------------------
X = df[["selfLR", "age", "educ", "income", "TVnews"]]
om = OrderedModel(df.PID.astype(int), X, distr="logit").fit(method="bfgs", disp=False)
ot = pd.DataFrame({"coef": om.params[:5], "se (ML)": om.bse[:5], "odds ratio": np.exp(om.params[:5]), "p-value": om.pvalues[:5]}).round(3)
log("Ordered logit: party identification (0 strong Democrat ... 6 strong Republican) on covariates")
log(ot.to_string())
log("(odds ratio > 1: higher values of the covariate are associated with higher odds of a more Republican category; "
    "ordered logit assumes proportional odds, which I did not test.)\n")

log("Caveats: ANES-style samples need survey weights and design-based standard errors, which are not in this extract, so the")
log("standard errors are too small if the design is not simple random sampling; only 1996, only respondents who voted for one of two")
log("candidates; ideology and party ID are measured at the same time as the vote and probably co-determined with it, so the")
log("coefficients are associations, not causal effects; income is coded as percentile groups; no interactions or non-linearities")
log("beyond the single linearity check; no treatment of missing data (the extract is complete).")
out.close()
print("\nDone. See results.txt and figures/.")
