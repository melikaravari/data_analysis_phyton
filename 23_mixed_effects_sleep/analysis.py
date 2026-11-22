"""
Project 23 - Mixed-effects (multilevel) models for repeated measurements: sleep deprivation and reaction time

Data: `sleepstudy` (Belenky et al. 2003; from the R package lme4): 18 professional drivers restricted to 3 hours of sleep per night for 9 nights.
Their average reaction time (ms) was measured every day, days 0 (baseline) to 9: 180 observations, 10 per person.

Questions
  1. How much slower do reaction times get per day of sleep restriction, and how much does this differ between people?
  2. Why can't we simply run OLS on all 180 observations? What changes with random intercepts and random slopes?
  3. Is the random slope needed (likelihood-ratio test)? How do the models predict the later days of each person from the first days?

Run:  python analysis.py      (needs numpy pandas scipy statsmodels matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


d = pd.read_csv(HERE.parent / "data" / "sleepstudy.csv")
d["Subject"] = d.Subject.astype(str)
log(f"{len(d)} observations, {d.Subject.nunique()} subjects, days {d.Days.min()}-{d.Days.max()}; mean reaction time {d.groupby('Days').Reaction.mean().iloc[0]:.0f} ms at day 0, "
    f"{d.groupby('Days').Reaction.mean().iloc[-1]:.0f} ms at day 9\n")

# ----------------------------------------------------------------------------
# 1. Ignoring the grouping: OLS (naive and cluster-robust) and the two-stage approach
# ----------------------------------------------------------------------------
ols = smf.ols("Reaction ~ Days", d).fit()
olsc = smf.ols("Reaction ~ Days", d).fit(cov_type="cluster", cov_kwds={"groups": d.Subject.astype("category").cat.codes})
per = d.groupby("Subject").apply(lambda g: pd.Series(np.polyfit(g.Days, g.Reaction, 1), index=["slope", "intercept"]))
t_s = stats.ttest_1samp(per.slope, 0)
ci_s = stats.t.interval(0.95, len(per) - 1, loc=per.slope.mean(), scale=stats.sem(per.slope))
log("Estimated slope (ms per day) from approaches that ignore or partly respect the grouping")
log(f"  Pooled OLS, naive SE:           {ols.params['Days']:.2f} (SE {ols.bse['Days']:.2f}, 95% CI {ols.conf_int().loc['Days', 0]:.2f}-{ols.conf_int().loc['Days', 1]:.2f})")
log(f"  Pooled OLS, cluster-robust SE:  {olsc.params['Days']:.2f} (SE {olsc.bse['Days']:.2f}, 95% CI {olsc.conf_int().loc['Days', 0]:.2f}-{olsc.conf_int().loc['Days', 1]:.2f})")
log(f"  Two-stage (mean of the 18 individual slopes): {per.slope.mean():.2f} (SE {stats.sem(per.slope):.2f}, 95% CI {ci_s[0]:.2f}-{ci_s[1]:.2f})")
log(f"  Individual slopes range from {per.slope.min():.1f} to {per.slope.max():.1f} ms/day (SD {per.slope.std():.1f}); "
    f"{int((per.slope < 0).sum())} of 18 people show a negative slope\n")
log("The naive OLS standard error is too small because observations from the same person are not independent (the intercepts differ a lot between people).\n")

# ----------------------------------------------------------------------------
# 2. Mixed models (REML for estimates, ML for likelihood-ratio tests)
# ----------------------------------------------------------------------------
m_int = smf.mixedlm("Reaction ~ Days", d, groups=d.Subject).fit(reml=True)
m_slope = smf.mixedlm("Reaction ~ Days", d, groups=d.Subject, re_formula="~Days").fit(reml=True)
m_unc = smf.mixedlm("Reaction ~ Days", d, groups=d.Subject, re_formula="0 + C(Subject)").fit(reml=True) if False else None


def summarise(m, label):
    fe = m.fe_params
    se = m.bse_fe
    cov = m.cov_re
    r = {"model": label, "intercept": fe["Intercept"], "slope": fe["Days"], "slope SE": se["Days"], "SD intercept": np.sqrt(cov.iloc[0, 0]),
         "SD residual": np.sqrt(m.scale)}
    if cov.shape[0] > 1:
        r["SD slope"] = np.sqrt(cov.iloc[1, 1])
        r["corr(int, slope)"] = cov.iloc[0, 1] / np.sqrt(cov.iloc[0, 0] * cov.iloc[1, 1])
    return r


tab = pd.DataFrame([summarise(m_int, "Random intercept"), summarise(m_slope, "Random intercept + slope")]).set_index("model").round(2)
log("Mixed models (REML)")
log(tab.T.to_string(), "\n")
ci = m_slope.conf_int().loc["Days"]
log(f"Fixed effect of one more day without sleep (random-slope model): {m_slope.fe_params['Days']:.2f} ms/day, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
log(f"-> the interval is about {m_slope.bse_fe['Days'] / ols.bse['Days']:.1f} times as wide as the naive OLS interval; the point estimate is the same as OLS here because the design is balanced.\n")

# Likelihood-ratio tests with ML
ml_ols = smf.ols("Reaction ~ Days", d).fit()
ml_int = smf.mixedlm("Reaction ~ Days", d, groups=d.Subject).fit(reml=False)
ml_slope = smf.mixedlm("Reaction ~ Days", d, groups=d.Subject, re_formula="~Days").fit(reml=False)
lr1 = 2 * (ml_int.llf - ml_ols.llf)
lr2 = 2 * (ml_slope.llf - ml_int.llf)
p1 = 0.5 * (1 - stats.chi2.cdf(lr1, 1))                              # variance on the boundary: 50:50 mixture of chi2(0) and chi2(1)
p2 = 0.5 * (1 - stats.chi2.cdf(lr2, 1)) + 0.5 * (1 - stats.chi2.cdf(lr2, 2))  # slope variance + covariance: mixture of chi2(1) and chi2(2)
log("Likelihood-ratio tests (ML; p-values use the boundary-corrected chi-square mixtures)")
log(f"  random intercept vs no random effect:        LR = {lr1:.1f}, p {"< 0.001" if p1 < 0.001 else "= " + format(p1, ".3f")}")
log(f"  random intercept + slope vs random intercept: LR = {lr2:.1f}, p {"< 0.001" if p2 < 0.001 else "= " + format(p2, ".3f")}")
log(f"  AIC: OLS {ml_ols.aic:.0f}, random intercept {ml_int.aic:.0f}, random intercept + slope {ml_slope.aic:.0f}\n")

# ----------------------------------------------------------------------------
# 3. Shrinkage: individual OLS slopes vs mixed-model predictions (BLUPs)
# ----------------------------------------------------------------------------
re = pd.DataFrame(m_slope.random_effects).T
re.columns = ["int_dev", "slope_dev"]
blup = pd.DataFrame({"no pooling": per.slope, "mixed model": m_slope.fe_params["Days"] + re.slope_dev})
shrink = (blup["mixed model"] - m_slope.fe_params["Days"]).abs().sum() / (blup["no pooling"] - m_slope.fe_params["Days"]).abs().sum()
log(f"Shrinkage: the mixed model pulls individual slopes toward the average; the sum of absolute deviations from the mean slope is {shrink * 100:.0f}% "
    f"of the no-pooling value (SD of slopes {blup['no pooling'].std():.1f} -> {blup['mixed model'].std():.1f} ms/day)\n")

# ----------------------------------------------------------------------------
# 4. Predicting the later days from the first five (days 0-4): who predicts better?
# ----------------------------------------------------------------------------
tr, te = d[d.Days <= 4], d[d.Days >= 5]
mm = smf.mixedlm("Reaction ~ Days", tr, groups=tr.Subject, re_formula="~Days").fit(reml=True)
pred_mixed = te.apply(lambda r: mm.fe_params["Intercept"] + mm.random_effects[r.Subject].iloc[0] + (mm.fe_params["Days"] + mm.random_effects[r.Subject].iloc[1]) * r.Days, axis=1)
pool = smf.ols("Reaction ~ Days", tr).fit()
pred_pool = pool.predict(te)
pp = tr.groupby("Subject").apply(lambda g: pd.Series(np.polyfit(g.Days, g.Reaction, 1), index=["s", "i"]))
pred_np = te.apply(lambda r: pp.loc[r.Subject, "i"] + pp.loc[r.Subject, "s"] * r.Days, axis=1)
rm = lambda p: float(np.sqrt(((te.Reaction - p) ** 2).mean()))
log("Out-of-sample prediction of days 5-9 from the data of days 0-4 (RMSE in ms)")
log(f"  Pooled OLS (everyone gets the same line):     {rm(pred_pool):.1f}")
log(f"  No pooling (separate line for each person):   {rm(pred_np):.1f}")
log(f"  Mixed model (random intercept + slope):       {rm(pred_mixed):.1f}")
log("With only 5 observations per person, the individual lines are noisy, and the mixed model compromises between the group and the individual.\n")

# ----------------------------------------------------------------------------
# 5. Diagnostics and figure
# ----------------------------------------------------------------------------
resid = m_slope.resid
log(f"Residual diagnostics (random-slope model): residual SD {resid.std():.1f} ms; Shapiro-Wilk p {'< 0.001' if stats.shapiro(resid)[1] < 0.001 else '= ' + format(stats.shapiro(resid)[1], '.3f')}; "
    f"skewness {stats.skew(resid):.2f}; the spread of the data grows over the days (SD of reaction times {d.groupby('Days').Reaction.std().iloc[0]:.0f} -> {d.groupby('Days').Reaction.std().iloc[-1]:.0f} ms), "
    "which the constant residual variance does not capture; this is mostly due to the random slopes.\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for s, g in d.groupby("Subject"):
    axes[0].plot(g.Days, g.Reaction, color="grey", lw=0.7, alpha=0.7)
xs = np.arange(0, 10)
axes[0].plot(xs, m_slope.fe_params["Intercept"] + m_slope.fe_params["Days"] * xs, "r-", lw=2.5, label="average (fixed effect)")
axes[0].set_xlabel("days of sleep restriction")
axes[0].set_ylabel("reaction time (ms)")
axes[0].set_title("18 subjects")
axes[0].legend()
for s in blup.index:
    axes[1].plot([0, 1], [blup.loc[s, "no pooling"], blup.loc[s, "mixed model"]], color="C0", lw=0.8)
axes[1].axhline(m_slope.fe_params["Days"], color="r", ls="--", lw=1)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["individual OLS slope", "mixed-model prediction"])
axes[1].set_ylabel("slope (ms per day)")
axes[1].set_title("Shrinkage toward the average slope")
fig.tight_layout()
fig.savefig(HERE / "figures" / "sleepstudy.png", dpi=150)
plt.close(fig)

log("Caveats: 18 people, all professional drivers, so the variance parameters are imprecise and the findings may not apply to others; a linear trend over")
log("days is assumed (the early days may include training effects and the late days may saturate); residual variance is assumed constant and errors")
log("independent within a person (no autocorrelation structure was fitted); the random effects are assumed normal; statsmodels' MixedLM does not provide")
log("Kenward-Roger or Satterthwaite degrees of freedom, so Wald intervals are slightly too narrow in such a small sample.")
out.close()
print("\nDone. See results.txt and figures/.")
