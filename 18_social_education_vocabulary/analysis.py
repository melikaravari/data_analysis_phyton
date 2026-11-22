"""
Project 18 - Education, origin and vocabulary in the US General Social Survey 1978-2016

Data: `GSSvocab` from the R package `carData`: 28,867 adult respondents in 20 waves of the General Social Survey (NORC),
with a 10-item vocabulary test (WORDSUM, number of words correct, 0-10), education (years), age, gender, and whether the
respondent was born in the US. Source file: data/gss_vocab.csv (downloaded from the Rdatasets collection).

Questions
  1. How strongly is vocabulary related to education, and is the relationship linear?
  2. How large are the gaps by gender and by place of birth, and how do they depend on education?
  3. Did vocabulary scores change between 1978 and 2016, and did the education gradient change?
  4. Does the result depend on how the bounded 0-10 outcome is modelled (OLS vs binomial regression)?
  5. Is missing vocabulary data related to the covariates?

Everything here is descriptive/associational: the data are observational and cross-sectional.

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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


g = pd.read_csv(HERE.parent / "data" / "gss_vocab.csv")
log(f"{len(g)} respondents, {g.year.nunique()} survey years ({g.year.min()}-{g.year.max()}); vocabulary score missing for {g.vocab.isna().sum()} ({g.vocab.isna().mean() * 100:.1f}%)")
log(f"Vocabulary (words correct out of 10): mean {g.vocab.mean():.2f}, SD {g.vocab.std():.2f}\n")

d = g.dropna(subset=["vocab", "educ", "age", "nativeBorn"]).copy()
d["female"] = (d.gender == "female").astype(int)
d["foreign"] = (d.nativeBorn == "no").astype(int)
d["age10"] = (d.age - 45) / 10
d["decade"] = (d.year - 1997) / 10
d["educ12"] = d.educ - 12  # centred at high-school graduation
log(f"Analysis sample (no missing values in the variables used): {len(d)} respondents; foreign-born {d.foreign.mean() * 100:.1f}%; women {d.female.mean() * 100:.0f}%\n")

# ----------------------------------------------------------------------------
# 1. Education and vocabulary; linear or not?
# ----------------------------------------------------------------------------
m_lin = smf.ols("vocab ~ educ12", d).fit(cov_type="HC3")
log(f"Simple regression: one more year of education is associated with {m_lin.params['educ12']:.2f} more words correct "
    f"(95% CI {m_lin.conf_int().loc['educ12', 0]:.2f}-{m_lin.conf_int().loc['educ12', 1]:.2f}); R2 = {m_lin.rsquared:.2f}; "
    f"in SD units: {m_lin.params['educ12'] / d.vocab.std():.2f} SD per year")
grp = d.groupby("educGroup").vocab.agg(["mean", "std", "count"]).reindex(["<12 yrs", "12 yrs", "13-15 yrs", "16 yrs", ">16 yrs"])
grp["95% CI half-width"] = 1.96 * grp["std"] / np.sqrt(grp["count"])
log("\nMean vocabulary by education group")
log(grp.round(2).to_string())

m_cat = smf.ols("vocab ~ C(educ)", d).fit()
m_l = smf.ols("vocab ~ educ", d).fit()
F, p, _ = m_cat.compare_f_test(m_l)
log(f"\nLinearity of education: F-test of education as {int(m_cat.df_model)} categories vs a linear term: F = {F:.1f}, p = {p:.2g} "
    f"(R2 {m_l.rsquared:.3f} linear vs {m_cat.rsquared:.3f} categorical: the departure is statistically clear but small in size)\n")

# ----------------------------------------------------------------------------
# 2-3. Nested models
# ----------------------------------------------------------------------------
forms = {
    "M1 education": "vocab ~ educ12",
    "M2 + age, age^2, gender, foreign-born": "vocab ~ educ12 + age10 + I(age10**2) + female + foreign",
    "M3 + survey year (decades)": "vocab ~ educ12 + age10 + I(age10**2) + female + foreign + decade",
    "M4 + education x year": "vocab ~ educ12 + age10 + I(age10**2) + female + foreign + decade + educ12:decade",
    "M5 + education x foreign-born": "vocab ~ educ12 + age10 + I(age10**2) + female + foreign + decade + educ12:decade + educ12:foreign",
}
fits = {k: smf.ols(f, d).fit(cov_type="HC3") for k, f in forms.items()}
log("Nested OLS models, HC3 robust standard errors")
log(pd.DataFrame({"model": list(fits), "R2": [round(m.rsquared, 3) for m in fits.values()], "AIC": [round(m.aic, 0) for m in fits.values()]}).to_string(index=False), "\n")

m5 = fits["M5 + education x foreign-born"]
ci = m5.conf_int()
tab = pd.DataFrame({"coef": m5.params, "ci low": ci[0], "ci high": ci[1], "p": m5.pvalues}).round(3)
tab["p"] = tab["p"].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
log("Model M5 (coefficients in words correct out of 10; educ12 = years of education minus 12; decade = years since 1997 divided by 10)")
log(tab.to_string(), "\n")
sd = d.vocab.std()
log(f"Effect sizes in SD units: foreign-born gap {m5.params['foreign'] / sd:.2f} SD, gender gap (women - men) {m5.params['female'] / sd:.2f} SD, "
    f"change per decade {m5.params['decade'] / sd:.3f} SD")
age_peak = 45 - 10 * m5.params["age10"] / (2 * m5.params["I(age10 ** 2)"])
log(f"The age profile is a quadratic with its peak at about age {age_peak:.0f} (cross-sectional: age, birth cohort and survey year cannot be separated in a linear model).\n")

# ----------------------------------------------------------------------------
# 4. Robustness: the outcome is a count of correct answers out of 10 -> binomial (logit) regression
# ----------------------------------------------------------------------------
X = sm.add_constant(d[["educ12", "age10", "female", "foreign", "decade"]].assign(age10sq=d.age10 ** 2, educ12_dec=d.educ12 * d.decade))
y2 = np.column_stack([d.vocab, 10 - d.vocab])
gl = sm.GLM(y2, X, family=sm.families.Binomial()).fit(cov_type="HC1")
ols_s = smf.ols("vocab ~ educ12 + age10 + I(age10**2) + female + foreign + decade + educ12:decade", d).fit(cov_type="HC3")
log("Robustness: binomial logit (each of the 10 words is a correct/incorrect trial) vs OLS - same signs and significance?")
rows = []
for v in ["educ12", "female", "foreign", "decade", "educ12_dec"]:
    ols_name = {"educ12_dec": "educ12:decade"}.get(v, v)
    rows.append({"term": v, "OLS coef (words)": round(ols_s.params[ols_name], 3), "OLS p": f"{ols_s.pvalues[ols_name]:.3g}",
                 "binomial odds ratio": round(np.exp(gl.params[v]), 3), "binomial p": f"{gl.pvalues[v]:.3g}"})
log(pd.DataFrame(rows).to_string(index=False), "\n")

# ----------------------------------------------------------------------------
# 5. Missing outcome
# ----------------------------------------------------------------------------
mm = g.dropna(subset=["educ", "age", "nativeBorn"]).copy()
mm["miss"] = mm.vocab.isna().astype(int)
mm["female"] = (mm.gender == "female").astype(int)
mm["foreign"] = (mm.nativeBorn == "no").astype(int)
mm["age10"] = (mm.age - 45) / 10
mm["decade"] = (mm.year - 1997) / 10
mm["educ12"] = mm.educ - 12
lm = smf.logit("miss ~ educ12 + age10 + female + foreign + decade", mm).fit(disp=False)
log("Is a missing vocabulary score related to the covariates? Logit for 'vocabulary missing' (odds ratios)")
log(pd.DataFrame({"odds ratio": np.exp(lm.params).round(2), "p": lm.pvalues.map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")}).to_string())
log("(Missingness is related to some covariates, so complete-case results may be biased; the size of the missing share is small.)\n")

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].errorbar(range(5), grp["mean"], yerr=grp["95% CI half-width"], fmt="o-", capsize=3)
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(grp.index, fontsize=8)
axes[0].set_ylabel("mean words correct (of 10)")
axes[0].set_title("Vocabulary by education")
by = d.groupby(["year", "educGroup"]).vocab.mean().unstack().reindex(columns=grp.index)
by.plot(ax=axes[1], marker="o", ms=3, lw=1)
axes[1].set_title("Vocabulary by survey year and education group")
axes[1].set_ylabel("mean words correct")
axes[1].legend(fontsize=7, title="education")
fig.tight_layout()
fig.savefig(HERE / "figures" / "vocab_by_education_and_year.png", dpi=150)
plt.close(fig)

log("Caveats: the GSS is a complex survey; the sampling weights, strata and clusters are not in this extract, so the standard errors")
log("assume simple random sampling and are too small, and the descriptive means are not exactly population estimates; a test with 10")
log("items has a ceiling and floor; vocabulary is an imperfect proxy for cognitive ability and depends on language background;")
log("the cross-sectional comparisons of age groups mix age and cohort; education is partly a consequence of ability, so the education")
log("coefficient is not the causal effect of schooling; the 'foreign-born' variable does not capture time in the US or first language.")
out.close()
print("\nDone. See results.txt and figures/.")
