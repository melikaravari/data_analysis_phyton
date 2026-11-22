"""
Project 21 - Instrumental variables: the return to education (Card 1995)

Data: `card` (David Card's analysis of the National Longitudinal Survey of Young Men, in Wooldridge's textbook data): 3,010 men, wages in 1976
(`lwage` = log hourly wage), years of education, experience, region, race, parental education, test scores, and indicators for growing up near a
2-year (`nearc2`) or 4-year (`nearc4`) college.

Question: what is the causal effect of one more year of schooling on wages? OLS is biased if ability or family background affect both schooling and
wages (omitted variables) or if schooling is measured with error. Instrument: proximity to a four-year college in 1966, which should shift schooling
costs but (hopefully) not wages directly.

Steps: OLS; first stage and instrument strength; reduced form; 2SLS; endogeneity test; over-identification test; weak-instrument-robust
(Anderson-Rubin) confidence interval; checks of the exclusion restriction.

Run:  python analysis.py      (needs numpy pandas scipy statsmodels linearmodels matplotlib)
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
from linearmodels.iv import IV2SLS
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


d = pd.read_csv(HERE.parent / "data" / "card.csv")
reg = [f"reg66{i}" for i in range(2, 10)]  # region 1966 dummies; region 1 is the baseline
ctrl = ["exper", "expersq", "black", "smsa", "south", "smsa66"] + reg  # Card's basic controls (region in 1966 dummies)
log(f"{len(d)} men; mean log wage {d.lwage.mean():.2f}; mean education {d.educ.mean():.1f} years; share growing up near a 4-year college {d.nearc4.mean() * 100:.0f}%\n")

# ----------------------------------------------------------------------------
# 1. OLS
# ----------------------------------------------------------------------------
X = sm.add_constant(d[["educ"] + ctrl])
ols = sm.OLS(d.lwage, X).fit(cov_type="HC1")
log(f"OLS: one more year of education is associated with {ols.params['educ'] * 100:.1f}% higher wages (SE {ols.bse['educ'] * 100:.1f} pp, 95% CI "
    f"{ols.conf_int().loc['educ', 0] * 100:.1f}-{ols.conf_int().loc['educ', 1] * 100:.1f}%)\n")

# ----------------------------------------------------------------------------
# 2. First stage and reduced form
# ----------------------------------------------------------------------------
fs = smf.ols("educ ~ nearc4 + " + " + ".join(ctrl), d).fit(cov_type="HC1")
F = (fs.tvalues["nearc4"]) ** 2
log(f"First stage: growing up near a 4-year college is associated with {fs.params['nearc4']:.2f} more years of education "
    f"(SE {fs.bse['nearc4']:.2f}); F-statistic for the instrument = {F:.1f} (rule of thumb: > 10; Stock-Yogo 10% size critical value 16.4 for one instrument)")
rf = smf.ols("lwage ~ nearc4 + " + " + ".join(ctrl), d).fit(cov_type="HC1")
log(f"Reduced form: near a 4-year college -> log wage {rf.params['nearc4']:.3f} (SE {rf.bse['nearc4']:.3f}, p = {rf.pvalues['nearc4']:.3f}). "
    f"Wald/IV ratio = {rf.params['nearc4'] / fs.params['nearc4']:.3f}\n")

# ----------------------------------------------------------------------------
# 3. 2SLS
# ----------------------------------------------------------------------------
def iv(instr, data=d):
    dd = data.dropna(subset=["lwage", "educ"] + ctrl + instr)
    m = IV2SLS(dd["lwage"], sm.add_constant(dd[ctrl]), dd[["educ"]], dd[instr]).fit(cov_type="robust")
    return m, dd


rows = [{"model": "OLS", "instruments": "-", "N": int(ols.nobs), "return to education": ols.params["educ"], "SE": ols.bse["educ"]}]
res = {}
for label, instr in [("2SLS", ["nearc4"]), ("2SLS", ["nearc4", "nearc2"]), ("2SLS", ["fatheduc", "motheduc"]), ("2SLS", ["nearc4", "fatheduc", "motheduc"])]:
    m, dd = iv(instr)
    res[tuple(instr)] = m
    rows.append({"model": label, "instruments": " + ".join(instr), "N": int(m.nobs), "return to education": m.params["educ"], "SE": m.std_errors["educ"]})
tab = pd.DataFrame(rows)
tab["95% CI"] = [f"[{r['return to education'] - 1.96 * r['SE']:.3f}, {r['return to education'] + 1.96 * r['SE']:.3f}]" for _, r in tab.iterrows()]
log("Return to one year of education (log points per year)")
log(tab.round({"return to education": 3, "SE": 3}).to_string(index=False), "\n")
log("The 2SLS estimate with nearc4 is clearly larger than the OLS estimate (0.13 vs 0.08) but has a much wider confidence interval. Explanations: (i) measurement error")
log("in schooling attenuates OLS; (ii) the IV estimates a local effect for men whose schooling responds to college proximity (often high-return, credit-")
log("constrained men), not the average effect (LATE); (iii) the instrument may be imperfectly exogenous. These cannot be separated with these data.\n")

# ----------------------------------------------------------------------------
# 4. Tests: endogeneity (Wu-Hausman), over-identification (Sargan), weak instruments
# ----------------------------------------------------------------------------
m_one = res[("nearc4",)]
wu = m_one.wu_hausman()
log(f"Wu-Hausman test of exogeneity of education (nearc4 as instrument): statistic {wu.stat:.2f}, p = {wu.pval:.3f}  "
    f"(low power here: IV is imprecise, so OLS cannot be rejected formally although the point estimates differ)")
m_ov = res[("nearc4", "nearc2")]
j = m_ov.sargan
log(f"Over-identification test, nearc4 + nearc2: Sargan statistic {j.stat:.2f}, p = {j.pval:.3f}")
m_par = res[("nearc4", "fatheduc", "motheduc")]
log(f"Over-identification test, nearc4 + father's + mother's education: Sargan {m_par.sargan.stat:.2f}, p = {m_par.sargan.pval:.3f}")
log("Parental education is a very strong instrument (F > 200) but is the least credible one: family background probably affects wages directly, so its estimate (about 0.10) is not trusted more than the college-proximity one.")
log("Failing to reject is reassuring but not proof: the test is only valid if at least one instrument is valid, and has limited power.")
for k, m in res.items():
    f = m.first_stage.diagnostics.loc["educ", "f.stat"]
    log(f"  First-stage partial F for instruments {k}: {f:.1f}")
log("")

# ----------------------------------------------------------------------------
# 5. Anderson-Rubin confidence set (valid even if the instrument is weak), nearc4 only
# ----------------------------------------------------------------------------
grid = np.linspace(-0.2, 0.7, 181)
Z = d[["nearc4"]]
pvals = []
for b0 in grid:
    y0 = d.lwage - b0 * d.educ
    r_ = smf.ols("y0 ~ nearc4 + " + " + ".join(ctrl), d.assign(y0=y0)).fit(cov_type="HC1")
    pvals.append(r_.pvalues["nearc4"])
pvals = np.array(pvals)
inside = grid[pvals > 0.05]
log(f"Anderson-Rubin 95% confidence set for the return to education (nearc4): [{inside.min():.2f}, {inside.max():.2f}] "
    f"(2SLS: [{m_one.params['educ'] - 1.96 * m_one.std_errors['educ']:.2f}, {m_one.params['educ'] + 1.96 * m_one.std_errors['educ']:.2f}])\n")

# ----------------------------------------------------------------------------
# 6. Exclusion restriction: is proximity unrelated to other determinants of wages?
# ----------------------------------------------------------------------------
log("Balance of predetermined characteristics by proximity to a 4-year college (difference near - not near)")
rows = []
for v in ["fatheduc", "motheduc", "IQ", "KWW", "black", "smsa66", "south66"]:
    dd = d.dropna(subset=[v])
    a, b = dd.loc[dd.nearc4 == 1, v], dd.loc[dd.nearc4 == 0, v]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    rows.append({"variable": v, "mean near": a.mean(), "mean not near": b.mean(), "difference": a.mean() - b.mean(), "p-value": p, "N": len(dd)})
log(pd.DataFrame(rows).round(3).to_string(index=False))
iq = smf.ols("IQ ~ nearc4 + " + " + ".join(ctrl), d.dropna(subset=["IQ"])).fit(cov_type="HC1")
log(f"After controlling for region and urban location: IQ difference {iq.params['nearc4']:.2f} points (p = {iq.pvalues['nearc4']:.3f}). "
    "Raw differences in ability and family background between near and far men suggest the instrument may not be randomly assigned;")
log("the controls absorb part of this. This is the main threat to the IV result.\n")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
axes[0].plot(grid, pvals, lw=1.2)
axes[0].axhline(0.05, color="r", ls="--", lw=0.8)
axes[0].set_xlabel("hypothesised return to education")
axes[0].set_ylabel("Anderson-Rubin p-value")
axes[0].set_title("Anderson-Rubin confidence set (p > 0.05)")
pts = [("OLS", ols.params["educ"], ols.bse["educ"])] + [(" + ".join(k), m.params["educ"], m.std_errors["educ"]) for k, m in res.items()]
for i, (lbl, est, se) in enumerate(pts):
    axes[1].errorbar(est, i, xerr=1.96 * se, fmt="o", capsize=3)
axes[1].set_yticks(range(len(pts)))
axes[1].set_yticklabels([p[0] for p in pts], fontsize=8)
axes[1].axvline(0, color="k", lw=0.6)
axes[1].set_title("Estimated return to education (95% CI)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "iv_estimates.png", dpi=150)
plt.close(fig)

log("Caveats: the exclusion restriction (proximity affects wages only through schooling) cannot be tested directly; urban areas near colleges may have")
log("different labour markets; the data are from 1976 for one cohort of men, so the result may not hold today or for women; the sample with parental")
log("education is smaller and selected; IV estimates are local effects; confidence intervals are wide.")
out.close()
print("\nDone. See results.txt and figures/.")
