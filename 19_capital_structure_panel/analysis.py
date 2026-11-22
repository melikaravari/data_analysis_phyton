"""
Project 19 - What determines corporate leverage? Panel-data econometrics with Compustat

Data: Compustat North America, annual fundamentals (WRDS), fiscal years 2000-2024. NOT included in this repository (licensed data).
Put your own WRDS extract at  data/compustat_funda.csv  (variables: gvkey, datadate, fyear, curcd, fic, sic, at, ceq, dlc, dltt,
ppent, oibdp, xrd, csho, prcc_f, plus others; screens consol = C, indfmt = INDL, datafmt = STD, popsrc = D).

Questions
  1. Which firm characteristics are related to book leverage (debt / total assets): size, profitability, asset tangibility,
     market-to-book, R&D? Do the signs agree with trade-off and pecking-order theories?
  2. How much do the estimates change when we control for unobserved firm heterogeneity (firm fixed effects)?
  3. Which specification is appropriate (Mundlak/Hausman-type test), and how sensitive are the results to the measure of leverage,
     to firm size, to the standard errors and to the period?

Theory (signs of the coefficients)
  Trade-off theory: size +, tangibility +, profitability + (more tax shield use, lower bankruptcy risk), market-to-book -
  Pecking order:    profitability -, market-to-book -  (internal funds first; growth firms keep debt capacity)
  Empirically the best-known result is a NEGATIVE profitability coefficient (Rajan & Zingales 1995; Frank & Goyal 2009).

Run:  python analysis.py      (needs pandas numpy scipy statsmodels linearmodels matplotlib)
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
from linearmodels.panel import PanelOLS
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# 1. Data preparation (every filter is reported)
# ----------------------------------------------------------------------------
raw = pd.read_csv(HERE.parent / "data" / "compustat_funda.csv", dtype={"gvkey": str, "sic": str}, parse_dates=["datadate"])
log(f"Raw extract: {len(raw):,} firm-years, {raw.gvkey.nunique():,} firms")
d = raw.sort_values(["gvkey", "fyear", "datadate"])
n = len(d)
d = d.drop_duplicates(["gvkey", "fyear"], keep="last")
log(f"1. Removed {n - len(d)} duplicate firm-years (fiscal-year-end changes); kept the latest statement")
n = len(d)
d = d[d.fyear.between(2000, 2024)]
log(f"2. Fiscal years 2000-2024: removed {n - len(d):,}")
n = len(d)
d = d[(d.curcd == "USD") & (d.fic == "USA")]
log(f"3. US-incorporated firms reporting in USD (no currency mixing): removed {n - len(d):,}")
n = len(d)
sic = pd.to_numeric(d.sic, errors="coerce")
d = d[~sic.between(4900, 4999) & ~sic.between(6000, 6999)]
log(f"4. Excluded utilities (SIC 49xx) and financial firms (SIC 6xxx), whose leverage is driven by regulation: removed {n - len(d):,}")
n = len(d)
d = d[d["at"] >= 10]
log(f"5. Total assets of at least USD 10 million (removes tiny shells): removed {n - len(d):,}")
n = len(d)
d = d.dropna(subset=["dlc", "dltt", "ppent", "oibdp", "csho", "prcc_f"])
log(f"6. Complete values for debt, PPE, EBITDA, shares, price: removed {n - len(d):,}")

d["debt"] = d.dlc + d.dltt.fillna(0)
d["mktcap"] = d.csho * d.prcc_f
d["lev"] = d.debt / d["at"]                              # book leverage (main dependent variable)
d["lev_mkt"] = d.debt / (d.debt + d.mktcap)              # market leverage (robustness)
d["size"] = np.log(d["at"])                              # ln(total assets, USD million)
d["prof"] = d.oibdp / d["at"]                            # EBITDA / assets
d["tang"] = d.ppent / d["at"]                            # net PPE / assets
d["mtb"] = (d["at"] - d.ceq + d.mktcap) / d["at"]        # market-to-book assets
d["rd"] = (d.xrd.fillna(0)) / d["at"]                    # R&D / assets (missing R&D treated as zero)
d["rd_missing"] = d.xrd.isna().astype(int)
d["sic2"] = (pd.to_numeric(d.sic, errors="coerce") // 100).fillna(-1).astype(int)
d = d.dropna(subset=["lev", "lev_mkt", "size", "prof", "tang", "mtb"])
d = d[np.isfinite(d[["lev", "lev_mkt", "size", "prof", "tang", "mtb", "rd"]]).all(axis=1)]

# Winsorise ratios at the 1st and 99th percentile (outliers from tiny denominators)
RATIOS = ["lev", "lev_mkt", "prof", "tang", "mtb", "rd"]
for c in RATIOS:
    lo, hi = d[c].quantile([0.01, 0.99])
    d[c] = d[c].clip(lo, hi)

# Regressors are lagged by one fiscal year (only if the previous year is observed): characteristics at t-1 explain leverage at t.
d = d.sort_values(["gvkey", "fyear"])
X = ["size", "prof", "tang", "mtb", "rd"]
for c in X + ["rd_missing"]:
    prev = d.groupby("gvkey")[c].shift(1)
    ok = d.groupby("gvkey")["fyear"].diff() == 1
    d[c + "_l"] = prev.where(ok)
XL = [c + "_l" for c in X] + ["rd_missing_l"]
d = d.dropna(subset=XL)
cnt = d.groupby("gvkey").size()
d = d[d.gvkey.isin(cnt[cnt >= 3].index)]
log(f"7. Lagged regressors available and at least 3 observations per firm: final sample {len(d):,} firm-years, {d.gvkey.nunique():,} firms, "
    f"{d.fyear.min()}-{d.fyear.max()} (average {len(d) / d.gvkey.nunique():.1f} years per firm)\n")

# ----------------------------------------------------------------------------
# 2. Descriptive statistics, between and within variation, collinearity
# ----------------------------------------------------------------------------
vars_ = ["lev", "lev_mkt", "size", "prof", "tang", "mtb", "rd"]
desc = d[vars_].describe().T[["mean", "std", "min", "50%", "max"]].round(3)
between = d.groupby("gvkey")[vars_].mean().std().round(3)
within = (d[vars_] - d.groupby("gvkey")[vars_].transform("mean")).std().round(3)
desc["between SD"], desc["within SD"] = between, within
log("Summary statistics (winsorised ratios). 'between SD' = variation of firm averages, 'within SD' = variation over time inside firms")
log(desc.to_string(), "\n")
log("Tangibility and size vary mostly BETWEEN firms; a firm-fixed-effects estimator uses only the within variation, so these coefficients can change a lot.\n")

Xv = d[[c + "_l" for c in X]].assign(const=1.0)
vif = pd.Series([variance_inflation_factor(Xv.values, i) for i in range(len(X))], index=X).round(2)
log("Variance inflation factors (values above 5-10 signal problematic collinearity): " + ", ".join(f"{k} {v}" for k, v in vif.items()) + "\n")

# ----------------------------------------------------------------------------
# 3. Models. All use lagged regressors; standard errors clustered by firm unless stated.
# ----------------------------------------------------------------------------
dd = d.set_index(["gvkey", "fyear"])
names = {"size_l": "Size (ln assets)", "prof_l": "Profitability", "tang_l": "Tangibility", "mtb_l": "Market-to-book", "rd_l": "R&D / assets"}


def fe_fit(y, entity=True, time=True, cluster_time=False, data=dd, regs=None):
    regs = regs or XL
    m = PanelOLS(data[y], data[regs], entity_effects=entity, time_effects=time, drop_absorbed=True)
    return m.fit(cov_type="clustered", cluster_entity=True, cluster_time=cluster_time)


def pooled_fit(y, formula_extra="", data=d):
    f = f"{y} ~ size_l + prof_l + tang_l + mtb_l + rd_l + rd_missing_l {formula_extra}"
    return smf.ols(f, data).fit(cov_type="cluster", cov_kwds={"groups": data.gvkey.astype("category").cat.codes})


m1 = pooled_fit("lev")                                              # (1) pooled OLS, no fixed effects
m2 = pooled_fit("lev", "+ C(fyear) + C(sic2)")                      # (2) pooled OLS with year and 2-digit industry effects
m3 = fe_fit("lev", entity=True, time=False)                         # (3) firm fixed effects
m4 = fe_fit("lev", entity=True, time=True)                          # (4) firm and year fixed effects

rows = []
for c in XL[:5]:
    r = {"variable": names[c]}
    for label, m in [("(1) Pooled OLS", m1), ("(2) + year, industry FE", m2), ("(3) Firm FE", m3), ("(4) Firm + year FE", m4)]:
        p, se = m.params[c], m.std_errors[c] if hasattr(m, "std_errors") else m.bse[c]
        pv = m.pvalues[c]
        star = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
        r[label] = f"{p:.3f}{star} ({se:.3f})"
    rows.append(r)
tab = pd.DataFrame(rows).set_index("variable")
nobs = {"(1) Pooled OLS": int(m1.nobs), "(2) + year, industry FE": int(m2.nobs), "(3) Firm FE": int(m3.nobs), "(4) Firm + year FE": int(m4.nobs)}
r2 = {"(1) Pooled OLS": m1.rsquared, "(2) + year, industry FE": m2.rsquared, "(3) Firm FE": m3.rsquared_within, "(4) Firm + year FE": m4.rsquared_within}
log("Dependent variable: book leverage (debt / total assets). Coefficients with firm-clustered standard errors in parentheses.")
log("Significance: *** 1%, ** 5%, * 10%. Models (1)-(2) also include a missing-R&D dummy; (3)-(4) R-squared is the WITHIN R-squared.")
log(tab.to_string())
log("Observations: " + ", ".join(f"{k} {v:,}" for k, v in nobs.items()))
log("R-squared:    " + ", ".join(f"{k} {v:.3f}" for k, v in r2.items()), "\n")

# ----------------------------------------------------------------------------
# 4. Which specification? Mundlak test (cluster-robust alternative to the Hausman test),
#    F-test for the joint significance of the firm effects
# ----------------------------------------------------------------------------
dm = d.copy()
for c in XL[:5]:
    dm[c + "_bar"] = dm.groupby("gvkey")[c].transform("mean")
bars = [c + "_bar" for c in XL[:5]]
mun = smf.ols("lev ~ " + " + ".join(XL + bars) + " + C(fyear)", dm).fit(cov_type="cluster", cov_kwds={"groups": dm.gvkey.astype("category").cat.codes})
R = np.zeros((5, len(mun.params)))
for i, b in enumerate(bars):
    R[i, list(mun.params.index).index(b)] = 1
w = mun.wald_test(R, scalar=True)
log(f"Mundlak test: are the firm averages of the regressors jointly significant when added to a random-effects-type regression? "
    f"Wald chi2(5) = {float(w.statistic):.1f}, p = {float(w.pvalue):.3g}")
log("-> a significant result means that firm effects are correlated with the regressors, so pooled OLS / random effects are inconsistent and fixed effects are preferred.")
fe_test = m4.f_pooled
log(f"F-test that all firm effects are zero (model 4): F = {fe_test.stat:.1f}, p {"< 0.001" if fe_test.pval < 0.001 else "= " + format(fe_test.pval, ".3g")}\n")

# ----------------------------------------------------------------------------
# 5. Robustness (firm + year fixed effects unless noted)
# ----------------------------------------------------------------------------
rows = []


def add(label, m, sample=""):
    row = {"specification": label, "N": int(m.nobs)}
    for c in XL[:5]:
        if c not in m.params.index:
            row[names[c]] = "-"
            continue
        pv = m.pvalues[c]
        star = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
        row[names[c]] = f"{m.params[c]:.3f}{star} ({m.std_errors[c]:.3f})"
    rows.append(row)


add("Baseline (model 4)", m4)
add("Market leverage as dependent variable", fe_fit("lev_mkt"))
add("Two-way clustered SE (firm and year)", fe_fit("lev", cluster_time=True))
big = dd[dd["at"] >= 100]
add("Only firms with assets >= USD 100 million", fe_fit("lev", data=big))
add("Fiscal years 2000-2007", fe_fit("lev", data=dd[dd.index.get_level_values("fyear") <= 2007]))
add("Fiscal years 2008-2015", fe_fit("lev", data=dd[dd.index.get_level_values("fyear").isin(range(2008, 2016))]))
add("Fiscal years 2016-2024", fe_fit("lev", data=dd[dd.index.get_level_values("fyear") >= 2016]))
add("Without R&D variable", fe_fit("lev", regs=[c for c in XL if not c.startswith("rd")]))
rob = pd.DataFrame(rows).set_index("specification")
log("Robustness: firm and year fixed effects, coefficients with standard errors in parentheses (stars as above)")
log(rob.to_string(), "\n")

# ----------------------------------------------------------------------------
# 6. Economic magnitude: effect of a one-standard-deviation WITHIN-firm change (model 4)
# ----------------------------------------------------------------------------
log("Economic magnitude (model 4): change in book leverage, in percentage points, for a one-SD change within a firm")
mag = []
for c in XL[:5]:
    base = c[:-2]
    mag.append({"variable": names[c], "within SD": round(float(within[base]), 3), "effect (pp of leverage)": round(m4.params[c] * float(within[base]) * 100, 2)})
log(pd.DataFrame(mag).to_string(index=False))
log(f"For comparison: the standard deviation of book leverage is {desc.loc['lev', 'std'] * 100:.1f} pp overall and {within['lev'] * 100:.1f} pp within firms.\n")

# ----------------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
yearly = d.groupby("fyear")[["lev", "lev_mkt"]].mean() * 100
yearly.plot(ax=axes[0], marker="o", ms=3)
axes[0].set_title("Average leverage by fiscal year (%)")
axes[0].set_xlabel("")
axes[0].legend(["book leverage", "market leverage"], fontsize=8)
labels = list(names.values())
for label, m, off in [("Pooled OLS", m1, -0.15), ("Firm + year FE", m4, 0.15)]:
    est = [m.params[c] for c in XL[:5]]
    se = [m.std_errors[c] if hasattr(m, "std_errors") else m.bse[c] for c in XL[:5]]
    axes[1].errorbar(est, np.arange(5) + off, xerr=1.96 * np.array(se), fmt="o", label=label, capsize=3)
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_yticks(range(5))
axes[1].set_yticklabels(labels)
axes[1].set_title("Coefficients (95% CI, firm-clustered)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "figures" / "leverage_and_coefficients.png", dpi=150)
plt.close(fig)

log("Caveats: the associations are not causal (leverage and profitability are jointly determined; omitted time-varying firm factors);")
log("lagging the regressors reduces but does not remove simultaneity; fixed effects remove time-invariant heterogeneity only; with a lagged")
log("dependent variable (partial adjustment) fixed effects would suffer from Nickell bias, which this static model avoids; winsorising and the")
log("USD 10 million asset filter are choices; Compustat covers listed firms only, so private firms (the majority) are not represented;")
log("book leverage is affected by accounting rules (e.g. leases after 2019).")
out.close()
print("\nDone. See results.txt and figures/.")
