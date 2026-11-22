"""
Project 20 - Difference-in-differences: did a planned waste incinerator lower nearby house prices?

Data: `kielmc` (Kiel & McClain 1995; in Wooldridge's textbook data): 321 house sales in North Andover, Massachusetts, in 1978 and 1981
(179 in 1978, 142 in 1981). A garbage incinerator was being planned and built around this period (the textbook treats 1978 as 'before' and 1981 as 'after'; I did not verify the exact
timeline); houses within 3 miles of it are the "treated" group (`nearinc`).
Different houses are observed in the two years (repeated cross-sections, not a panel). Prices are in 1978 dollars (`rprice`).

Questions
  1. What is the difference-in-differences estimate of the effect of the incinerator on the price of nearby houses?
  2. Is it robust to controls, to the functional form (log price), to the standard error method, and to a randomisation test?
  3. How credible is the identifying assumption (parallel trends) with only two periods, and did the sample composition change?

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

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")
rng = np.random.default_rng(3)


def log(*a):
    print(*a)
    print(*a, file=out)


d = pd.read_csv(HERE.parent / "data" / "kielmc.csv")
d["post"] = d.y81
d["treat"] = d.nearinc
d["did"] = d.post * d.treat
d["price_k"] = d.rprice / 1000
log(f"{len(d)} house sales; 1978: {int((d.year == 1978).sum())}, 1981: {int((d.year == 1981).sum())}; near the incinerator: {int(d.treat.sum())}")

# ----------------------------------------------------------------------------
# 1. The 2x2 table of mean prices
# ----------------------------------------------------------------------------
m = d.groupby(["treat", "post"]).price_k.mean().unstack()
m.index = ["far (> 3 miles)", "near (<= 3 miles)"]
m.columns = ["1978", "1981"]
log("\nMean real price (USD thousand, 1978 dollars)")
log(m.round(1).to_string())
diff_far = m.loc["far (> 3 miles)", "1981"] - m.loc["far (> 3 miles)", "1978"]
diff_near = m.loc["near (<= 3 miles)", "1981"] - m.loc["near (<= 3 miles)", "1978"]
log(f"Change 1978-1981: far {diff_far:+.1f}, near {diff_near:+.1f}; difference-in-differences = {diff_near - diff_far:+.1f} thousand dollars\n")
log(f"In 1978 (the 'before' year) near houses were already cheaper by {m.loc['far (> 3 miles)', '1978'] - m.loc['near (<= 3 miles)', '1978']:.1f} thousand: "
    "levels differ between groups, and the DiD only needs trends to be parallel.\n")

# ----------------------------------------------------------------------------
# 2. Regression DiD, several specifications
# ----------------------------------------------------------------------------
controls = "age + I(age**2) + intst + land + area + rooms + baths"
specs = {
    "(1) Levels, no controls": "rprice ~ post + treat + did",
    "(2) Levels, with controls": f"rprice ~ post + treat + did + {controls}",
    "(3) Log price, no controls": "lrprice ~ post + treat + did",
    "(4) Log price, with controls": f"lrprice ~ post + treat + did + {controls}",
}
rows, fits = [], {}
for name, f in specs.items():
    r = smf.ols(f, d).fit(cov_type="HC3")
    fits[name] = r
    ci = r.conf_int().loc["did"]
    rows.append({"specification": name, "DiD estimate": r.params["did"], "robust SE (HC3)": r.bse["did"], "95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]", "p-value": r.pvalues["did"], "R2": r.rsquared})
tab = pd.DataFrame(rows).set_index("specification")
tab = tab.round({"DiD estimate": 3, "robust SE (HC3)": 3, "p-value": 3, "R2": 3})
log("Difference-in-differences estimates (coefficient on near x 1981); levels in 1978 dollars, logs in log points")
log(tab.to_string(), "\n")
pv = pd.Series({k: r.pvalues["did"] for k, r in fits.items()})
n_sig = int((pv < 0.05).sum())
log(f"All four estimates are negative, but only {n_sig} of 4 reaches p < 0.05 (and that one only just: p = {pv.min():.4f}); the 95% intervals are wide and in {4 - n_sig} of the four include zero.")
log("Controls matter because the mix of houses sold differs between years (see the composition check below).\n")

# ----------------------------------------------------------------------------
# 3. Standard errors: HC3 vs cluster by neighbourhood (few clusters) vs randomisation inference
# ----------------------------------------------------------------------------
f4 = specs["(4) Log price, with controls"]
r_cl = smf.ols(f4, d).fit(cov_type="cluster", cov_kwds={"groups": d.nbh})
log(f"Cluster-robust SE by neighbourhood ({d.nbh.nunique()} clusters; too few for reliable cluster inference): DiD {r_cl.params['did']:.3f}, SE {r_cl.bse['did']:.3f}, p = {r_cl.pvalues['did']:.3f}")

# Randomisation inference: reassign the 'near' label at random among houses, keeping the sample sizes fixed, 2,000 times
obs = fits["(4) Log price, with controls"].params["did"]
perm = []
dd = d.copy()
for _ in range(2000):
    dd["treat"] = rng.permutation(d.treat.values)
    dd["did"] = dd.post * dd.treat
    perm.append(smf.ols(f4, dd).fit().params["did"])
perm = np.array(perm)
log(f"Randomisation inference (2,000 random reassignments of 'near'): observed {obs:.3f}; two-sided p = {(np.abs(perm) >= abs(obs)).mean():.3f}")
log("(This tests the sharp null of no effect for any house; it ignores that near and far houses differ systematically in 1978.)\n")

# ----------------------------------------------------------------------------
# 4. Dose-response: distance to the incinerator instead of near/far
# ----------------------------------------------------------------------------
r_d = smf.ols(f"lrprice ~ post + ldist + post:ldist + {controls}", d).fit(cov_type="HC3")
log(f"Continuous version: coefficient on post x ln(distance) = {r_d.params['post:ldist']:.3f} (SE {r_d.bse['post:ldist']:.3f}, p = {r_d.pvalues['post:ldist']:.3f}); "
    f"a 10% larger distance is associated with a {r_d.params['post:ldist'] * np.log(1.1) * 100:.1f}% smaller fall in price after 1981\n")

# ----------------------------------------------------------------------------
# 5. Diagnostics: composition of the sample and balance
# ----------------------------------------------------------------------------
bal = d.groupby(["post", "treat"])[["age", "area", "land", "rooms", "baths", "intst"]].mean().round(1)
log("Average house characteristics by year and group (repeated cross-section: composition can change)")
log(bal.to_string())
chg = d.groupby("post")[["age", "area", "rooms", "baths"]].mean()
log(f"\nAverage age of houses sold: {chg.loc[0, 'age']:.0f} years in 1978 vs {chg.loc[1, 'age']:.0f} in 1981 -> the sold houses differ between years, which is why controls matter.\n")
log("Note: the 'placebo' test for pre-trends is not possible with one pre-period; the variable `nearinc` is also defined by a 3-mile radius,")
log("so the treatment boundary is arbitrary.\n")

fig, ax = plt.subplots(figsize=(6, 4))
for lbl, row in m.iterrows():
    ax.plot([1978, 1981], row.values, "o-", label=lbl)
ax.plot([1978, 1981], [m.loc["near (<= 3 miles)", "1978"], m.loc["near (<= 3 miles)", "1978"] + diff_far], "k:", label="near, if it had followed the far trend")
ax.set_ylabel("mean real price (USD thousand, 1978 dollars)")
ax.set_xticks([1978, 1981])
ax.legend(fontsize=8)
ax.set_title("Difference-in-differences: house prices")
fig.tight_layout()
fig.savefig(HERE / "figures" / "did_means.png", dpi=150)
plt.close(fig)

log("Caveats: only two time periods, so parallel trends cannot be tested; repeated cross-sections with a changing mix of houses; the")
log("treatment group is defined by a distance threshold; only 7 neighbourhoods, too few for cluster-robust standard errors; the 1978 'before' period")
log("may already reflect anticipation of the incinerator; 321 observations give wide confidence intervals;")
log("no check for spillovers (houses beyond 3 miles may also be affected).")
out.close()
print("\nDone. See results.txt and figures/.")
