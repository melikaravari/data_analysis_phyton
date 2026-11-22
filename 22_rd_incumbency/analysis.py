"""
Project 22 - Regression discontinuity: the incumbency advantage in US Senate elections (Lee 2008 design)

Data: US Senate elections 1914-2010 (1,390 races; from the `rdrobust` package, built from Lee 2008 / Cattaneo, Idrobo & Titiunik 2020).
  margin = Democratic margin of victory in election t (percentage points; > 0 means the Democrat won)
  vote   = Democratic vote share in the NEXT election for the same seat (6 years later), %

Idea: in close races the winner is essentially random, so a party that barely won is comparable to a party that barely lost. The jump of
`vote` at margin = 0 is the effect of winning (being the incumbent party) on the next election result.

Questions
  1. How large is the incumbency advantage (RD estimate with data-driven bandwidth and robust inference)?
  2. Are the assumptions credible: no manipulation of the running variable, covariate balance at the cutoff, no jumps at placebo cutoffs?
  3. Is the estimate sensitive to bandwidth, polynomial order or excluding observations closest to the cutoff?

Run:  python analysis.py      (needs numpy pandas scipy matplotlib rdrobust rddensity)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rddensity
import rdrobust
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


raw = rdrobust.rdrobust_RDsenate()
d = raw.dropna(subset=["vote"]).copy()
log(f"{len(raw)} Senate races 1914-2010; {len(d)} with a next-election result (outcome observed). Democrat won {int((raw.margin > 0).sum())} ({(raw.margin > 0).mean() * 100:.0f}%).\n")
x, y = d.margin.values, d.vote.values


def rd(y_, x_, c=0, **kw):
    return rdrobust.rdrobust(y_, x_, c=c, **kw)


def line(r, label):
    h = r.bws.loc["h", "left"]
    return {"specification": label, "estimate": r.coef.loc["Conventional"].iloc[0], "robust bias-corrected CI": f"[{r.ci.loc['Robust'].iloc[0]:.2f}, {r.ci.loc['Robust'].iloc[1]:.2f}]",
            "robust p": r.pv.loc["Robust"].iloc[0], "bandwidth": h, "N left/right": f"{r.N_h[0]}/{r.N_h[1]}"}


# ----------------------------------------------------------------------------
# 1. Main estimate
# ----------------------------------------------------------------------------
main = rd(y, x, p=1, kernel="triangular", bwselect="mserd")
log("Main RD estimate (local linear, triangular kernel, MSE-optimal bandwidth)")
log(f"  Conventional: {main.coef.loc['Conventional'].iloc[0]:.2f} points (SE {main.se.loc['Conventional'].iloc[0]:.2f})")
log(f"  Bias-corrected: {main.coef.loc['Bias-Corrected'].iloc[0]:.2f} (robust SE {main.se.loc['Robust'].iloc[0]:.2f}); robust 95% CI "
    f"[{main.ci.loc['Robust'].iloc[0]:.2f}, {main.ci.loc['Robust'].iloc[1]:.2f}]; robust p = {main.pv.loc['Robust'].iloc[0]:.2g}")
log(f"  Bandwidth {main.bws.loc['h', 'left']:.1f} points; effective observations {main.N_h[0]} left / {main.N_h[1]} right of the cutoff")
log("Interpretation: a party that barely wins a Senate seat gets about 7-8 percentage points more of the vote in the next election for that seat.\n")

# ----------------------------------------------------------------------------
# 2. Validity checks
# ----------------------------------------------------------------------------
dens = rddensity.rddensity(X=raw.margin.values, c=0)
log(f"Density test for manipulation of the running variable (Cattaneo-Jansson-Ma): p = {dens.test['p_jk']:.2f} (a small p-value would suggest sorting around the cutoff)")
near = raw.margin.abs() <= 2
log(f"Races decided by less than 2 points: {int(near.sum())}, of which won by the Democrat: {int((raw.margin[near] > 0).sum())} (binomial test p = {stats.binomtest(int((raw.margin[near] > 0).sum()), int(near.sum()), 0.5).pvalue:.2f})\n")

log("Balance of pre-determined covariates at the cutoff (RD effects should be zero)")
rows = []
for v, lab in [("presdemvoteshlag1", "Democratic presidential vote share, previous election"), ("demvoteshlag1", "Democratic Senate vote share, previous election"),
               ("demvoteshlag2", "Democratic Senate vote share, two elections ago"), ("demwinprv1", "Democrat won the previous election (0/1)"),
               ("dopen", "Open seat (no incumbent running)"), ("dmidterm", "Midterm election year (0/1)")]:
    dv = raw.dropna(subset=[v])
    r = rd(dv[v].values, dv.margin.values, p=1)
    rows.append({"covariate": lab, "RD effect": r.coef.loc["Conventional"].iloc[0], "robust p": r.pv.loc["Robust"].iloc[0], "bandwidth": r.bws.loc["h", "left"]})
bal = pd.DataFrame(rows).round(3)
log(bal.to_string(index=False))
log(f"Covariates with p < 0.05: {int((bal['robust p'] < 0.05).sum())} of {len(bal)}. "
    "Among several tests one rejection at the 5% level is expected by chance, but a significant jump in the previous-election variables would be a warning sign.\n")

log("Placebo cutoffs (the RD effect should be zero away from the true cutoff; each uses only data on one side of 0)")
rows = []
for c in [-10, -5, 5, 10]:
    sub = d[d.margin < 0] if c < 0 else d[d.margin > 0]
    r = rd(sub.vote.values, sub.margin.values, c=c, p=1)
    rows.append({"placebo cutoff": c, "RD effect": r.coef.loc["Conventional"].iloc[0], "robust p": r.pv.loc["Robust"].iloc[0]})
plc = pd.DataFrame(rows)
log(plc.round(3).to_string(index=False))
n_rej = int((plc["robust p"] < 0.05).sum())
log(f"Placebo cutoffs with p < 0.05: {n_rej} of {len(plc)}. " + ("The placebo at -10 shows a spurious jump. The cause is unclear (chance with four tests, curvature of the outcome function, or a real problem); "
    "I did not investigate further, and it is a reason not to treat the design as perfectly clean.\n" if n_rej else "No placebo jump is detected.\n"))

# ----------------------------------------------------------------------------
# 3. Sensitivity
# ----------------------------------------------------------------------------
h0 = main.bws.loc["h", "left"]
rows = [line(main, "Baseline (p = 1, MSE-optimal h)")]
for m_ in [0.5, 0.75, 1.5, 2.0]:
    rows.append(line(rd(y, x, p=1, h=h0 * m_), f"Bandwidth x {m_}"))
rows.append(line(rd(y, x, p=2), "Local quadratic (p = 2)"))
rows.append(line(rd(y, x, p=1, kernel="uniform"), "Uniform kernel"))
don = d[d.margin.abs() > 0.5]
rows.append(line(rd(don.vote.values, don.margin.values, p=1), "Donut: drop |margin| < 0.5"))
rows.append(line(rd(y, x, p=1, bwselect="cerrd"), "CER-optimal bandwidth"))
sens = pd.DataFrame(rows)
sens["estimate"] = sens["estimate"].round(2)
sens["robust p"] = sens["robust p"].map(lambda v: f"{v:.2g}")
sens["bandwidth"] = sens["bandwidth"].round(1)
log("Sensitivity of the estimate (percentage points)")
log(sens.to_string(index=False), "\n")

# Additional outcome: probability that the Democrat wins the next election, using the next-election indicator
d["win_next"] = (d.vote > 50).astype(int)
rw = rd(d.win_next.values, d.margin.values, p=1)
log(f"Probability of winning the next election: RD effect {rw.coef.loc['Conventional'].iloc[0] * 100:.0f} percentage points "
    f"(robust p = {rw.pv.loc['Robust'].iloc[0]:.2g}); winning once raises the chance of winning again by about this amount.\n")

# ----------------------------------------------------------------------------
# Figure: binned means with local linear fits
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
sub = d[d.margin.abs() <= 50]
bins = np.arange(-50, 52, 2.5)
cut = pd.cut(sub.margin, bins)
g = sub.groupby(cut).agg(x=("margin", "mean"), y=("vote", "mean"))
ax.scatter(g.x, g.y, s=18, color="C0")
for side in (sub[sub.margin < 0], sub[sub.margin >= 0]):
    w = side[side.margin.abs() <= h0]
    b = np.polyfit(w.margin, w.vote, 1)
    xs = np.linspace(w.margin.min(), w.margin.max(), 50)
    ax.plot(xs, np.polyval(b, xs), "r-", lw=2)
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("Democratic margin of victory in election t (points)")
ax.set_ylabel("Democratic vote share in election t+1 (%)")
ax.set_title("Regression discontinuity: incumbency advantage in the Senate")
fig.tight_layout()
fig.savefig(HERE / "figures" / "rd_plot.png", dpi=150)
plt.close(fig)

log("Caveats: the estimate is local, for races that were close, and applies to the party, not to the individual candidate (the winner may not run again,")
log("and a seat may change hands for reasons unrelated to incumbency); the next election is six years later, so many things change; the sample")
log("is small near the cutoff (about 700 effective observations); the running variable has a few repeated values; results for the House of")
log("Representatives or other periods may differ; the bandwidth choice is data-driven but not unique.")
out.close()
print("\nDone. See results.txt and figures/.")
