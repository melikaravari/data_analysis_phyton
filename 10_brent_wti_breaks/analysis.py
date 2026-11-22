"""
Project 10 - The Brent-WTI spread: structural breaks and a vector error-correction model (VECM)

Questions
  1. Is the (log) Brent-WTI spread stationary, and does a structural break change that conclusion?
  2. When is the break in the level of the spread? Does it match known events (US shale boom, 2015 lifting of
     the US crude export ban)?
  3. Which price does the adjustment back to equilibrium? Does the VECM change across the break?

Run:  python analysis.py      (needs pandas numpy scipy statsmodels matplotlib)
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
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import adfuller, zivot_andrews
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


b = pd.read_csv(HERE.parent / "data" / "brent.csv", parse_dates=["date"])
e = pd.read_csv(HERE.parent / "data" / "energy.csv", parse_dates=["date"])
df = b.merge(e, on="date").set_index("date").sort_index()[["BRENT", "WTI"]]
df = df.where(df > 0).dropna()  # WTI was negative on 2020-04-20; logs are undefined
lp = np.log(df)
spread = (lp.BRENT - lp.WTI) * 100  # percent
log(f"Sample: {df.index.min().date()} to {df.index.max().date()}, {len(df)} days (days with both prices; WTI <= 0 day removed)")
log("Spread = 100 x (log Brent - log WTI)\n")
log("Mean spread by year (%): " + ", ".join(f"{y}: {v:.1f}" for y, v in spread.groupby(spread.index.year).mean().items()) + "\n")

# ----------------------------------------------------------------------------
# 1. Stationarity of the spread, with and without allowing for a break
# ----------------------------------------------------------------------------
adf = adfuller(spread, autolag="AIC")
log(f"ADF on the spread (H0 unit root): stat {adf[0]:.2f}, p = {adf[1]:.3f}")
za = zivot_andrews(spread, regression="c", autolag="AIC")
za_date = spread.index[int(za[4])]
log(f"Zivot-Andrews (one break in the mean allowed): stat {za[0]:.2f}, p = {za[1]:.3f}, estimated break {za_date.date()}")
log("(Zivot-Andrews rejection = stationary around a mean that shifts at the break date.)\n")

# ----------------------------------------------------------------------------
# 2. Structural break in the level of the spread: scan of Chow-type F statistics, plus OLS-CUSUM test
# ----------------------------------------------------------------------------
y = spread.values
n = len(y)
lo, hi = int(0.15 * n), int(0.85 * n)
ssr_full = ((y - y.mean()) ** 2).sum()
F = np.full(n, np.nan)
for t in range(lo, hi):
    ssr = ((y[:t] - y[:t].mean()) ** 2).sum() + ((y[t:] - y[t:].mean()) ** 2).sum()
    F[t] = (ssr_full - ssr) / (ssr / (n - 2))
tb = int(np.nanargmax(F))
break_date = spread.index[tb]
log(f"Largest Chow F for a shift in the mean of the spread: {np.nanmax(F):.0f} at {break_date.date()} (15% trimming)")
log(f"  mean spread before: {y[:tb].mean():.1f}%, after: {y[tb:].mean():.1f}%")
cs = breaks_cusumolsresid(sm.OLS(y, np.ones(n)).fit().resid, ddof=1)
log(f"OLS-CUSUM test of parameter stability in the mean: stat {cs[0]:.2f}, p = {cs[1]:.4f}")
log("(The F scan only locates the most likely single break; its p-value is not standard, so I use it for the date, not for inference.)\n")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
spread.plot(ax=axes[0], lw=0.6)
axes[0].axvline(break_date, color="r", ls="--", lw=1, label=f"estimated break {break_date.date()}")
axes[0].axvline(pd.Timestamp("2015-12-18"), color="grey", ls=":", lw=1, label="US crude export ban lifted (2015-12-18)")
axes[0].legend(fontsize=8)
axes[0].set_title("Log Brent-WTI spread (%)")
pd.Series(F, index=spread.index).plot(ax=axes[1], lw=0.8)
axes[1].set_title("Chow F statistic for a break in the mean at each date")
fig.tight_layout()
fig.savefig(HERE / "figures" / "spread_and_break.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. VECM: log Brent and log WTI, full sample and the two sub-samples around the break
# ----------------------------------------------------------------------------
def vecm_summary(data, label):
    k = select_order(data, maxlags=10, deterministic="co").aic
    k = max(1, k)
    rank = select_coint_rank(data, det_order=0, k_ar_diff=k, method="trace", signif=0.05).rank
    log(f"--- {label}: {len(data)} days, lag order (differences) chosen by AIC = {k}, Johansen trace rank at 5% = {rank}")
    if rank == 0:
        log("    no cointegration found at 5%; VECM not estimated\n")
        return
    m = VECM(data, k_ar_diff=k, coint_rank=1, deterministic="co").fit()
    beta = m.beta[:, 0] / m.beta[0, 0]
    alpha = m.alpha[:, 0] * m.beta[0, 0]
    se = m.stderr_alpha[:, 0] * abs(m.beta[0, 0])
    log(f"    cointegrating vector (Brent = 1): Brent {beta[0]:.3f}, WTI {beta[1]:.3f}")
    log(f"    adjustment alpha: Brent {alpha[0]:.4f} (t = {alpha[0] / se[0]:.1f}), WTI {alpha[1]:.4f} (t = {alpha[1] / se[1]:.1f})")
    log("    (negative alpha for the variable that is 'too high' when the equilibrium error is positive = it adjusts back)\n")


log("Vector error-correction models, estimated on log prices")
vecm_summary(lp, "Full sample")
vecm_summary(lp.loc[:break_date], f"Before {break_date.date()}")
vecm_summary(lp.loc[break_date:], f"From {break_date.date()}")

log("Caveats: a single mean break is searched for although the spread probably has several regimes (2011-14 discount, 2020, 2022);")
log("Johansen results are sensitive to lag choice and sample; the cointegration rank is estimated, not known; I did not test the")
log("cointegrating vector restriction (Brent = WTI, i.e. a pure spread) formally.")
out.close()
print("\nDone. See results.txt and figures/.")
