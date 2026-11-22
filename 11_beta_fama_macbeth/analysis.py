"""
Project 11 - Market betas, their stability, and a Fama-MacBeth test of whether beta was rewarded in 2020

Questions
  1. How stable are CAPM betas between a calm year (2019) and the COVID year (2020)?
  2. Did high-beta stocks lose more in the crash and gain more in the rebound (portfolio sorts on 2019 betas)?
  3. Fama-MacBeth: was beta priced in the cross-section of daily returns in 2020?

Data: daily prices of S&P 500-type stocks (see project 04 for the cleaning steps), 2019-01-02 to 2020-11-12.
The market return is proxied by the equally weighted average of the cleaned stocks (no market caps in the file).

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

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# 1. Cleaning (same rules as project 04) and returns
# ----------------------------------------------------------------------------
raw = pd.read_csv(HERE.parent / "data" / "sp500_prices.csv", parse_dates=["datadate"]).set_index("datadate").sort_index()
px = raw[[c for c in raw.columns if "." not in c]]
px = px.loc[:, px.notna().mean() >= 0.95].ffill(limit=3)
px = px.dropna(axis=1)
ret = 100 * np.log(px).diff().dropna()
ret = ret.loc[:, ret.abs().max() < 60]  # remove one non-standard instrument with implausible returns
mkt = ret.mean(axis=1).rename("mkt")
log(f"Clean panel: {ret.shape[1]} stocks, {len(ret)} daily returns, {ret.index.min().date()} to {ret.index.max().date()}")

est = ret.loc[:"2019-12-31"]
tst = ret.loc["2020-01-01":]
log(f"Estimation (2019): {len(est)} days; evaluation (2020): {len(tst)} days\n")


def betas(r, m):
    m_c = m - m.mean()
    return (r - r.mean()).mul(m_c, axis=0).sum() / (m_c ** 2).sum()


b19 = betas(est, mkt.loc[est.index])
b20 = betas(tst, mkt.loc[tst.index])

# ----------------------------------------------------------------------------
# 2. Beta stability
# ----------------------------------------------------------------------------
fit = sm.OLS(b20, sm.add_constant(b19)).fit()
log("Beta stability, 2019 vs 2020 (OLS betas on the equally weighted market)")
log(f"  mean beta 2019: {b19.mean():.2f} (std {b19.std():.2f}); mean beta 2020: {b20.mean():.2f} (std {b20.std():.2f})")
log(f"  correlation between 2019 and 2020 betas: {np.corrcoef(b19, b20)[0, 1]:.2f}; R2 of beta2020 on beta2019: {fit.rsquared:.2f}")
log(f"  slope {fit.params.iloc[1]:.2f} (a slope below 1 means betas regress toward the cross-sectional mean)")
log(f"  share of stocks whose beta moved by more than 0.3: {(abs(b20 - b19) > 0.3).mean() * 100:.0f}%\n")

# ----------------------------------------------------------------------------
# 3. Portfolio sorts on 2019 beta
# ----------------------------------------------------------------------------
q = pd.qcut(b19, 5, labels=[f"Q{i}" for i in range(1, 6)])  # Q1 = lowest beta
windows = {
    "Crash (2020-02-19 to 2020-03-23)": ("2020-02-19", "2020-03-23"),
    "Rebound (2020-03-24 to 2020-06-08)": ("2020-03-24", "2020-06-08"),
    "Rest of 2020 (after 2020-06-08)": ("2020-06-09", "2020-11-12"),
    "Whole of 2020": ("2020-01-01", "2020-11-12"),
}
rows = {}
for name, (a, b) in windows.items():
    rr = tst.loc[a:b]
    port = pd.DataFrame({k: rr.loc[:, q == k].mean(axis=1) for k in q.cat.categories})  # equally weighted, daily rebalanced
    rows[name] = (np.exp(port.sum() / 100) - 1) * 100
    rows[name].loc["Market (EW)"] = (np.exp(mkt.loc[a:b].sum() / 100) - 1) * 100
tab = pd.DataFrame(rows).T.round(1)
log("Cumulative return (%) of equally weighted quintile portfolios formed on 2019 beta (Q1 = lowest beta, Q5 = highest)")
log(tab.to_string(), "\n")

# ----------------------------------------------------------------------------
# 4. Fama-MacBeth on daily returns in 2020: r_i,t = g0_t + g1_t * beta_i(2019) + e_i,t
# ----------------------------------------------------------------------------
def fama_macbeth(r, beta, lags=5):
    X = sm.add_constant(beta.rename("beta"))
    g = pd.DataFrame([sm.OLS(r.loc[t], X).fit().params for t in r.index], index=r.index)
    res = {}
    for c in g.columns:
        s = sm.OLS(g[c].values, np.ones(len(g))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        res[c] = (g[c].mean(), float(s.tvalues[0]))
    return res, g


log("Fama-MacBeth: average daily slope on 2019 beta (% per day per unit of beta), Newey-West t-statistics (5 lags)")
rows = []
for name, (a, b) in [("Whole of 2020", ("2020-01-01", "2020-11-12")), ("Crash", windows["Crash (2020-02-19 to 2020-03-23)"]),
                     ("Rebound", windows["Rebound (2020-03-24 to 2020-06-08)"]), ("Rest of 2020", windows["Rest of 2020 (after 2020-06-08)"])]:
    res, g = fama_macbeth(tst.loc[a:b], b19)
    rows.append({"period": name, "days": len(g), "avg slope on beta": round(res["beta"][0], 3), "t-stat": round(res["beta"][1], 2),
                 "avg market return, %/day": round(mkt.loc[a:b].mean(), 3)})
log(pd.DataFrame(rows).to_string(index=False))
log("The slope on beta should track the market's return if beta is rewarded: positive when the market rises, negative when it falls.\n")

# Figures
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(b19, b20, s=6, alpha=0.5)
lim = [min(b19.min(), b20.min()), max(b19.max(), b20.max())]
axes[0].plot(lim, lim, "k--", lw=0.8, label="45-degree line")
axes[0].set_xlabel("beta 2019")
axes[0].set_ylabel("beta 2020")
axes[0].legend()
axes[0].set_title("Beta stability")
tab.drop(columns="Market (EW)").loc[list(windows)[:3]].T.plot.bar(ax=axes[1])
axes[1].set_title("Return of beta quintiles by 2020 phase (%)")
axes[1].set_xlabel("quintile (Q1 = low beta)")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "figures" / "beta_stability_and_sorts.png", dpi=150)
plt.close(fig)

log("Caveats: the market proxy is equally weighted, so betas differ from betas against a value-weighted index; the sample contains only")
log("surviving stocks and one extreme episode; the Fama-MacBeth sample is 2020 only (about 220 days), so the average slope over the")
log("whole year is a noisy estimate; no size, value or momentum controls; daily portfolios are rebalanced each day (a simplification).")
out.close()
print("\nDone. See results.txt and figures/.")
