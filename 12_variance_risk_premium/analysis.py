"""
Project 12 - The variance risk premium: does the VIX help to predict S&P 500 returns?

Variance risk premium (VRP) = implied variance (VIX^2) minus realised variance over the last 21 trading days,
both in annualised percent^2. A high VRP means options are expensive relative to recent realised risk.

Questions
  1. How large is the VRP on average and when is it negative?
  2. Do VRP (and the VIX level) predict S&P 500 returns at horizons of 1, 5, 21 and 63 trading days (in-sample)?
  3. Out of sample (expanding window, 2010 onwards), does a VRP predictor beat the historical-mean forecast?
     Campbell-Thompson out-of-sample R2 and the Clark-West test for nested models.

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
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


spx = pd.read_csv(HERE.parent / "data" / "stock_indices.csv", parse_dates=["date"]).set_index("date").sort_index()["SPX"]
vix = pd.read_csv(HERE.parent / "data" / "vix.csv", parse_dates=["date"]).set_index("date").sort_index()["vix"]
d = pd.concat([spx, vix], axis=1).dropna()
r = 100 * np.log(d.SPX).diff()  # daily log return, %
d["rv21"] = 252 * (r ** 2).rolling(21).mean()  # realised variance (annualised %^2), uses returns up to day t
d["iv"] = d.vix ** 2
d["vrp"] = (d.iv - d.rv21) / 100  # scaled for readable coefficients
d["vix_l"] = d.vix
d["ret21_past"] = r.rolling(21).sum()
log(f"Sample: {d.index.min().date()} to {d.index.max().date()}, {len(d)} days")
v = d.vrp.dropna() * 100
log(f"VRP (annualised %^2): mean {v.mean():.0f}, median {v.median():.0f}, share of days with negative VRP {(v < 0).mean() * 100:.0f}%")
log("Mean VIX {:.1f} vs mean realised vol (21d) {:.1f} (annualised %)\n".format(d.vix.mean(), np.sqrt(d.rv21).mean()))

fig, ax = plt.subplots(figsize=(10, 3.6))
(d.vrp * 100).rolling(21).mean().plot(ax=ax, lw=0.8)
ax.axhline(0, color="k", lw=0.6)
ax.set_title("Variance risk premium, VIX^2 minus 21-day realised variance (21-day average, annualised %^2)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "vrp.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# Forward returns (the target is the sum of daily log returns over days t+1 .. t+h)
# ----------------------------------------------------------------------------
HORIZONS = [1, 5, 21, 63]
for h in HORIZONS:
    d[f"y{h}"] = r[::-1].rolling(h).sum()[::-1].shift(-1)  # sum of r over t+1..t+h

# ----------------------------------------------------------------------------
# In-sample predictive regressions with HAC standard errors (overlap = h-1 lags, use 1.5h + 5 to be safe)
# ----------------------------------------------------------------------------
log("In-sample predictive regressions: future h-day return (%) on VRP (units of 100 %^2) , whole sample")
rows = []
for h in HORIZONS:
    dd = d[[f"y{h}", "vrp", "vix_l", "ret21_past"]].dropna()
    for name, cols in [("VRP", ["vrp"]), ("VIX level", ["vix_l"]), ("VRP + VIX + past return", ["vrp", "vix_l", "ret21_past"])]:
        m = sm.OLS(dd[f"y{h}"], sm.add_constant(dd[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": int(1.5 * h) + 5})
        c0 = cols[0]
        rows.append({"h": h, "predictors": name, "first predictor": c0, "coef": round(m.params[c0], 3), "t (HAC)": round(m.tvalues[c0], 2), "R2 %": round(m.rsquared * 100, 2)})
res = pd.DataFrame(rows)
log(res.to_string(index=False), "\n")

# ----------------------------------------------------------------------------
# Out-of-sample: expanding window, first forecast 2010-01-01, forecasts every 5 days.
# At forecast date t, the model is trained only on observations whose target is already known (target end <= t).
# ----------------------------------------------------------------------------
def clark_west(y, f_mean, f_model, lags):
    adj = (y - f_mean) ** 2 - ((y - f_model) ** 2 - (f_mean - f_model) ** 2)
    m = sm.OLS(adj, np.ones(len(adj))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.tvalues[0]), float(1 - stats.norm.cdf(m.tvalues[0]))  # one-sided: model better than mean


log("Out-of-sample forecasts 2010-01 to the end of the sample, expanding window (R2_OS > 0: model beats the historical mean)")
oos_rows = []
dates = d.index
start = np.searchsorted(dates, pd.Timestamp("2010-01-01"))
pos = {t: i for i, t in enumerate(dates)}
for h in HORIZONS:
    for name, cols in [("VRP", ["vrp"]), ("VIX level", ["vix_l"])]:
        ys, fm, fx, ts = [], [], [], []
        data = d[[f"y{h}"] + cols].copy()
        for i in range(start, len(dates) - h, 5):
            train_idx = slice(0, i - h + 1)  # targets ending at or before day i
            tr = data.iloc[train_idx].dropna()
            if len(tr) < 500:
                continue
            X = np.column_stack([np.ones(len(tr)), tr[cols].values])
            beta = np.linalg.lstsq(X, tr[f"y{h}"].values, rcond=None)[0]
            xt = data.iloc[i][cols].values
            if np.isnan(xt).any() or np.isnan(data.iloc[i][f"y{h}"]):
                continue
            fx.append(beta[0] + xt @ beta[1:])
            fm.append(tr[f"y{h}"].mean())
            ys.append(data.iloc[i][f"y{h}"])
            ts.append(dates[i])
        ys, fm, fx = map(np.array, (ys, fm, fx))
        r2 = 1 - ((ys - fx) ** 2).sum() / ((ys - fm) ** 2).sum()
        cw_t, cw_p = clark_west(ys, fm, fx, lags=max(1, int(h / 5)) + 2)
        oos_rows.append({"h": h, "predictor": name, "n forecasts": len(ys), "R2_OS %": round(r2 * 100, 2), "Clark-West t": round(cw_t, 2), "p (one-sided)": round(cw_p, 3)})
log(pd.DataFrame(oos_rows).to_string(index=False), "\n")

log("Caveats: overlapping multi-day returns make the t-statistics and the Clark-West test only approximately valid (HAC corrections")
log("are used but the samples are short for h = 63); forecasts are made every fifth day; the sample ends in August 2019 and contains few")
log("crises; the VRP uses a backward-looking realised variance, which is only one of several possible definitions; transaction costs and")
log("any trading rule are not considered. Predictability results in this literature are often fragile.")
out.close()
print("\nDone. See results.txt and figures/.")
