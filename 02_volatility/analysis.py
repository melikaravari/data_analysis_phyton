"""
Project 2 - Volatility modelling for US equity indices and the VIX

Questions
  1. Do GARCH-type models capture the volatility clustering and the leverage effect in S&P 500 returns?
  2. Does an asymmetric model (GJR-GARCH) with fat-tailed errors beat a plain GARCH(1,1)?
  3. How closely does model-implied volatility track the VIX?
  4. Out of sample: does GARCH forecast next-day volatility better than a simple EWMA benchmark?

Run:  python analysis.py      (needs pandas numpy arch matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# 1. Load and prepare
# ----------------------------------------------------------------------------
idx = pd.read_csv(HERE.parent / "data" / "stock_indices.csv", parse_dates=["date"]).set_index("date")
vix = pd.read_csv(HERE.parent / "data" / "vix.csv", parse_dates=["date"]).set_index("date")
df = idx.join(vix, how="inner").dropna().sort_index()
log(f"Sample: {df.index.min().date()} to {df.index.max().date()}, {len(df)} trading days")

ret = 100 * np.log(df[["SPX", "DJI", "NDX", "RUT"]]).diff().dropna()
r = ret["SPX"]

log("\nSPX daily return summary (%):")
log(r.describe().round(3).to_string())
log(f"Skewness {stats.skew(r):.2f}, excess kurtosis {stats.kurtosis(r):.2f} (normal = 0)")
lb = stats.pearsonr(r.abs()[1:], r.abs().shift(1).dropna())[0]
log(f"Autocorrelation of |returns| at lag 1: {lb:.2f}  -> volatility clustering\n")

# ----------------------------------------------------------------------------
# 2. In-sample model comparison
# ----------------------------------------------------------------------------
specs = {
    "GARCH(1,1) normal": dict(vol="GARCH", p=1, q=1, dist="normal"),
    "GARCH(1,1) Student-t": dict(vol="GARCH", p=1, q=1, dist="t"),
    "GJR-GARCH(1,1,1) Student-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
}
fits, rows = {}, []
for name, kw in specs.items():
    f = arch_model(r, mean="Constant", **kw).fit(disp="off")
    fits[name] = f
    rows.append({"model": name, "loglik": round(f.loglikelihood, 1), "AIC": round(f.aic, 1), "BIC": round(f.bic, 1)})
log("In-sample comparison (lower AIC/BIC is better)")
log(pd.DataFrame(rows).to_string(index=False), "\n")

best = fits["GJR-GARCH(1,1,1) Student-t"]
log("Parameters of GJR-GARCH(1,1,1) with Student-t errors")
log(best.params.round(4).to_string())
log(f"gamma (asymmetry) t-stat: {best.tvalues['gamma[1]']:.1f}  -> positive gamma = bad news raises volatility more than good news")
log(f"Persistence alpha + beta + gamma/2 = {best.params['alpha[1]'] + best.params['beta[1]'] + best.params['gamma[1]'] / 2:.3f}\n")

# ----------------------------------------------------------------------------
# 3. Model volatility vs VIX
# ----------------------------------------------------------------------------
ann_vol = best.conditional_volatility * np.sqrt(252)  # annualised, in %
cmp_ = pd.concat([ann_vol.rename("model"), df["vix"].rename("vix")], axis=1).dropna()
log(f"Correlation between model volatility and VIX (levels): {cmp_.corr().iloc[0, 1]:.2f}")
log(f"Mean model vol {cmp_['model'].mean():.1f}%  vs mean VIX {cmp_['vix'].mean():.1f}%  "
    "(VIX is typically higher: it contains a volatility risk premium)\n")

fig, ax = plt.subplots(figsize=(10, 4))
cmp_["model"].plot(ax=ax, lw=0.8, label="GJR-GARCH annualised vol")
cmp_["vix"].plot(ax=ax, lw=0.8, label="VIX", alpha=0.8)
ax.legend()
ax.set_title("Model-implied volatility vs VIX")
fig.tight_layout()
fig.savefig(HERE / "figures" / "garch_vs_vix.png", dpi=150)
plt.close(fig)

# News impact curve of the fitted model
z = np.linspace(-5, 5, 200)
p = best.params
news = p["omega"] + (p["alpha[1]"] + p["gamma[1]"] * (z < 0)) * z**2
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(z, news)
ax.set_xlabel("yesterday's shock (%)")
ax.set_ylabel("next-day variance contribution")
ax.set_title("News impact curve (asymmetric)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "news_impact_curve.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4. Out-of-sample forecast: parameters fitted up to 2017, then rolling 1-day-ahead
# ----------------------------------------------------------------------------
split = pd.Timestamp("2018-01-01")
train = r[r.index < split]
test = r[r.index >= split]
fit_tr = arch_model(train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
# Re-use the training parameters on the full sample, so test-period forecasts only use past data
mod_all = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
res_all = mod_all.fix(fit_tr.params)
h_garch = res_all.conditional_volatility ** 2  # variance for day t uses info up to t-1

# Benchmarks: RiskMetrics EWMA (lambda = 0.94) and a plain 250-day rolling variance
lam = 0.94
ew = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean().shift(1)
roll = (r ** 2).rolling(250).mean().shift(1)

proxy = (r - train.mean()) ** 2  # squared return as a noisy proxy for true variance


def qlike(forecast, proxy_):
    return float(np.mean(np.log(forecast) + proxy_ / forecast))


rows = []
for name, h in [("GJR-GARCH-t (fixed params)", h_garch), ("EWMA 0.94", ew), ("Rolling 250d variance", roll)]:
    d = pd.concat([h.rename("h"), proxy.rename("p")], axis=1).loc[test.index].dropna()
    rows.append({
        "forecast": name,
        "MSE": round(float(np.mean((d.h - d.p) ** 2)), 2),
        "QLIKE": round(qlike(d.h, d.p), 4),
    })
log(f"Out-of-sample 1-day variance forecasts, {test.index.min().date()} to {test.index.max().date()} (lower is better)")
log(pd.DataFrame(rows).to_string(index=False))
log("Note: squared returns are a noisy proxy for true variance; QLIKE is robust to that noise, MSE is dominated by crisis days.")
out.close()
print("\nDone. See results.txt and figures/.")
