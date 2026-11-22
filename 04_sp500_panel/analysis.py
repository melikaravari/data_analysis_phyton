"""
Project 4 - S&P 500 stock panel: cleaning, factor structure (PCA) and minimum-variance portfolio

Questions
  1. How should a messy wide price panel (700 columns, many missing values, preferred shares, warrants)
     be cleaned before computing returns?
  2. How much of the co-movement of stock returns is explained by a few common factors (PCA)?
     Is the first factor the market?
  3. Does a minimum-variance portfolio (shrinkage covariance estimate) have lower out-of-sample
     risk than an equally weighted portfolio, including through the COVID crash of 2020?

Run:  python analysis.py      (needs pandas numpy scikit-learn scipy matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# 1. Cleaning
# ----------------------------------------------------------------------------
raw = pd.read_csv(HERE.parent / "data" / "sp500_prices.csv", parse_dates=["datadate"]).set_index("datadate").sort_index()
log(f"Raw panel: {raw.shape[0]} days x {raw.shape[1]} tickers, {raw.index.min().date()} to {raw.index.max().date()}")
log(f"Share of missing cells: {raw.isna().mean().mean() * 100:.1f}%")

# Step 1: tickers containing '.' are mostly preferred shares, warrants or rights (e.g. BAC.PA, AIG.WS),
# which trade near par and are not comparable to common stock. Dropped.
common = [c for c in raw.columns if "." not in c]
log(f"Step 1: dropped {raw.shape[1] - len(common)} tickers with '.' in the name (preferred shares, warrants, share classes)")
px = raw[common]

# Step 2: keep tickers observed on at least 95% of days
cov_ok = px.notna().mean() >= 0.95
log(f"Step 2: dropped {int((~cov_ok).sum())} tickers with more than 5% missing prices")
px = px.loc[:, cov_ok]

# Step 3: fill short gaps (max 3 days) forward, then drop tickers that still contain gaps
px = px.ffill(limit=3)
still_missing = px.columns[px.isna().any()]
px = px.drop(columns=still_missing)
log(f"Step 3: forward-filled gaps up to 3 days; dropped {len(still_missing)} tickers with remaining gaps")

ret = np.log(px).diff().dropna()
log(f"Clean panel: {px.shape[1]} tickers, {len(ret)} daily returns")

# Step 4: sanity check for implausible returns (data errors, splits not adjusted)
extreme = ret.abs().stack()
extreme = extreme[extreme > 0.5]
log(f"Step 4: returns larger than 50% in absolute value: {len(extreme)}")
if len(extreme):
    log(extreme.sort_values(ascending=False).head(5).round(2).to_string())
log("")

# ----------------------------------------------------------------------------
# 2. PCA: how many common factors?
# ----------------------------------------------------------------------------
z = (ret - ret.mean()) / ret.std()
pca = PCA(n_components=10).fit(z)
evr = pd.Series(pca.explained_variance_ratio_ * 100, index=[f"PC{i}" for i in range(1, 11)])
log("Share of total return variance explained (%), first 10 principal components")
log(evr.round(1).to_string())
log(f"Cumulative, first 5: {evr[:5].sum():.1f}%\n")

pc1 = pd.Series(pca.transform(z)[:, 0], index=z.index)
ew_mkt = ret.mean(axis=1)
corr = np.corrcoef(pc1, ew_mkt)[0, 1]
log(f"Correlation of PC1 with the equally weighted market return: {abs(corr):.3f}  (sign of a PC is arbitrary)")
log("-> PC1 is the market factor.\n")

# Excluding the crash: is the factor structure stable?
calm = z.loc[:"2020-02-18"]
evr_calm = PCA(n_components=1).fit(calm).explained_variance_ratio_[0] * 100
evr_crash = PCA(n_components=1).fit(z.loc["2020-02-19":"2020-04-30"]).explained_variance_ratio_[0] * 100
log(f"PC1 share before the COVID crash (to 2020-02-18): {evr_calm:.1f}%")
log(f"PC1 share during the crash (2020-02-19 to 2020-04-30): {evr_crash:.1f}%  -> correlations rise in crises\n")

fig, ax = plt.subplots(figsize=(6, 3.5))
evr.plot.bar(ax=ax)
ax.set_ylabel("% of variance")
ax.set_title("Scree plot: PCA of daily stock returns")
fig.tight_layout()
fig.savefig(HERE / "figures" / "scree_plot.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. Minimum-variance portfolio vs equal weight (train on 2019, test afterwards)
# ----------------------------------------------------------------------------
train = ret.loc[:"2019-12-31"]
test = ret.loc["2020-01-01":]
log(f"Estimation window: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)")
log(f"Evaluation window: {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)\n")


def min_var_weights(returns, cap=0.05):
    """Long-only minimum-variance weights with a 5% cap per stock, Ledoit-Wolf shrinkage covariance."""
    S = LedoitWolf().fit(returns.values).covariance_
    n = S.shape[0]
    res = minimize(
        lambda w: w @ S @ w, np.full(n, 1 / n), jac=lambda w: 2 * S @ w, method="SLSQP",
        bounds=[(0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    return pd.Series(res.x, index=returns.columns)


w_mv = min_var_weights(train)
w_ew = pd.Series(1 / ret.shape[1], index=ret.columns)
log(f"Min-variance portfolio holds {int((w_mv > 1e-4).sum())} stocks with weight > 0.01%; largest weight {w_mv.max():.1%}")

# Static weights, no rebalancing: simple but a reasonable first benchmark.
simple = pd.DataFrame({"Equal weight": test @ w_ew, "Min variance": test @ w_mv})


def perf(r):
    cum = np.exp(r.cumsum())
    dd = (cum / cum.cummax() - 1).min()
    return pd.Series({
        "Annualised return %": r.mean() * 252 * 100,
        "Annualised volatility %": r.std() * np.sqrt(252) * 100,
        "Return/vol (rf = 0)": r.mean() / r.std() * np.sqrt(252),
        "Max drawdown %": dd * 100,
        "Worst day %": (np.exp(r.min()) - 1) * 100,
    })


tab = simple.apply(perf).round(2)
log("Out-of-sample performance")
log(tab.to_string(), "\n")

fig, ax = plt.subplots(figsize=(9, 3.8))
np.exp(simple.cumsum()).plot(ax=ax)
ax.set_title("Out-of-sample cumulative growth of 1 (log returns, static weights)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "cumulative_performance.png", dpi=150)
plt.close(fig)

log("Caveats: (1) the panel contains only stocks that still exist in the data (survivorship bias);")
log("(2) the test window is dominated by one extreme event; (3) no transaction costs or rebalancing.")
out.close()
print("\nDone. See results.txt and figures/.")
