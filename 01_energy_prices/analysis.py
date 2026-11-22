"""
Project 1 - Energy price dynamics: Brent, WTI, Henry Hub gas, jet fuel (daily, 2010-2026)

Questions
  1. Are the price series non-stationary, and are returns stationary?
  2. Do crude oil benchmarks (Brent, WTI) and jet fuel share a long-run equilibrium (cointegration)?
  3. How do short-run shocks transmit between markets (VAR, Granger causality, impulse responses)?

Run:  python analysis.py      (needs pandas numpy statsmodels matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# 1. Load and clean
# ----------------------------------------------------------------------------
brent = pd.read_csv(HERE.parent / "data" / "brent.csv", parse_dates=["date"])
energy = pd.read_csv(HERE.parent / "data" / "energy.csv", parse_dates=["date"])
df = brent.merge(energy, on="date").set_index("date").sort_index()
log(f"Sample: {df.index.min().date()} to {df.index.max().date()}, {len(df)} rows")
log("Missing values per column:\n", df.isna().sum().to_string(), "\n")

# WTI was negative on 2020-04-20 (storage crisis). Logs are undefined there,
# so I treat non-positive prices as missing. This is a documented modelling choice.
n_bad = int((df <= 0).sum().sum())
df = df.where(df > 0)
log(f"Non-positive prices set to missing: {n_bad} (WTI, 2020-04-20)\n")

# Missing days are mostly public holidays that differ by market. Use only days
# where all four series are observed so that returns are comparable across markets.
df = df.dropna()
log(f"Rows after keeping only days with all four prices: {len(df)}\n")

logp = np.log(df)
ret = logp.diff().dropna() * 100  # daily log returns in percent

# ----------------------------------------------------------------------------
# 2. Stationarity (ADF test)
# ----------------------------------------------------------------------------
def adf_table(frame, label):
    rows = []
    for c in frame.columns:
        stat, p, *_ = adfuller(frame[c], autolag="AIC")
        rows.append({"series": c, "ADF stat": round(stat, 2), "p-value": round(p, 4)})
    log(f"ADF test, {label} (H0: unit root)\n", pd.DataFrame(rows).to_string(index=False), "\n")


adf_table(logp, "log prices")
adf_table(ret, "log returns")

# ----------------------------------------------------------------------------
# 3. Cointegration of Brent, WTI, jet fuel (Johansen trace test)
# ----------------------------------------------------------------------------
oil = logp[["BRENT", "WTI", "JETFUEL"]]
rank_sel = select_coint_rank(oil, det_order=0, k_ar_diff=5, method="trace", signif=0.05)
log("Johansen trace test, Brent / WTI / jet fuel (5 lagged differences, constant)")
log(rank_sel.summary(), "\n")
log(f"Selected cointegration rank at 5%: {rank_sel.rank}\n")

jo = coint_johansen(oil, det_order=0, k_ar_diff=5)
beta = pd.Series(np.real(jo.evec[:, 0] / jo.evec[0, 0]), index=oil.columns)
log("First cointegrating vector (normalised on Brent):\n", beta.round(3).to_string(), "\n")


# Robustness: a full-rank result (rank 3 of 3) would mean all log prices are
# stationary, which clashes with the borderline ADF results above. Structural
# breaks (2014 price collapse, 2020, 2022) can cause this. So check pairwise
# Engle-Granger tests and two subperiods instead of relying on one result.
log("Engle-Granger cointegration tests (H0: no cointegration)")
rows = []
for a, b in [("BRENT", "WTI"), ("BRENT", "JETFUEL"), ("WTI", "JETFUEL"), ("BRENT", "HH")]:
    stat, p, _ = coint(logp[a], logp[b])
    rows.append({"pair": f"{a} ~ {b}", "EG stat": round(stat, 2), "p-value": round(p, 4)})
log(pd.DataFrame(rows).to_string(index=False), "\n")

log("Johansen trace test by subperiod (Brent / WTI / jet fuel): selected rank")
for label, sl in [("2010-2019", slice("2010", "2019")), ("2020-2026", slice("2020", "2026"))]:
    r = select_coint_rank(oil.loc[sl], det_order=0, k_ar_diff=5, method="trace", signif=0.05).rank
    log(f"  {label}: rank {r}")
log("")

# ----------------------------------------------------------------------------
# 3c. Vector error-correction model (VECM) for the cointegrated block Brent / WTI / jet fuel
#     Rank 2 is used because the two subperiods both select rank 2 (the full-sample rank 3 is doubtful, see above).
# ----------------------------------------------------------------------------
vecm = VECM(oil, k_ar_diff=5, coint_rank=2, deterministic="co").fit()
log("VECM(5 lagged differences, rank 2) on log Brent, WTI, jet fuel")
log("Cointegrating vectors beta (columns = vectors; statsmodels normalises the first two rows to the identity):")
log(pd.DataFrame(vecm.beta, index=oil.columns, columns=["vector 1", "vector 2"]).round(3).to_string())
log("Adjustment coefficients alpha (rows = equations) with t-values:")
alpha = pd.DataFrame(vecm.alpha, index=oil.columns, columns=["vector 1", "vector 2"])
tval = pd.DataFrame(vecm.tvalues_alpha, index=oil.columns, columns=["t (vector 1)", "t (vector 2)"])
log(pd.concat([alpha, tval], axis=1).round(3).to_string())
log("Reading: a significant negative alpha on a vector means that equation reacts by moving back toward equilibrium.\n")


spread = oil @ beta
fig, ax = plt.subplots(figsize=(10, 3.5))
spread.plot(ax=ax, lw=0.8)
ax.axhline(spread.mean(), color="k", ls="--", lw=0.8)
ax.set_title("Estimated long-run equilibrium error (first Johansen vector)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "equilibrium_error.png", dpi=150)
plt.close(fig)

# Price levels figure
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for ax, c in zip(axes.ravel(), df.columns):
    df[c].plot(ax=ax, lw=0.8)
    ax.set_title(c)
fig.tight_layout()
fig.savefig(HERE / "figures" / "price_levels.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4. VAR on returns: lag choice, Granger causality, impulse responses
# ----------------------------------------------------------------------------
model = VAR(ret)
sel = model.select_order(maxlags=10)
log("VAR lag selection\n", sel.summary(), "\n")
p_bic = max(1, sel.selected_orders["bic"])
res = model.fit(p_bic)
log(f"Fitted VAR({p_bic}) on returns (lag chosen by BIC)\n")

log("Granger causality (H0: row does NOT Granger-cause the other variables)")
rows = []
for c in ret.columns:
    t = res.test_causality(caused=[x for x in ret.columns if x != c], causing=[c], kind="f")
    rows.append({"cause": c, "F-stat": round(t.test_statistic, 2), "p-value": round(t.pvalue, 4)})
log(pd.DataFrame(rows).to_string(index=False), "\n")

irf = res.irf(10)
fig = irf.plot(orth=True, impulse="BRENT", figsize=(9, 8))
fig.suptitle("Orthogonalised impulse responses to a Brent shock (ordering: BRENT, WTI, HH, JETFUEL)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "irf_brent_shock.png", dpi=150)
plt.close(fig)

fevd = res.fevd(10)
log("Forecast error variance decomposition of jet fuel returns, 10-day horizon (%):")
log((pd.Series(fevd.decomp[ret.columns.get_loc("JETFUEL")][-1], index=ret.columns) * 100).round(1).to_string())
out.close()
print("\nDone. See results.txt and figures/.")
