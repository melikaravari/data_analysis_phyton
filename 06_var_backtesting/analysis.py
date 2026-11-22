"""
Project 6 - Value-at-Risk and Expected Shortfall: model comparison and statistical backtesting

Questions
  1. Which method gives 1-day VaR forecasts with the right number of violations?
     (historical simulation, normal with EWMA volatility, GJR-GARCH with Student-t errors)
  2. Are violations independent over time, or do they cluster?
  3. Is the Expected Shortfall of the best model consistent with the realised losses on violation days?

Backtests: Kupiec proportion-of-failures test (unconditional coverage), Christoffersen independence test,
and the combined conditional coverage test.

Run:  python analysis.py      (needs pandas numpy scipy arch matplotlib)
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


idx = pd.read_csv(HERE.parent / "data" / "stock_indices.csv", parse_dates=["date"]).set_index("date").sort_index()
rets = 100 * np.log(idx).diff().dropna()
log(f"Sample: {rets.index.min().date()} to {rets.index.max().date()}, {len(rets)} daily returns, indices: {list(rets.columns)}")

WINDOW = 500  # estimation window / minimum history (days)
REFIT = 250  # GARCH parameters re-estimated every 250 trading days (expanding window)
LEVELS = [0.01, 0.025]  # 99% and 97.5% VaR


# ----------------------------------------------------------------------------
# Backtest statistics
# ----------------------------------------------------------------------------
def kupiec(hits, p):
    n, x = len(hits), int(hits.sum())
    if x in (0, n):
        return np.nan
    pi = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi)
    return 1 - stats.chi2.cdf(-2 * (ll0 - ll1), 1)


def christoffersen_ind(hits):
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = ((a == 0) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum()
    n10, n11 = ((a == 1) & (b == 0)).sum(), ((a == 1) & (b == 1)).sum()
    if n01 + n11 == 0 or n01 == 0 or n11 == 0 or n10 == 0:
        return np.nan
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll1 = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    return 1 - stats.chi2.cdf(-2 * (ll0 - ll1), 1)


def cc_test(hits, p):
    """Conditional coverage = Kupiec + independence; chi-square with 2 d.f."""
    n, x = len(hits), int(hits.sum())
    pu, pi = kupiec(hits, p), christoffersen_ind(hits)
    if np.isnan(pu) or np.isnan(pi):
        return np.nan
    lr = stats.chi2.isf(pu, 1) + stats.chi2.isf(pi, 1)
    return 1 - stats.chi2.cdf(lr, 2)


def t_quantile_std(a, nu):
    """Quantile of a Student-t rescaled to unit variance."""
    return stats.t.ppf(a, nu) * np.sqrt((nu - 2) / nu)


def t_es_std(a, nu):
    """Expected shortfall (as a negative number) of the unit-variance Student-t at level a."""
    q = stats.t.ppf(a, nu)
    es = -(stats.t.pdf(q, nu) / a) * (nu + q ** 2) / (nu - 1)
    return es * np.sqrt((nu - 2) / nu)


# ----------------------------------------------------------------------------
# Forecasts
# ----------------------------------------------------------------------------
def forecasts(r):
    """Return dict of DataFrames with VaR forecasts per method/level, and ES forecasts of the GARCH model."""
    r = r.copy()
    n = len(r)
    f = {(m, a): pd.Series(np.nan, index=r.index) for m in ["Historical", "Normal-EWMA", "GJR-GARCH-t"] for a in LEVELS}
    es = {a: pd.Series(np.nan, index=r.index) for a in LEVELS}

    # Historical simulation: empirical quantile of the previous WINDOW returns
    for a in LEVELS:
        f[("Historical", a)] = r.rolling(WINDOW).quantile(a).shift(1)

    # Normal with RiskMetrics EWMA volatility (lambda = 0.94), zero mean
    ewma_sd = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean().shift(1))
    for a in LEVELS:
        f[("Normal-EWMA", a)] = stats.norm.ppf(a) * ewma_sd

    # GJR-GARCH(1,1,1) with Student-t errors; parameters re-estimated every REFIT days using only past data
    starts = list(range(WINDOW, n, REFIT))
    for s0 in starts:
        s1 = min(s0 + REFIT, n)
        res = arch_model(r.iloc[:s0], mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
        full = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fix(res.params)
        sig = full.conditional_volatility.iloc[s0:s1]  # sigma_t uses information up to t-1
        mu, nu = res.params["mu"], res.params["nu"]
        for a in LEVELS:
            f[("GJR-GARCH-t", a)].iloc[s0:s1] = mu + sig * t_quantile_std(a, nu)
            es[a].iloc[s0:s1] = mu + sig * t_es_std(a, nu)
    for key in f:  # same evaluation window for every method
        f[key].iloc[:WINDOW] = np.nan
    return f, es


summary, es_rows = [], []
all_fc = {}
for col in rets.columns:
    r = rets[col]
    fc, es = forecasts(r)
    all_fc[col] = (fc, es)
    for (m, a), var in fc.items():
        d = pd.concat([r.rename("r"), var.rename("v")], axis=1).dropna()
        hits = (d.r < d.v).values
        summary.append({
            "index": col, "method": m, "level": f"{(1 - a) * 100:.1f}%", "n": len(d),
            "violations": int(hits.sum()), "expected": round(a * len(d), 1),
            "rate %": round(hits.mean() * 100, 2),
            "Kupiec p": round(kupiec(hits, a), 3), "Indep. p": round(christoffersen_ind(hits), 3), "Cond.cov. p": round(cc_test(hits, a), 3),
        })
    # ES check for the GARCH model: realised mean loss on violation days vs mean forecast ES on those days
    for a in LEVELS:
        d = pd.concat([r.rename("r"), fc[("GJR-GARCH-t", a)].rename("v"), es[a].rename("es")], axis=1).dropna()
        h = d.r < d.v
        if h.sum() > 0:
            es_rows.append({"index": col, "level": f"{(1 - a) * 100:.1f}%", "violations": int(h.sum()),
                            "mean realised return on violation days": round(d.r[h].mean(), 2),
                            "mean forecast ES on those days": round(d.es[h].mean(), 2)})

S = pd.DataFrame(summary)
log(f"\nBacktest period: first forecast after {WINDOW} days of history; GARCH re-fitted every {REFIT} days\n")
for lvl in S.level.unique():
    log(f"=== VaR {lvl}: violations vs expected, and p-values (p < 0.05 = model rejected) ===")
    log(S[S.level == lvl].drop(columns="level").to_string(index=False), "\n")

# Count how often each method is rejected at 5% by the unconditional coverage test
S["reject_uc"] = S["Kupiec p"] < 0.05
S["reject_cc"] = S["Cond.cov. p"] < 0.05
log("Number of rejections at 5% across the four indices and two levels (8 tests per method)")
log(S.groupby("method")[["reject_uc", "reject_cc"]].sum().rename(columns={"reject_uc": "Kupiec rejections", "reject_cc": "Cond. coverage rejections"}).to_string(), "\n")

log("Expected Shortfall check, GJR-GARCH-t (realised average loss on violation days vs forecast ES)")
log(pd.DataFrame(es_rows).to_string(index=False), "\n")

# Figure: SPX returns with the three 99% VaR forecasts around the 2008 crisis
r = rets["SPX"]
fc, _ = all_fc["SPX"]
sl = slice("2008-01-01", "2009-12-31")
fig, ax = plt.subplots(figsize=(10, 4))
r.loc[sl].plot(ax=ax, lw=0.5, color="grey", label="SPX return")
for m in ["Historical", "Normal-EWMA", "GJR-GARCH-t"]:
    fc[(m, 0.01)].loc[sl].plot(ax=ax, lw=1.1, label=f"{m} 99% VaR")
ax.legend(fontsize=8)
ax.set_title("1-day 99% VaR forecasts during the 2008-09 crisis (SPX, % return)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "var_2008_crisis.png", dpi=150)
plt.close(fig)
out.close()
print("Done. See results.txt and figures/.")
