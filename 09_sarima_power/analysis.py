"""
Project 9 - Box-Jenkins (SARIMA) forecasting of daily German electricity prices, with Diebold-Mariano tests

Questions
  1. What do the ACF/PACF and unit-root tests say about the daily average price series (weekly seasonality)?
  2. Which SARIMA specification does the information criterion pick, and are its residuals white noise?
  3. Does SARIMA beat naive benchmarks for one-day-ahead forecasts out of sample, and are the differences
     statistically significant (Diebold-Mariano test)?

Run:  python analysis.py      (needs pandas numpy scipy statsmodels matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


hourly = pd.read_csv(HERE.parent / "data" / "power_DE.csv", parse_dates=["timestamp"]).set_index("timestamp")["price"]
y = hourly.resample("D").mean()
y.index.freq = "D"
log(f"Daily average price, {y.index.min().date()} to {y.index.max().date()}, {len(y)} days (EUR/MWh)")
log("Mean price by weekday (Mon..Sun): " + ", ".join(f"{v:.0f}" for v in y.groupby(y.index.dayofweek).mean()) + "\n")

split = pd.Timestamp("2025-01-01")
train, test = y[y.index < split], y[y.index >= split]
log(f"Training: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days); test: {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)\n")

# ----------------------------------------------------------------------------
# 1. Identification: unit-root tests and ACF/PACF (training data only)
# ----------------------------------------------------------------------------
adf = adfuller(train, autolag="AIC")
kp = kpss(train, regression="c", nlags="auto")
log(f"ADF (H0 unit root) on levels: stat {adf[0]:.2f}, p = {adf[1]:.3f}; KPSS (H0 stationary): stat {kp[0]:.2f}, p = {kp[1]:.3f}")
adf7 = adfuller(train.diff(7).dropna(), autolag="AIC")
log(f"ADF on the 7-day seasonal difference: stat {adf7[0]:.2f}, p = {adf7[1]:.4f}")
log("The series is persistent (the 2021-22 energy crisis dominates) and has a strong weekly pattern, so both a regular and a")
log("seasonal difference are candidates; the information criterion below decides.\n")

fig, axes = plt.subplots(2, 2, figsize=(11, 6))
plot_acf(train, lags=40, ax=axes[0, 0], title="ACF, levels")
plot_pacf(train, lags=40, ax=axes[0, 1], title="PACF, levels", method="ywm")
d = train.diff().diff(7).dropna()
plot_acf(d, lags=40, ax=axes[1, 0], title="ACF, after regular + seasonal (7) differencing")
plot_pacf(d, lags=40, ax=axes[1, 1], title="PACF, after regular + seasonal (7) differencing", method="ywm")
fig.tight_layout()
fig.savefig(HERE / "figures" / "acf_pacf.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 2. Model selection by AIC on the training data (small grid, seasonal period 7)
# ----------------------------------------------------------------------------
cands = []
for (p, d_, q), (P, D, Q) in itertools.product(
        [(1, 0, 1), (2, 0, 1), (1, 1, 1), (2, 1, 1), (1, 1, 2)],
        [(0, 1, 1), (1, 1, 1), (1, 0, 1)]):
    try:
        r = SARIMAX(train, order=(p, d_, q), seasonal_order=(P, D, Q, 7), trend="c" if d_ == 0 and D == 0 else None).fit(disp=False, maxiter=200)
        cands.append({"order": (p, d_, q), "seasonal": (P, D, Q, 7), "AIC": round(r.aic, 1), "BIC": round(r.bic, 1), "res": r})
    except Exception:
        pass
tab = pd.DataFrame(cands).drop(columns="res").sort_values("AIC")
log("SARIMA candidates (training data), best five by AIC")
log(tab.head(5).to_string(index=False), "\n")
best = min(cands, key=lambda c: c["AIC"])
best_res = best["res"]
log(f"Selected: SARIMA{best['order']}x{best['seasonal']}")
log(best_res.summary().tables[1].as_text(), "\n")

lb = acorr_ljungbox(best_res.resid.iloc[14:], lags=[7, 14, 28], return_df=True)
log("Ljung-Box test on residuals (H0: no autocorrelation)")
log(lb.round(3).to_string())
log(f"Residual std {best_res.resid.iloc[14:].std():.1f}; excess kurtosis {stats.kurtosis(best_res.resid.iloc[14:]):.1f} "
    "(heavy tails from price spikes)\n")

# ----------------------------------------------------------------------------
# 3. One-day-ahead forecasts on the test period.
#    Parameters stay fixed at their training estimates; the filter is updated with each new observation.
# ----------------------------------------------------------------------------
res_full = best_res.apply(y, refit=False)
pred = res_full.get_prediction(start=test.index[0]).predicted_mean

fc = pd.DataFrame({
    "SARIMA": pred,
    "Naive (yesterday)": y.shift(1),
    "Seasonal naive (same day last week)": y.shift(7),
    "Mean of last 7 days": y.rolling(7).mean().shift(1),
}).loc[test.index]
err = fc.rsub(test, axis=0)  # actual minus forecast
res_tab = pd.DataFrame({
    "MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
    "MAE vs. naive %": (err.abs().mean() / err["Naive (yesterday)"].abs().mean() - 1) * 100,
}).round(2)
log("One-day-ahead forecast accuracy, test period (EUR/MWh)")
log(res_tab.to_string(), "\n")


def diebold_mariano(e1, e2, h=1):
    """DM test, squared-error loss, Newey-West variance (h-1 lags, at least 5 for robustness). Negative stat: model 1 better."""
    dd = (e1 ** 2 - e2 ** 2).values
    n, mean = len(dd), dd.mean()
    L = max(h - 1, 5)
    var = np.var(dd, ddof=0)
    for k in range(1, L + 1):
        var += 2 * (1 - k / (L + 1)) * np.cov(dd[k:], dd[:-k], ddof=0)[0, 1]
    stat = mean / np.sqrt(var / n)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


log("Diebold-Mariano tests (squared error loss): SARIMA vs. each benchmark. Negative statistic = SARIMA more accurate.")
rows = []
for b in ["Naive (yesterday)", "Seasonal naive (same day last week)", "Mean of last 7 days"]:
    s, p = diebold_mariano(err["SARIMA"], err[b])
    rows.append({"benchmark": b, "DM statistic": round(s, 2), "p-value": round(p, 4)})
log(pd.DataFrame(rows).to_string(index=False), "\n")

fig, ax = plt.subplots(figsize=(10, 3.8))
w = slice("2025-09-01", "2025-11-15")
test.loc[w].plot(ax=ax, color="k", lw=1.5, label="actual")
fc["SARIMA"].loc[w].plot(ax=ax, lw=1.1, label="SARIMA 1-day-ahead")
fc["Naive (yesterday)"].loc[w].plot(ax=ax, lw=0.9, alpha=0.8, label="naive")
ax.legend(fontsize=8)
ax.set_title("Example period in the test set (daily average price, EUR/MWh)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "forecast_example.png", dpi=150)
plt.close(fig)

log("Caveats: fixed parameters (no re-estimation) and only one-day-ahead forecasts; the grid of orders is small; the training period")
log("contains the 2021-22 crisis, which dominates the variance; squared-error loss is dominated by price spikes.")
out.close()
print("\nDone. See results.txt and figures/.")
