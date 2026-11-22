"""
Project 3 - German hourly electricity prices: patterns, negative prices and day-ahead forecasting

Questions
  1. How do prices differ by hour of day, weekday and year, and how often are prices negative?
  2. Can a simple regression with calendar effects and lagged prices beat naive benchmarks
     when forecasting hourly prices one day ahead?
  3. Does a non-linear model (gradient boosting) add anything over the linear model?

Run:  python analysis.py      (needs pandas numpy scikit-learn matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# 1. Load and check the time index
# ----------------------------------------------------------------------------
raw = pd.read_csv(HERE.parent / "data" / "power_DE.csv", parse_dates=["timestamp"]).set_index("timestamp").sort_index()
full_idx = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
missing = full_idx.difference(raw.index)
log(f"Sample: {raw.index.min()} to {raw.index.max()}, {len(raw)} hours")
log(f"Hours missing from a regular hourly grid: {len(missing)} (e.g. daylight-saving changes)")
price = raw["price"].reindex(full_idx).interpolate(limit=2)  # fill only these short gaps
price.name = "price"
log(f"Price summary (EUR/MWh):\n{price.describe().round(1).to_string()}\n")

df = price.to_frame()
df["year"], df["month"], df["hour"], df["dow"] = df.index.year, df.index.month, df.index.hour, df.index.dayofweek

# ----------------------------------------------------------------------------
# 2. Descriptive analysis
# ----------------------------------------------------------------------------
by_year = df.groupby("year")["price"].agg(
    mean="mean", std="std", min="min", max="max",
    share_negative=lambda s: (s < 0).mean() * 100,
).round(1)
by_year["hours_negative"] = df.groupby("year")["price"].apply(lambda s: int((s < 0).sum()))
log("Prices by year (share_negative in %)")
log(by_year.to_string(), "\n")

neg_by_hour = df.assign(neg=df.price < 0).groupby("hour")["neg"].mean().mul(100).round(1)
log("Share of negative-price hours by hour of day, all years (%):")
log(neg_by_hour.to_string(), "\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for y, g in df.groupby("year"):
    g.groupby("hour")["price"].mean().plot(ax=axes[0], label=str(y), lw=1.2)
axes[0].set_title("Average price by hour of day, per year")
axes[0].set_xlabel("hour")
axes[0].set_ylabel("EUR/MWh")
axes[0].legend(fontsize=7, ncol=2)
by_year["share_negative"].plot.bar(ax=axes[1])
axes[1].set_title("Share of negative-price hours (%)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "daily_profile_and_negative_prices.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. Forecasting: predict the price at hour t using information available the day before
#    (so only lags of 24 hours or more are allowed as predictors)
# ----------------------------------------------------------------------------
X = pd.DataFrame(index=df.index)
for k in (24, 48, 168):
    X[f"lag_{k}"] = df.price.shift(k)
X["mean_prev_day"] = df.price.rolling(24).mean().shift(24)
X["min_prev_day"] = df.price.rolling(24).min().shift(24)
X["max_prev_day"] = df.price.rolling(24).max().shift(24)
X["weekend"] = (df.dow >= 5).astype(int)

cal = pd.get_dummies(df[["hour", "dow", "month"]].astype(str), drop_first=True).astype(float)
# hour x weekend interaction: weekend daily profile differs (e.g. no morning peak)
for h in range(1, 24):
    cal[f"hour{h}_x_wkd"] = ((df.hour == h) & (df.dow >= 5)).astype(float)
X = pd.concat([X, cal], axis=1)

data = pd.concat([df.price, X], axis=1).dropna()
test_start = pd.Timestamp("2025-01-01")
train, test = data[data.index < test_start], data[data.index >= test_start]
log(f"Train: {train.index.min().date()} to {train.index.max().date()} ({len(train)} h)")
log(f"Test : {test.index.min().date()} to {test.index.max().date()} ({len(test)} h)\n")

mu, sd = train.drop(columns="price").mean(), train.drop(columns="price").std().replace(0, 1)


def scale(d):
    return (d.drop(columns="price") - mu) / sd


preds = {
    "Naive: same hour yesterday": test["lag_24"],
    "Naive: same hour last week": test["lag_168"],
}
ridge = Ridge(alpha=10.0).fit(scale(train), train.price)
preds["Linear (ridge): calendar + lags"] = pd.Series(ridge.predict(scale(test)), index=test.index)
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(train.drop(columns="price"), train.price)
preds["Gradient boosting"] = pd.Series(gb.predict(test.drop(columns="price")), index=test.index)

rows = []
for name, p in preds.items():
    e = test.price - p
    rows.append({"model": name, "MAE": round(e.abs().mean(), 2), "RMSE": round(np.sqrt((e ** 2).mean()), 2)})
res = pd.DataFrame(rows)
log("Out-of-sample forecast accuracy (EUR/MWh, lower is better)")
log(res.to_string(index=False), "\n")

# Where do the errors come from? Break MAE down by hour for the two best models
best_name = res.sort_values("MAE").iloc[0]["model"]
err_h = pd.DataFrame({n: (test.price - preds[n]).abs().groupby(test.index.hour).mean()
                      for n in ["Naive: same hour last week", best_name]}).round(1)
log(f"MAE by hour of day, naive weekly vs. best model ({best_name}):")
log(err_h.to_string(), "\n")

# Figure: one example week
wk = test.index[(test.index >= "2025-06-02") & (test.index < "2025-06-09")]
fig, ax = plt.subplots(figsize=(10, 3.8))
test.price.loc[wk].plot(ax=ax, label="actual", lw=1.5, color="k")
preds["Naive: same hour last week"].loc[wk].plot(ax=ax, label="naive (last week)", lw=1, alpha=0.8)
preds[best_name].loc[wk].plot(ax=ax, label=best_name, lw=1, alpha=0.8)
ax.legend(fontsize=8)
ax.set_title("Example week in the test period")
ax.set_ylabel("EUR/MWh")
fig.tight_layout()
fig.savefig(HERE / "figures" / "example_week_forecast.png", dpi=150)
plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Classical time-series benchmark: SARIMA on the DAILY average price.
#    To compare with the hourly models, the hourly forecasts are averaged to daily forecasts. All forecasts only use
#    information up to the previous day. SARIMA(1,1,2)(0,1,1,7) was chosen by AIC on the training data in project 09.
# ----------------------------------------------------------------------------
daily = price.resample("D").mean()
daily.index.freq = "D"
d_train = daily[daily.index < test_start]
sar = SARIMAX(d_train, order=(1, 1, 2), seasonal_order=(0, 1, 1, 7)).fit(disp=False, maxiter=200)
sar_pred = sar.apply(daily, refit=False).get_prediction(start=test_start).predicted_mean

counts = test.price.groupby(test.index.normalize()).size()
full_days = counts[counts == 24].index  # drop a possibly incomplete last day
d_actual = test.price.groupby(test.index.normalize()).mean().loc[full_days]
cmp_ = {"SARIMA(1,1,2)(0,1,1)7 on daily series": sar_pred.loc[full_days]}
for name in ["Linear (ridge): calendar + lags", "Gradient boosting"]:
    cmp_[name + " (hourly forecasts averaged)"] = preds[name].groupby(test.index.normalize()).mean().loc[full_days]
cmp_["Naive: yesterday's daily average"] = daily.shift(1).loc[full_days]
rows = []
for name, p in cmp_.items():
    e = d_actual - p
    rows.append({"model": name, "MAE": round(e.abs().mean(), 2), "RMSE": round(np.sqrt((e ** 2).mean()), 2)})
log(f"Daily-average forecasts, {len(full_days)} complete test days (EUR/MWh)")
log(pd.DataFrame(rows).to_string(index=False), "\n")

log("Caveat: no fundamentals (load, wind, solar, gas price) are used, so the models cannot anticipate price spikes or")
log("negative-price hours caused by weather. Adding such exogenous variables is the natural next step.")
out.close()
print("\nDone. See results.txt and figures/.")
