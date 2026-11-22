"""
Project 29 - Climate time series: CO2 growth, global temperature trends, and the danger of spurious regression

Data
  * Atmospheric CO2 at Mauna Loa, Hawaii, monthly, March 1958 onward (NOAA GML; file from the Datahub "co2-ppm" repository). The file header does not match the
    seven data fields, so the columns are read by position: date, decimal date, monthly mean (ppm), seasonally adjusted value, number of days, std. dev., uncertainty.
    Missing values are coded -99.99 / -1 / -9.99.
  * Global mean surface temperature anomalies, monthly (NASA GISTEMP, 1880-2025, relative to 1951-1980; NOAA GCAG 1850-2026 for a robustness check; Datahub "global-temp").

Questions
  1. How fast is CO2 rising, is the growth accelerating, and how does the seasonal cycle behave? (STL decomposition, quadratic-trend regression with HAC errors)
  2. How large is the warming trend, and how much does ignoring autocorrelation understate its uncertainty? Is there evidence of acceleration?
  3. How does annual temperature relate to CO2? Why are regressions in levels misleading (spurious regression), and what do differences and cointegration tests show?
  4. Which simple model forecasts CO2 best out of sample?

Run:  python analysis.py      (needs numpy pandas scipy statsmodels matplotlib)
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
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, coint, kpss

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
c = pd.read_csv(HERE.parent / "data" / "co2_mauna_loa_monthly.csv", header=None, skiprows=1,
                names=["date", "decimal", "co2", "co2_sa", "ndays", "sd", "unc"])
c["date"] = pd.PeriodIndex(c.date, freq="M").to_timestamp()
c = c.set_index("date")
c["co2"] = c.co2.where(c.co2 > 0)  # missing values coded as negative numbers
n_miss = int(c.co2.isna().sum())
co2 = c.co2.asfreq("MS").interpolate()
log(f"CO2: {c.index.min():%Y-%m} to {c.index.max():%Y-%m}, {len(c)} monthly values ({n_miss} missing values interpolated); "
    f"{co2.iloc[0]:.1f} ppm in {co2.index[0]:%Y} to {co2.iloc[-1]:.1f} ppm in {co2.index[-1]:%Y}")

t = pd.read_csv(HERE.parent / "data" / "gistemp_monthly.csv")
t["date"] = pd.PeriodIndex(t.Year, freq="M").to_timestamp()
gis = t[t.Source == "GISTEMP"].set_index("date").Mean
gc = t[t.Source == "GCAG"].set_index("date").Mean


def annual(s):
    g = s.groupby(s.index.year)
    return g.mean()[g.count() == 12]  # full years only


ta, tg = annual(gis), annual(gc)
log(f"Temperature anomalies: GISTEMP {ta.index.min()}-{ta.index.max()} ({len(ta)} full years), GCAG {tg.index.min()}-{tg.index.max()} ({len(tg)} full years); "
    f"warmest year in GISTEMP: {ta.idxmax()} ({ta.max():.2f} C); correlation of the two sources over the common years: {ta.corr(tg):.3f}\n")

# ----------------------------------------------------------------------------
# 1. CO2: STL decomposition, growth by decade, acceleration, seasonal amplitude
# ----------------------------------------------------------------------------
stl = STL(co2, period=12, robust=True).fit()
trend, seas = stl.trend, stl.seasonal
amp = seas.groupby(seas.index.year).agg(lambda v: v.max() - v.min())
amp = amp[seas.groupby(seas.index.year).count() == 12]
am = smf.ols("a ~ y", pd.DataFrame({"a": amp.values, "y": amp.index})).fit(cov_type="HC3")
log(f"Seasonal cycle (STL): peak-to-trough amplitude {amp.loc[1960:1969].mean():.1f} ppm in the 1960s, {amp.loc[2010:2019].mean():.1f} ppm in the 2010s; "
    f"linear change {am.params['y'] * 10:.2f} ppm per decade (95% CI {am.conf_int().loc['y', 0] * 10:.2f} to {am.conf_int().loc['y', 1] * 10:.2f})")
peak_month = seas.groupby(seas.index.month).mean().idxmax()
low_month = seas.groupby(seas.index.month).mean().idxmin()
log(f"The CO2 cycle peaks in month {peak_month} (May) and bottoms out in month {low_month}: this is consistent with Northern-hemisphere vegetation taking up CO2 in the growing season (a physical interpretation, not tested here).\n")

ca = annual(co2)
rows = []
for dec in range(1960, 2030, 10):
    sub = ca.loc[dec:min(dec + 9, ca.index.max())]
    if len(sub) >= 5:
        sl = np.polyfit(sub.index, sub.values, 1)[0]
        rows.append({"decade": f"{dec}s", "mean CO2 (ppm)": sub.mean(), "growth (ppm per year)": sl})
log("Growth of annual mean CO2 by decade")
log(pd.DataFrame(rows).round(2).to_string(index=False))
log("")

tt = np.arange(len(trend)) / 12.0  # years since start
df_ = pd.DataFrame({"trend": trend.values, "t": tt})
q = smf.ols("trend ~ t + I(t**2)", df_).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
l = smf.ols("trend ~ t", df_).fit()
accel = 2 * q.params["I(t ** 2)"]
rate0 = q.params["t"]
rate1 = q.params["t"] + 2 * q.params["I(t ** 2)"] * tt[-1]
log(f"Quadratic trend fitted to the STL trend: acceleration {accel:.3f} ppm per year per year (HAC t = {q.tvalues['I(t ** 2)']:.1f}); implied growth {rate0:.2f} ppm/yr at the start "
    f"and {rate1:.2f} ppm/yr at the end; R2 linear {l.rsquared:.4f} vs quadratic {q.rsquared:.4f}")
log(f"The implied growth rate of CO2 has risen by a factor of {rate1 / rate0:.1f} since 1958; a linear trend is clearly rejected (R2 {l.rsquared:.3f} vs {q.rsquared:.3f}).\n")

# ----------------------------------------------------------------------------
# 2. Temperature: trend with and without accounting for autocorrelation
# ----------------------------------------------------------------------------
def trend_fit(s, start):
    y = s.loc[start:]
    d = pd.DataFrame({"y": y.values, "t": (y.index - y.index[0]) / 10.0})  # time in decades
    X = sm.add_constant(d[["t"]])
    ols = sm.OLS(d.y, X).fit()
    hac = sm.OLS(d.y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    gl = sm.GLSAR(d.y, X, rho=1).iterative_fit(maxiter=20)
    return ols, hac, gl, d


rows = []
for start in (1880, 1950, 1975, 2000):
    ols, hac, gl, d = trend_fit(ta, start)
    resid = ols.resid
    rows.append({"period": f"{start}-{ta.index.max()}", "trend (C/decade)": ols.params["t"], "naive SE": ols.bse["t"], "HAC SE": hac.bse["t"], "AR(1) GLS trend": gl.params["t"],
                 "AR(1) GLS SE": gl.bse["t"], "residual lag-1 autocorr.": pd.Series(resid).autocorr(1)})
tab = pd.DataFrame(rows).round(3)
log("Linear warming trend of annual global temperature anomaly (GISTEMP)")
log(tab.to_string(index=False))
log("The naive standard error ignores that residuals are autocorrelated; the HAC and AR(1) standard errors are larger, so naive confidence intervals are too narrow.")
ols75, hac75, gl75, d75 = trend_fit(ta, 1975)
gc_t = trend_fit(tg, 1975)
log(f"Since 1975: {ols75.params['t']:.3f} C per decade (95% CI with HAC {hac75.conf_int().loc['t', 0]:.3f} to {hac75.conf_int().loc['t', 1]:.3f}); "
    f"with the NOAA series (GCAG): {gc_t[0].params['t']:.3f} C per decade -> the two sources agree.\n")

# Acceleration: one-knot piecewise linear model, knot chosen by minimum SSE (p-values for a data-chosen knot are not valid)
yrs = ta.index.values.astype(float)
best = None
for knot in range(1920, 2001):
    X = np.column_stack([np.ones(len(yrs)), yrs - 1880, np.maximum(0, yrs - knot)])
    b, res_, *_ = np.linalg.lstsq(X, ta.values, rcond=None)
    sse = ((ta.values - X @ b) ** 2).sum()
    if best is None or sse < best[0]:
        best = (sse, knot, b)
sse_pw, knot, b = best
lin = np.polyfit(yrs, ta.values, 1)
sse_lin = ((ta.values - np.polyval(lin, yrs)) ** 2).sum()
log(f"Piecewise-linear fit with one knot (best knot {knot}): slope {b[1] * 10:.3f} C/decade before, {(b[1] + b[2]) * 10:.3f} C/decade after; SSE falls by {(1 - sse_pw / sse_lin) * 100:.0f}% versus one straight line.")
quad = smf.ols("y ~ t + I(t**2)", pd.DataFrame({"y": ta.values, "t": (yrs - 1880) / 10})).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
log(f"Quadratic trend: curvature coefficient {quad.params['I(t ** 2)']:.4f} C per decade squared (HAC t = {quad.tvalues['I(t ** 2)']:.1f}); positive curvature means accelerating warming.")
log("The knot year is chosen from the data, so the evidence for a change in slope is stronger than a naive test suggests; the exact knot is uncertain.\n")

# Unit-root tests (annual anomalies and annual ln CO2)
adf_t = adfuller(ta, regression="ct", autolag="AIC")
kp_t = kpss(ta, regression="ct", nlags="auto")
log(f"Annual temperature anomaly: ADF with trend (H0 unit root) p = {adf_t[1]:.3f}; KPSS with trend (H0 trend-stationary) p = {kp_t[1]:.3f}")
if adf_t[1] > 0.05 and kp_t[1] < 0.05:
    log("ADF does not reject a unit root and KPSS rejects trend-stationarity: both point to non-stationarity even around a linear trend. This fits either a stochastic trend or a trend that is not "
        "linear (the piecewise and quadratic fits above); with 146 annual observations these tests cannot tell the two apart, and the debate in the literature is unresolved.\n")
else:
    log("The unit-root and stationarity tests do not agree on a clear answer; they have low power in samples of this size.\n")

# ----------------------------------------------------------------------------
# 3. Temperature vs CO2: levels, differences, cointegration
# ----------------------------------------------------------------------------
lc = np.log(ca) / np.log(2)  # log base 2: coefficient = degrees C per doubling of CO2
d3 = pd.DataFrame({"temp": ta, "lco2": lc}).dropna()
d3 = d3.loc[1959:]
lev = smf.ols("temp ~ lco2", d3).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
dd = d3.diff().dropna()
dif = smf.ols("temp ~ lco2", dd).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
eg = coint(d3.temp, d3.lco2, trend="c")
log(f"Annual temperature anomaly vs log2(CO2), {d3.index.min()}-{d3.index.max()} ({len(d3)} years)")
log(f"  Levels regression: {lev.params['lco2']:.2f} C per doubling of CO2 (HAC 95% CI {lev.conf_int().loc['lco2', 0]:.2f} to {lev.conf_int().loc['lco2', 1]:.2f}), R2 = {lev.rsquared:.2f}, residual lag-1 autocorrelation {pd.Series(lev.resid).autocorr(1):.2f}")
log(f"  First differences: {dif.params['lco2']:.2f} C per doubling (HAC 95% CI {dif.conf_int().loc['lco2', 0]:.2f} to {dif.conf_int().loc['lco2', 1]:.2f}), R2 = {dif.rsquared:.2f}")
log(f"  Engle-Granger cointegration test (H0: no cointegration): statistic {eg[0]:.2f}, p = {eg[1]:.3f}")
coint_ok = eg[1] < 0.05
log(f"The levels regression has a high R2 ({lev.rsquared:.2f}) partly because both series trend, which can also happen between unrelated trending variables (spurious regression). "
    + ("Here the Engle-Granger test rejects 'no cointegration', so the long-run co-movement is not just a statistical artefact of two trends; " if coint_ok else
       "Here the Engle-Granger test does not reject 'no cointegration', so the levels relationship may be spurious; ") +
    "but the test has its own size and power limits with 67 annual observations.")
log(f"In first differences the R2 is only {dif.rsquared:.2f} and the coefficient is large and imprecise (CI {dif.conf_int().loc['lco2', 0]:.0f} to {dif.conf_int().loc['lco2', 1]:.0f}): annual changes in log CO2 are small and nearly constant, "
    "so they carry little variation, while year-to-year temperature changes are dominated by natural variability (El Nino, volcanoes). The levels coefficient should not be read as the "
    "climate sensitivity: temperature also responds to aerosols, solar output and ocean heat uptake, and the response lags the forcing. The point of this exercise is statistical: "
    "trending series need differencing or cointegration analysis before any regression result is interpreted.\n")

# ----------------------------------------------------------------------------
# 4. Out-of-sample forecasts of CO2, trained to 2009-12
# ----------------------------------------------------------------------------
tr_, te_ = co2.loc[:"2009-12"], co2.loc["2010-01":]
h = len(te_)
tm = np.arange(len(co2)) / 12.0
Xall = pd.DataFrame({"t": tm, "t2": tm ** 2}, index=co2.index)
for m_ in range(2, 13):
    Xall[f"m{m_}"] = (co2.index.month == m_).astype(float)
Xall = sm.add_constant(Xall)
m_quad = sm.OLS(tr_, Xall.loc[tr_.index]).fit()
m_lin = sm.OLS(tr_, Xall.loc[tr_.index].drop(columns="t2")).fit()
f_quad = m_quad.predict(Xall.loc[te_.index])
f_lin = m_lin.predict(Xall.loc[te_.index].drop(columns="t2"))
sar = SARIMAX(tr_, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
f_sar = sar.get_forecast(h).predicted_mean
hw = ExponentialSmoothing(tr_, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()
f_hw = hw.forecast(h)
fc = pd.DataFrame({"Linear trend + month effects": f_lin, "Quadratic trend + month effects": f_quad, "SARIMA(1,1,1)(0,1,1,12)": f_sar, "Holt-Winters (damped trend)": f_hw})
rows = []
for name in fc:
    e = te_ - fc[name]
    rows.append({"model": name, "RMSE (ppm)": np.sqrt((e ** 2).mean()), "bias (ppm)": e.mean(),
                 "RMSE first 5 years": np.sqrt((e.iloc[:60] ** 2).mean()), "RMSE years 10-15": np.sqrt((e.iloc[120:180] ** 2).mean())})
log(f"Out-of-sample forecasts for {te_.index[0]:%Y-%m} to {te_.index[-1]:%Y-%m} ({h} months) from a model trained on data to 2009-12 (bias = actual minus forecast; positive bias = forecast too low)")
log(pd.DataFrame(rows).round(2).to_string(index=False))
log("A model that includes acceleration (quadratic trend) extrapolates best; models with a constant or damped growth rate underpredict increasingly, because the growth rate of CO2 is still rising.\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
axes[0, 0].plot(co2.index, co2.values, lw=0.6, label="monthly mean")
axes[0, 0].plot(trend.index, trend.values, "r-", lw=1.5, label="STL trend")
axes[0, 0].set_ylabel("CO2 (ppm)")
axes[0, 0].legend()
axes[0, 0].set_title("Atmospheric CO2, Mauna Loa")
axes[0, 1].plot(seas.loc["2015":"2019"].index, seas.loc["2015":"2019"].values)
axes[0, 1].set_title("Seasonal component (2015-2019)")
axes[1, 0].plot(ta.index, ta.values, "o-", ms=3, lw=0.6, label="annual anomaly")
xs = np.arange(1880, 2026)
axes[1, 0].plot(xs, np.polyval(lin, xs), "k--", label="linear")
Xp = np.column_stack([np.ones(len(xs)), xs - 1880, np.maximum(0, xs - knot)])
axes[1, 0].plot(xs, Xp @ b, "r-", label=f"piecewise (knot {knot})")
axes[1, 0].set_ylabel("anomaly (C)")
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title("Global temperature anomaly (GISTEMP)")
for name in fc:
    axes[1, 1].plot(fc.index, fc[name].values, lw=1, label=name)
axes[1, 1].plot(te_.index, te_.values, "k", lw=1.5, label="actual")
axes[1, 1].legend(fontsize=6)
axes[1, 1].set_title("CO2 forecasts from a model trained to 2009")
fig.tight_layout()
fig.savefig(HERE / "figures" / "climate.png", dpi=150)
plt.close(fig)

log("Caveats: statistical description only; no physical model, so nothing here measures climate sensitivity or attributes warming to causes; annual CO2 and")
log("temperature regressions are affected by omitted forcings and lags; the knot of the piecewise model is data-chosen; Mauna Loa is one location (though it tracks")
log("the global mean closely); temperature series are anomalies from reanalysis and station networks with their own uncertainty; the file's last values may be provisional.")
out.close()
print("\nDone. See results.txt and figures/.")
