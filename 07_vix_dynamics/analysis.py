"""
Project 7 - VIX dynamics: mean reversion, volatility regimes and forecasting

Questions
  1. How persistent is the VIX and how fast does it revert to its mean (half-life)?
  2. Are there distinct low- and high-volatility regimes (Markov-switching model)?
  3. How does the VIX react to S&P 500 returns, and is the reaction asymmetric?
  4. Out of sample, does a HAR-type model (daily, weekly, monthly averages) beat an AR(1)
     and a random walk, at horizons of 1 and 22 trading days?

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
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


vix = pd.read_csv(HERE.parent / "data" / "vix.csv", parse_dates=["date"]).set_index("date").sort_index()["vix"]
spx = pd.read_csv(HERE.parent / "data" / "stock_indices.csv", parse_dates=["date"]).set_index("date").sort_index()["SPX"]
lv = np.log(vix)
log(f"VIX sample: {vix.index.min().date()} to {vix.index.max().date()}, {len(vix)} days")
log(f"VIX mean {vix.mean():.1f}, median {vix.median():.1f}, min {vix.min():.1f}, max {vix.max():.1f} (on {vix.idxmax().date()})\n")

# ----------------------------------------------------------------------------
# 1. Persistence and mean reversion
# ----------------------------------------------------------------------------
adf_lv = adfuller(lv, autolag="AIC")
adf_v = adfuller(vix, autolag="AIC")
log(f"ADF test log VIX: stat {adf_lv[0]:.2f}, p = {adf_lv[1]:.4f}; VIX level: stat {adf_v[0]:.2f}, p = {adf_v[1]:.4f} (H0: unit root)")

ar1 = sm.OLS(lv.iloc[1:].values, sm.add_constant(lv.shift(1).iloc[1:].values)).fit()
c, phi = ar1.params
half_life = np.log(0.5) / np.log(phi)
log(f"AR(1) on log VIX: phi = {phi:.4f}, long-run mean VIX = {np.exp(c / (1 - phi)):.1f}")
log(f"Half-life of a shock: {half_life:.0f} trading days (~{half_life / 21:.1f} months)\n")

# ----------------------------------------------------------------------------
# 2. Markov-switching regimes on weekly (Friday) log VIX
#    Two regimes, each with its own mean and variance. (I first tried a Markov-switching AR(1); there the AR term
#    absorbed the persistence and the regimes only separated "jumpy" from "calm" weeks lasting 3-7 weeks, so I
#    use the regression version, where regimes describe the level of VIX. A third regime was not identified.)
# ----------------------------------------------------------------------------
wk = lv.resample("W-FRI").last().dropna()
ms = MarkovRegression(wk, k_regimes=2, trend="c", switching_variance=True).fit(search_reps=20)
mu = ms.params[[n for n in ms.params.index if n.startswith("const")]].values
sig2 = ms.params[[n for n in ms.params.index if n.startswith("sigma2")]].values
P = ms.regime_transition[:, :, 0]  # P[i, j] = Pr(next = i | now = j)
order = np.argsort(mu)  # 0 = low-volatility regime
log("Two-regime Markov-switching model on weekly log VIX (own mean and variance per regime)")
for name, i in zip(["Low-volatility regime", "High-volatility regime"], order):
    stay = P[i, i]
    log(f"  {name}: VIX around {np.exp(mu[i]):.0f}, std of log VIX {np.sqrt(sig2[i]):.2f}, "
        f"persistence {stay:.3f}, expected duration {1 / (1 - stay):.0f} weeks")
prob_high = pd.Series(ms.smoothed_marginal_probabilities[order[1]], index=wk.index)
log(f"  Share of weeks in the high-volatility regime (smoothed prob > 0.5): {(prob_high > 0.5).mean() * 100:.0f}%")
for label, a_, b_ in [("2008-09 to 2009-06", "2008-09-01", "2009-06-30"), ("Mar-Apr 2020", "2020-03-01", "2020-04-30"), ("2017", "2017-01-01", "2017-12-31")]:
    log(f"  Average probability of the high regime, {label}: {prob_high.loc[a_:b_].mean():.2f}")
log("")

fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
vix.plot(ax=axes[0], lw=0.7)
axes[0].set_title("VIX")
prob_high.plot(ax=axes[1], lw=0.7)
axes[1].set_title("Smoothed probability of the high-volatility regime")
fig.tight_layout()
fig.savefig(HERE / "figures" / "vix_regimes.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3. VIX reaction to the S&P 500 (daily, asymmetric)
# ----------------------------------------------------------------------------
d = pd.concat([lv.diff().rename("dvix") * 100, 100 * np.log(spx).diff().rename("r")], axis=1).dropna()
d["r_pos"], d["r_neg"] = d.r.clip(lower=0), d.r.clip(upper=0)
X = sm.add_constant(d[["r_pos", "r_neg"]])
asym = sm.OLS(d.dvix, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
log(f"Daily regression: change in log VIX (x100) on positive and negative SPX returns (%), {d.index.min().date()} to {d.index.max().date()}")
log(asym.summary().tables[1].as_text())
wald = asym.t_test("r_pos = r_neg")
log(f"Test of symmetric reaction (r_pos = r_neg): difference {float(wald.effect[0]):.2f}, p = {float(wald.pvalue):.4f}")
log(f"R-squared {asym.rsquared:.2f}. Interpretation: a 1% market fall raises VIX by about {-asym.params['r_neg']:.1f}%, a 1% rise lowers it by about {-asym.params['r_pos']:.1f}%.\n")

# ----------------------------------------------------------------------------
# 4. Forecasting: random walk vs AR(1) vs HAR, train to 2009, test 2010 onwards
# ----------------------------------------------------------------------------
def har_features(s):
    return pd.DataFrame({"d": s, "w": s.rolling(5).mean(), "m": s.rolling(22).mean()})


feat = har_features(lv)
split = pd.Timestamp("2010-01-01")
rows = []
for h in (1, 22):
    y = lv.shift(-h)  # value h days ahead
    data = pd.concat([y.rename("y"), feat, lv.rename("lv")], axis=1).dropna()
    tr, te = data[data.index < split - pd.Timedelta(days=h * 2)], data[data.index >= split]
    har = sm.OLS(tr.y, sm.add_constant(tr[["d", "w", "m"]])).fit()
    ar = sm.OLS(tr.y, sm.add_constant(tr[["d"]])).fit()
    f = {
        "Random walk": te.lv,
        "AR(1)": ar.predict(sm.add_constant(te[["d"]])),
        "HAR (daily, weekly, monthly avg)": har.predict(sm.add_constant(te[["d", "w", "m"]])),
    }
    for name, p in f.items():
        e = te.y - p
        rows.append({"horizon (days)": h, "model": name, "RMSE (log VIX)": round(float(np.sqrt((e ** 2).mean())), 4), "MAE": round(float(e.abs().mean()), 4)})
    log(f"HAR coefficients, horizon {h}: " + ", ".join(f"{k} {v:.3f}" for k, v in har.params.items()))
log(f"\nOut-of-sample forecasts of log VIX, test period {split.date()} to {vix.index.max().date()} (lower is better)")
log(pd.DataFrame(rows).to_string(index=False))
log("\nCaveats: parameters are estimated once on the training period (no rolling re-estimation); no formal test of equal")
log("forecast accuracy; the VIX is a risk-neutral measure, so forecasts of the VIX are not forecasts of realised volatility.")
out.close()
print("\nDone. See results.txt and figures/.")
