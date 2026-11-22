"""
Project 8 - Jet fuel and Brent: error-correction model and asymmetric price pass-through ("rockets and feathers")

Questions
  1. How large and how persistent is the jet fuel "crack spread" over Brent?
  2. How quickly does jet fuel return to its long-run relationship with Brent (error-correction speed)?
  3. Do jet fuel prices react differently to rising and falling Brent prices (asymmetric pass-through),
     and does the speed of adjustment differ for positive and negative deviations?

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

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


brent = pd.read_csv(HERE.parent / "data" / "brent.csv", parse_dates=["date"])
energy = pd.read_csv(HERE.parent / "data" / "energy.csv", parse_dates=["date"])
df = brent.merge(energy, on="date").set_index("date").sort_index()[["BRENT", "JETFUEL"]].dropna()
log(f"Sample: {df.index.min().date()} to {df.index.max().date()}, {len(df)} days with both prices")

# ----------------------------------------------------------------------------
# 1. Crack spread. Jet fuel is quoted in USD/gallon, Brent in USD/barrel (1 barrel = 42 gallons)
# ----------------------------------------------------------------------------
df["jet_bbl"] = df.JETFUEL * 42
df["crack"] = df.jet_bbl - df.BRENT
by_year = df.groupby(df.index.year)["crack"].agg(["mean", "std", "min", "max"]).round(1)
log("\nJet fuel crack spread over Brent (USD per barrel), by year")
log(by_year.to_string())
log(f"Overall mean {df.crack.mean():.1f} USD/bbl; autocorrelation at lag 1: {df.crack.autocorr(1):.3f}, at lag 60: {df.crack.autocorr(60):.3f}\n")

fig, ax = plt.subplots(figsize=(10, 3.6))
df.crack.plot(ax=ax, lw=0.6)
ax.axhline(df.crack.mean(), color="k", ls="--", lw=0.8)
ax.set_title("Jet fuel crack spread over Brent (USD/bbl)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "crack_spread.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 2. Long-run relation (Engle-Granger step 1): log jet = a + b * log Brent + u
# ----------------------------------------------------------------------------
lj, lb = np.log(df.JETFUEL), np.log(df.BRENT)
lr = sm.OLS(lj, sm.add_constant(lb)).fit()
u = lr.resid * 100  # equilibrium error in percent, same units as the returns, so gamma = share of the gap closed per day
log(f"Long-run relation: log(jet) = {lr.params.iloc[0]:.3f} + {lr.params.iloc[1]:.3f} * log(Brent)  (elasticity {lr.params.iloc[1]:.2f}); R2 = {lr.rsquared:.2f}")
log("(Coefficient standard errors from this step are not valid in levels regressions with trending data; used only for the equilibrium error.)\n")

# ----------------------------------------------------------------------------
# 3. Error-correction models (step 2), daily data, HAC standard errors
#    dJet_t = c + gamma * u_{t-1} + sum_i a_i dBrent_{t-i} (i=0..5) + sum_j b_j dJet_{t-j} (j=1..2) + e_t
# ----------------------------------------------------------------------------
dj, db = lj.diff() * 100, lb.diff() * 100
LAGS = 5


def build(asym_brent=False, asym_ect=False):
    X = pd.DataFrame(index=df.index)
    for i in range(0, LAGS + 1):
        if asym_brent:
            X[f"dB+_{i}"] = db.shift(i).clip(lower=0)
            X[f"dB-_{i}"] = db.shift(i).clip(upper=0)
        else:
            X[f"dB_{i}"] = db.shift(i)
    for j in (1, 2):
        X[f"dJ_{j}"] = dj.shift(j)
    if asym_ect:
        X["ECT+"] = u.shift(1).clip(lower=0)
        X["ECT-"] = u.shift(1).clip(upper=0)
    else:
        X["ECT"] = u.shift(1)
    return X


def fit(X, y=dj, sample=None):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if sample is not None:
        d = d.loc[sample]
    return sm.OLS(d.y, sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 10})


# (a) Symmetric ECM
m1 = fit(build())
gamma = m1.params["ECT"]
log("(a) Symmetric error-correction model for daily jet fuel returns (%)")
log(f"  speed of adjustment gamma = {gamma:.3f} per day (t = {m1.tvalues['ECT']:.1f}); half-life of a deviation = {np.log(0.5) / np.log(1 + gamma):.0f} days")
sr = sum(m1.params[f"dB_{i}"] for i in range(LAGS + 1))
log(f"  cumulative short-run pass-through of Brent over lag 0-{LAGS}: {sr:.2f}")
log(f"  contemporaneous pass-through (dB_0): {m1.params['dB_0']:.2f}   R2 = {m1.rsquared:.2f}\n")

# (b) Asymmetric Brent pass-through: separate coefficients for rises and falls in Brent
m2 = fit(build(asym_brent=True))
up = [f"dB+_{i}" for i in range(LAGS + 1)]
dn = [f"dB-_{i}" for i in range(LAGS + 1)]
cum_up, cum_dn = m2.params[up].sum(), m2.params[dn].sum()
R = np.zeros((1, len(m2.params)))
for n in up:
    R[0, list(m2.params.index).index(n)] += 1
for n in dn:
    R[0, list(m2.params.index).index(n)] -= 1
w = m2.wald_test(R, scalar=True)
log("(b) Asymmetric pass-through (Brent rises vs falls)")
log(f"  cumulative pass-through of Brent rises  (lags 0-{LAGS}): {cum_up:.3f}")
log(f"  cumulative pass-through of Brent falls  (lags 0-{LAGS}): {cum_dn:.3f}")
log(f"  Wald test of equality: statistic {float(w.statistic):.2f}, p = {float(w.pvalue):.3f}")
log("  -> 'rockets and feathers' would mean jet fuel reacts MORE strongly to rises than to falls\n")

# (c) Asymmetric error correction: does jet fuel adjust differently when above or below equilibrium?
m3 = fit(build(asym_ect=True))
R = np.zeros((1, len(m3.params)))
R[0, list(m3.params.index).index("ECT+")] = 1
R[0, list(m3.params.index).index("ECT-")] = -1
w3 = m3.wald_test(R, scalar=True)
log("(c) Asymmetric error correction")
log(f"  adjustment when jet fuel is ABOVE equilibrium (ECT+): {m3.params['ECT+']:.3f} (t = {m3.tvalues['ECT+']:.1f})")
log(f"  adjustment when jet fuel is BELOW equilibrium (ECT-): {m3.params['ECT-']:.3f} (t = {m3.tvalues['ECT-']:.1f})")
log(f"  Wald test of equality: statistic {float(w3.statistic):.2f}, p = {float(w3.pvalue):.3f}\n")

# ----------------------------------------------------------------------------
# 4. Stability: repeat (a)-(c) before and after 2020 (the long-run relation u is the full-sample one)
# ----------------------------------------------------------------------------
rows = []
for label, sl in [("2010-2019", slice("2010", "2019")), ("2020-2026", slice("2020", "2026"))]:
    a = fit(build(), sample=sl)
    b = fit(build(asym_brent=True), sample=sl)
    c = fit(build(asym_ect=True), sample=sl)
    rows.append({
        "period": label,
        "gamma (symmetric)": round(a.params["ECT"], 3),
        "cum. pass-through, Brent rises": round(b.params[up].sum(), 3),
        "cum. pass-through, Brent falls": round(b.params[dn].sum(), 3),
        "ECT+": round(c.params["ECT+"], 3),
        "ECT-": round(c.params["ECT-"], 3),
    })
log("Stability across subperiods")
log(pd.DataFrame(rows).to_string(index=False))
log("\nCaveats: the equilibrium error u uses a full-sample cointegrating regression (so subperiod results use information from")
log("the whole sample); only one lag structure was tried; Brent is a crude benchmark, not the price paid by airlines;")
log("daily data with holiday gaps removed can shift the timing of reactions by a day.")
out.close()
print("\nDone. See results.txt and figures/.")
