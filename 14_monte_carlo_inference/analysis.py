"""
Project 14 - Monte Carlo study: how reliable are standard errors and confidence intervals in a simple regression?

Model: y_i = b0 + b1 * x_i + e_i, true b1 = 0.5. We test H0: b1 = 0.5 (true), so the rejection rate is the actual size of the
nominal 5% test, and the share of intervals containing 0.5 is the coverage of the nominal 95% interval.

Data-generating processes (DGPs)
  A  iid normal errors, iid normal x                      (textbook case: classical inference should work)
  B  heteroskedastic errors: sd(e_i) = 0.25 + x_i^2       (variance depends on the regressor)
  C  autocorrelated errors AND regressor, both AR(1) 0.7  (typical of time-series data)
  D  heavy-tailed errors: Student-t with 3 d.f.           (finite variance, but extreme observations)

Standard errors compared: classical OLS, heteroskedasticity-robust (HC3), Newey-West (HAC) with lag floor(4 (n/100)^(2/9)).
Critical values: Student-t with n-2 d.f. for all three.

Questions
  1. When does each method reach the nominal 5% size and 95% coverage, and how does that depend on n?
  2. What does robustness cost? (average confidence-interval width, power)
  3. How precise is the simulation itself? (Monte Carlo standard errors)

Run:  python analysis.py      (needs numpy pandas scipy matplotlib; about 1-2 minutes)
Outputs: results.txt and figures/*.png in this folder.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")
R = 20000  # Monte Carlo replications
B1 = 0.5
rng = np.random.default_rng(12345)


def log(*a):
    print(*a)
    print(*a, file=out)


def ar1(R_, n, rho):
    """R_ independent AR(1) series of length n with unit variance and a burn-in."""
    e = rng.standard_normal((R_, n + 50))
    z = np.zeros_like(e)
    z[:, 0] = e[:, 0]
    for t in range(1, n + 50):
        z[:, t] = rho * z[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    return z[:, 50:]


def simulate(dgp, n):
    if dgp == "A":
        x = rng.standard_normal((R, n))
        e = rng.standard_normal((R, n))
    elif dgp == "B":
        x = rng.standard_normal((R, n))
        e = (0.25 + x ** 2) * rng.standard_normal((R, n))
    elif dgp == "C":
        x = ar1(R, n, 0.7)
        e = ar1(R, n, 0.7)
    else:
        x = rng.standard_normal((R, n))
        e = rng.standard_t(3, (R, n)) / np.sqrt(3.0)  # variance 1
    return x, 1.0 + B1 * x + e


def estimate(x, y, b1_null):
    """OLS slope and three standard errors, vectorised over replications (rows)."""
    n = x.shape[1]
    xc = x - x.mean(1, keepdims=True)
    sxx = (xc ** 2).sum(1)
    b = (xc * y).sum(1) / sxx
    e = y - y.mean(1, keepdims=True) - b[:, None] * xc
    # classical
    se_cl = np.sqrt((e ** 2).sum(1) / (n - 2) / sxx)
    # HC3
    h = 1 / n + xc ** 2 / sxx[:, None]
    se_hc = np.sqrt(((xc * e / (1 - h)) ** 2).sum(1)) / sxx
    # Newey-West (Bartlett kernel)
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    u = xc * e
    s = (u ** 2).sum(1)
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (u[:, l:] * u[:, :-l]).sum(1)
    se_hac = np.sqrt(np.maximum(s, 1e-12)) / sxx
    return b, {"Classical": se_cl, "HC3 (robust)": se_hc, "Newey-West (HAC)": se_hac}


rows = []
for dgp in "ABCD":
    for n in (25, 50, 200):
        x, y = simulate(dgp, n)
        b, ses = estimate(x, y, B1)
        tcrit = stats.t.ppf(0.975, n - 2)
        for name, se in ses.items():
            reject = np.abs((b - B1) / se) > tcrit
            rows.append({"DGP": dgp, "n": n, "SE": name, "size (nominal 5%)": reject.mean() * 100,
                         "coverage 95% CI": (1 - reject.mean()) * 100, "mean CI width": (2 * tcrit * se).mean(),
                         "std of b1-hat (true)": b.std()})
res = pd.DataFrame(rows)
mc_se = np.sqrt(0.05 * 0.95 / R) * 100
log(f"{R} replications per cell; Monte Carlo standard error of a rejection rate near 5% is about {mc_se:.2f} percentage points.")
log("So rates between about 4.7% and 5.3% are statistically indistinguishable from 5%.\n")
log("Actual size of the nominal 5% test (%), by DGP, sample size and standard error")
piv = res.pivot_table(index=["DGP", "n"], columns="SE", values="size (nominal 5%)").round(1)
log(piv.to_string(), "\n")
log("Average width of the 95% confidence interval")
log(res.pivot_table(index=["DGP", "n"], columns="SE", values="mean CI width").round(3).to_string(), "\n")
sd_tab = res[res.SE == "Classical"].set_index(["DGP", "n"])["std of b1-hat (true)"].round(3)
log("True sampling standard deviation of the slope estimate (the quantity that a correct standard error should estimate):")
log(sd_tab.to_string(), "\n")

# ----------------------------------------------------------------------------
# Power at n = 50 under heteroskedasticity (DGP B) and iid errors (DGP A): H0: b1 = 0.5, true b1 = 0.5 + delta
# ----------------------------------------------------------------------------
log("Power of the 5% test at n = 50 (rejection rate, %) when the true slope is 0.5 + delta")
prow = []
for dgp in "AB":
    n = 50
    for delta in (0.0, 0.1, 0.2, 0.3, 0.5):
        x, y0 = simulate(dgp, n)
        y = y0 + delta * x
        b, ses = estimate(x, y, B1)
        tcrit = stats.t.ppf(0.975, n - 2)
        for name in ("Classical", "HC3 (robust)"):
            prow.append({"DGP": dgp, "delta": delta, "SE": name, "rejection %": (np.abs((b - B1) / ses[name]) > tcrit).mean() * 100})
pw = pd.DataFrame(prow).pivot_table(index=["DGP", "delta"], columns="SE", values="rejection %").round(1)
log(pw.to_string())
log("(delta = 0 is the size; classical power under B is not meaningful because its size is wrong.)\n")

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, dgp in zip(axes, "ABCD"):
    sub = res[res.DGP == dgp]
    for name in ["Classical", "HC3 (robust)", "Newey-West (HAC)"]:
        s_ = sub[sub.SE == name]
        ax.plot(s_.n, s_["size (nominal 5%)"], "o-", label=name)
    ax.axhline(5, color="k", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 200])
    ax.set_xticklabels([25, 50, 200])
    ax.set_title(f"DGP {dgp}")
    ax.set_xlabel("n")
axes[0].set_ylabel("actual size of nominal 5% test (%)")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "figures" / "size_by_dgp.png", dpi=150)
plt.close(fig)

log("Caveats: only a simple regression with one regressor and a few stylised DGPs; HAC with a fixed rule-of-thumb lag (no data-driven")
log("bandwidth); no bootstrap or fixed-b critical values, which would improve HAC in small samples; the true slope and error")
log("variance are fixed; results for other designs can differ.")
out.close()
print("\nDone. See results.txt and figures/.")
