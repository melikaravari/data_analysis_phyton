"""
Project 25 - Bayesian modelling with MCMC written from scratch: a hierarchical model and a Bayesian regression

Part A. Hierarchical model ("eight schools", Rubin 1981): coaching programmes for the SAT in 8 schools; estimated effects y_j (points) with known standard
    errors s_j: y = 28, 8, -3, 7, -1, 1, 18, 12; s = 15, 10, 16, 11, 9, 11, 10, 18 (published in Rubin 1981 and Gelman et al., Bayesian Data Analysis).
    Model: y_j ~ N(theta_j, s_j^2); theta_j ~ N(mu, tau^2); flat prior on mu; prior on tau varies. A Gibbs sampler (with a Metropolis step for tau)
    is implemented in numpy.
    Questions: how much should the estimates be shrunk toward the average (partial pooling)? How sensitive are the results to the prior on tau?
    Did the chains converge (R-hat, effective sample size)?

Part B. Bayesian linear regression for log wages (Card's data, as in project 21): log wage on education, experience, experience squared, race, urban
    residence and region, with a Gibbs sampler for (beta, sigma^2). Compare with OLS, check the sensitivity to an informative prior, and run
    posterior predictive checks.

Run:  python analysis.py      (needs numpy pandas scipy matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


# ----------------------------------------------------------------------------
# Convergence diagnostics
# ----------------------------------------------------------------------------
def split_rhat(chains):
    """Split R-hat. chains: array (n_chains, n_draws)."""
    c = np.asarray(chains)
    n = c.shape[1] // 2
    c = np.vstack([c[:, :n], c[:, n:2 * n]])
    W = c.var(axis=1, ddof=1).mean()
    B = n * c.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)


def ess(chains):
    """Effective sample size with Geyer's initial positive sequence, pooled over chains."""
    c = np.asarray(chains)
    m, n = c.shape
    xc = c - c.mean(axis=1, keepdims=True)
    f = np.fft.rfft(xc, 2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    rho = acov.mean(axis=0) / acov.mean(axis=0)[0]
    s, k = 0.0, 0
    while 2 * k + 1 < n:
        pair = rho[2 * k] + rho[2 * k + 1]
        if pair < 0:
            break
        s += pair
        k += 1
    tau_int = -1 + 2 * s
    return m * n / max(tau_int, 1.0)


# =============================================================================
# PART A: eight schools
# =============================================================================
log("=" * 70)
log("PART A: hierarchical model for eight schools (Gibbs + Metropolis, written from scratch)")
log("=" * 70)
y = np.array([28., 8, -3, 7, -1, 1, 18, 12])
s = np.array([15., 10, 16, 11, 9, 11, 10, 18])
J = len(y)

PRIORS = {  # log prior density of tau (up to a constant)
    "half-Cauchy(0, 5)": lambda t: -np.log1p((t / 5) ** 2),
    "uniform(0, 30)": lambda t: 0.0 if t < 30 else -np.inf,
    "half-normal(0, 10)": lambda t: -0.5 * (t / 10) ** 2,
    "inverse-gamma(0.001, 0.001) on tau^2": lambda t: -(0.001 + 1) * np.log(t ** 2) - 0.001 / t ** 2 + np.log(2 * t),
}


def sample_8schools(log_prior_tau, n_iter=30000, burn=5000, seed=0):
    rng = np.random.default_rng(seed)
    mu, tau, theta = y.mean(), 5.0, y.copy()
    draws = np.empty((n_iter, J + 2))
    acc = 0
    for it in range(n_iter):
        prec = 1 / s ** 2 + 1 / tau ** 2
        theta = rng.normal((y / s ** 2 + mu / tau ** 2) / prec, 1 / np.sqrt(prec))
        mu = rng.normal(theta.mean(), tau / np.sqrt(J))
        ss = ((theta - mu) ** 2).sum()
        eta = np.log(tau)
        eta_new = eta + 0.8 * rng.standard_normal()
        tn = np.exp(eta_new)

        def lp(t, e):  # log target for eta = log tau, includes the Jacobian e
            return log_prior_tau(t) + e - J * e - ss / (2 * t ** 2)

        if np.log(rng.random()) < lp(tn, eta_new) - lp(tau, eta):
            tau = tn
            acc += 1
        draws[it] = np.r_[theta, mu, tau]
    return draws[burn:], acc / n_iter


results = {}
for name, f in PRIORS.items():
    ch, accs = [], []
    for k in range(4):
        d_, a_ = sample_8schools(f, seed=100 + k)
        ch.append(d_)
        accs.append(a_)
    results[name] = (np.array(ch), np.mean(accs))

base = "half-Cauchy(0, 5)"
ch, acc = results[base]
log(f"Sampler: 4 chains x 25,000 draws after 5,000 warm-up; Metropolis acceptance rate for tau {acc * 100:.0f}%")
rows = []
for lab, idx in [("mu", J), ("tau", J + 1), ("theta_1", 0), ("theta_8", 7)]:
    rows.append({"parameter": lab, "R-hat": split_rhat(ch[:, :, idx]), "ESS": ess(ch[:, :, idx])})
log("Convergence diagnostics (R-hat should be below 1.01; ESS = effective number of independent draws)")
log(pd.DataFrame(rows).round(3).to_string(index=False), "\n")
if min(r["ESS"] for r in rows) < 400:
    log("A low ESS for tau is the known difficulty of hierarchical models: when tau is near 0 the sampler moves slowly (the 'funnel').\n")

flat = ch.reshape(-1, J + 2)
post = pd.DataFrame({"no pooling (y_j)": y, "partial pooling: posterior mean": flat[:, :J].mean(0), "posterior SD": flat[:, :J].std(0),
                     "95% interval": [f"[{np.percentile(flat[:, j], 2.5):.0f}, {np.percentile(flat[:, j], 97.5):.0f}]" for j in range(J)]},
                    index=[f"school {j + 1}" for j in range(J)]).round(1)
log("Effect of coaching by school: raw estimates versus partial-pooling posterior means")
log(post.to_string())
cp = np.sum(y / s ** 2) / np.sum(1 / s ** 2)
log(f"Complete pooling (all schools identical): {cp:.1f} points. Posterior of mu: mean {flat[:, J].mean():.1f}, 95% interval [{np.percentile(flat[:, J], 2.5):.1f}, {np.percentile(flat[:, J], 97.5):.1f}]; "
    f"posterior of tau: median {np.median(flat[:, J + 1]):.1f}, 95% interval [{np.percentile(flat[:, J + 1], 2.5):.1f}, {np.percentile(flat[:, J + 1], 97.5):.1f}]")
log(f"P(school 1 has the largest effect) = {np.mean(flat[:, :J].argmax(1) == 0):.2f}; P(effect of school 1 > effect of school 3) = {np.mean(flat[:, 0] > flat[:, 2]):.2f}")
log("The raw estimate of school 1 (28 points) is shrunk strongly toward the mean: the data cannot distinguish a real difference from sampling noise.\n")

rows = []
for name, (c_, a_) in results.items():
    f_ = c_.reshape(-1, J + 2)
    rows.append({"prior on tau": name, "tau median": np.median(f_[:, J + 1]), "tau 95% interval": f"[{np.percentile(f_[:, J + 1], 2.5):.1f}, {np.percentile(f_[:, J + 1], 97.5):.1f}]",
                 "theta_1 mean": f_[:, 0].mean(), "mu mean": f_[:, J].mean(), "R-hat tau": split_rhat(c_[:, :, J + 1]), "ESS tau": ess(c_[:, :, J + 1])})
log("Prior sensitivity: the data (8 schools) carry little information about tau, so the prior matters")
sens = pd.DataFrame(rows)
log(sens.round(2).to_string(index=False))
lo_, hi_ = sens.loc[sens["tau median"].idxmin()], sens.loc[sens["tau median"].idxmax()]
log(f"The posterior median of tau ranges from {lo_['tau median']:.1f} ({lo_['prior on tau']}) to {hi_['tau median']:.1f} ({hi_['prior on tau']}), and the posterior mean of theta_1 from "
    f"{sens['theta_1 mean'].min():.1f} to {sens['theta_1 mean'].max():.1f}. The inverse-gamma(0.001) prior is often called 'non-informative' but here it puts most weight near tau = 0 and "
    "gives the strongest shrinkage (Gelman 2006 warns about this); the uniform prior gives the weakest.\n")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for k in range(4):
    axes[0].plot(ch[k, :1500, J + 1], lw=0.5)
axes[0].set_title("Trace of tau (first 1,500 draws, 4 chains)")
for name, (c_, a_) in results.items():
    f_ = c_.reshape(-1, J + 2)
    axes[1].hist(f_[:, J + 1], bins=np.linspace(0, 30, 61), histtype="step", density=True, label=name.split(" on")[0])
axes[1].set_xlabel("tau")
axes[1].set_title("Posterior of tau under four priors")
axes[1].legend(fontsize=6)
for j in range(J):
    axes[2].plot([0, 1], [y[j], flat[:, j].mean()], color="C0", lw=1)
axes[2].axhline(flat[:, J].mean(), color="r", ls="--", lw=1)
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(["raw estimate", "posterior mean"])
axes[2].set_title("Shrinkage toward the average")
fig.tight_layout()
fig.savefig(HERE / "figures" / "eight_schools.png", dpi=150)
plt.close(fig)

# =============================================================================
# PART B: Bayesian linear regression (Gibbs sampler)
# =============================================================================
log("=" * 70)
log("PART B: Bayesian regression for log wage (Card data)")
log("=" * 70)
card = pd.read_csv(HERE.parent / "data" / "card.csv")
cols = ["educ", "exper", "expersq", "black", "smsa", "south"]
X = np.column_stack([np.ones(len(card)), card[cols].values])
yv = card.lwage.values
n, p = X.shape
names = ["intercept"] + cols


def gibbs_reg(prior_mean, prior_sd, n_iter=6000, burn=1000, seed=1):
    rng = np.random.default_rng(seed)
    S0inv = np.diag(1 / np.asarray(prior_sd, float) ** 2)
    m0 = np.asarray(prior_mean, float)
    a0 = b0 = 0.01
    XtX, Xty = X.T @ X, X.T @ yv
    sig2 = yv.var()
    B = np.empty((n_iter, p))
    S2 = np.empty(n_iter)
    for it in range(n_iter):
        V = np.linalg.inv(XtX / sig2 + S0inv)
        beta = rng.multivariate_normal(V @ (Xty / sig2 + S0inv @ m0), V)
        rss = ((yv - X @ beta) ** 2).sum()
        sig2 = 1 / rng.gamma(a0 + n / 2, 1 / (b0 + rss / 2))
        B[it], S2[it] = beta, sig2
    return B[burn:], S2[burn:]


weak = gibbs_reg(np.zeros(p), np.full(p, 10.0))
inf_mean = np.zeros(p)
inf_mean[1] = 0.05
inf_sd = np.full(p, 10.0)
inf_sd[1] = 0.005  # strong prior: educ coefficient 0.05 +- 0.005
inform = gibbs_reg(inf_mean, inf_sd, seed=2)
ols = np.linalg.lstsq(X, yv, rcond=None)[0]
res = yv - X @ ols
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (res @ res) / (n - p))
rows = []
for i, nm in enumerate(names):
    b = weak[0][:, i]
    bi = inform[0][:, i]
    rows.append({"coefficient": nm, "OLS": ols[i], "OLS 95% CI": f"[{ols[i] - 1.96 * se[i]:.3f}, {ols[i] + 1.96 * se[i]:.3f}]", "Bayes mean (weak prior)": b.mean(),
                 "95% credible interval": f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]", "Bayes mean (strong prior on educ)": bi.mean()})
log(f"{n} men; Gibbs sampler, 5,000 retained draws after 1,000 warm-up; weakly informative priors N(0, 10^2) on all coefficients")
log(pd.DataFrame(rows).round(3).to_string(index=False))
log(f"Posterior sd of the residual: {np.sqrt(weak[1]).mean():.3f} (OLS {np.sqrt(res @ res / (n - p)):.3f}). With weak priors and n = 3,010 the Bayesian and OLS answers coincide.")
log(f"Strong prior on the education coefficient (0.05 +- 0.005): the posterior mean is {inform[0][:, 1].mean():.3f}, a compromise between the prior (0.050) and the data (OLS {ols[1]:.3f}, se {se[1]:.3f}); "
    "a prior this tight is as informative as the data themselves.")
b_ed = weak[0][:, 1]
b_bl = weak[0][:, 4]
log(f"Posterior probabilities: P(return to education > 8%) = {np.mean(b_ed > 0.08):.2f}; P(black coefficient < -0.15) = {np.mean(b_bl < -0.15):.2f}\n")

# Posterior predictive checks
rng = np.random.default_rng(9)
idx = rng.choice(len(weak[1]), 500, replace=False)
rep = np.array([X @ weak[0][i] + rng.normal(0, np.sqrt(weak[1][i]), n) for i in idx])
lo_ed, hi_ed = card.educ.values <= 11, card.educ.values >= 16
tests = {"minimum of log wage": (np.min, None), "skewness of log wage": (stats.skew, None),
         "SD of log wage, educ <= 11": (lambda v: v[lo_ed].std(), None), "SD of log wage, educ >= 16": (lambda v: v[hi_ed].std(), None)}
rows = []
for nm, (f_, _) in tests.items():
    obs = f_(yv)
    reps = np.array([f_(r_) for r_ in rep])
    rows.append({"test statistic": nm, "observed": obs, "replicated mean": reps.mean(), "posterior predictive p": np.mean(reps >= obs)})
log("Posterior predictive checks (p near 0 or 1 = the model does not reproduce this feature of the data)")
ppc = pd.DataFrame(rows)
log(ppc.round(3).to_string(index=False))
bad = ppc[(ppc["posterior predictive p"] < 0.05) | (ppc["posterior predictive p"] > 0.95)]["test statistic"].tolist()
log(("Features the model does not reproduce: " + ", ".join(bad) + ". The observed wages are more negatively skewed than the normal model implies (a left tail of very low wages), "
     "so the normal-error assumption is imperfect; the coefficients are still informative but predictive intervals in the left tail would be too optimistic.\n") if bad else "The checks show no clear misfit.\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].hist(weak[0][:, 1], bins=50, density=True, alpha=0.7, label="weak prior")
axes[0].hist(inform[0][:, 1], bins=50, density=True, alpha=0.7, label="strong prior (0.05 +- 0.005)")
axes[0].axvline(ols[1], color="k", ls="--", lw=1, label="OLS")
axes[0].set_xlabel("return to education")
axes[0].legend(fontsize=7)
axes[0].set_title("Posterior of the education coefficient")
axes[1].hist(yv, bins=50, density=True, alpha=0.5, label="observed")
axes[1].hist(rep[0], bins=50, density=True, alpha=0.5, label="one replicated data set")
axes[1].legend(fontsize=7)
axes[1].set_title("Posterior predictive check: log wage")
fig.tight_layout()
fig.savefig(HERE / "figures" / "bayes_regression.png", dpi=150)
plt.close(fig)

log("Caveats: (A) eight observations carry almost no information about tau, so results depend on the prior; the Gibbs sampler mixes slowly near tau = 0 and")
log("modern samplers (Hamiltonian Monte Carlo with a non-centred parametrisation) work better; the normal-normal model assumes known standard errors.")
log("(B) a normal linear model with constant variance for log wages; the checks above indicate which features it misses; priors are subjective choices;")
log("the regression is descriptive (see project 21 for the endogeneity of education).")
out.close()
print("\nDone. See results.txt and figures/.")
