"""
Project 26 - Design and analysis of a factorial experiment in blocks (ANOVA)

Data: `npk` (Yates 1937; R's datasets package): a classical agricultural experiment on the yield of green peas (pounds per plot). Three fertilisers,
nitrogen N, phosphate P and potassium K, were each applied (1) or not (0) in all 8 combinations (a 2^3 factorial design) on 24 plots grouped into
6 blocks of 4 plots. Each block therefore contains only half of the 8 combinations: the three-factor interaction N:P:K is deliberately *confounded* with
blocks (cannot be separated from block differences) so that the main effects and two-factor interactions are estimated within blocks.

Questions
  1. How large are the main effects of N, P, K and their two-factor interactions, and which are statistically distinguishable from noise?
  2. How much did blocking help? Are the model assumptions reasonable?
  3. What effect size could this experiment detect (power), and do permutation tests agree with the ANOVA?

The same methods (factorial designs, blocking, ANOVA, effect plots) are used in chemistry, engineering and process optimisation.

Run:  python analysis.py      (needs numpy pandas scipy statsmodels matplotlib)
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")
rng = np.random.default_rng(11)


def log(*a):
    print(*a)
    print(*a, file=out)


d = pd.read_csv(HERE.parent / "data" / "npk.csv").rename(columns={"yield": "y"})
d["block"] = d.block.astype(str)
log(f"{len(d)} plots, {d.block.nunique()} blocks of {len(d) // d.block.nunique()}; mean yield {d.y.mean():.1f}, SD {d.y.std():.1f} (pounds per plot)")

# ----------------------------------------------------------------------------
# 1. Check the design: balanced, and the three-factor interaction is confounded with blocks
# ----------------------------------------------------------------------------
log("Treatment combinations (N,P,K) per factor level: " + ", ".join(f"{f}: {int(d[f].sum())} of {len(d)} treated" for f in "NPK"))
d["NPK"] = (d.N + d.P + d.K) % 2  # parity of the number of fertilisers: the sign of the 3-factor contrast
tab = d.groupby("block").NPK.agg(["min", "max"])
log(f"Within each block the 3-factor contrast (parity of N+P+K) takes a single value: {bool((tab['min'] == tab['max']).all())}; blocks alternate between the two halves -> "
    "N:P:K is confounded with blocks and cannot be estimated separately.\n")

# ----------------------------------------------------------------------------
# 2. ANOVA with blocks and all main effects and two-factor interactions
# ----------------------------------------------------------------------------
m = smf.ols("y ~ C(block) + C(N) * C(P) + C(N) * C(K) + C(P) * C(K)", d).fit()
aov = sm.stats.anova_lm(m, typ=1)
aov.index = [i.replace("C(", "").replace(")", "").replace("[T.1]", "") for i in aov.index]
log("ANOVA (sequential sums of squares; the design is orthogonal so the order does not matter)")
log(aov.round(3).to_string(), "\n")
mse, dfe = m.mse_resid, int(m.df_resid)
log(f"Residual mean square {mse:.2f} on {dfe} degrees of freedom (residual SD {np.sqrt(mse):.2f})")

# Effects (mean difference high - low) with confidence intervals from the residual mean square
rows = []
for name, cols in [("N", ["N"]), ("P", ["P"]), ("K", ["K"]), ("N:P", ["N", "P"]), ("N:K", ["N", "K"]), ("P:K", ["P", "K"])]:
    s_ = np.prod([2 * d[c] - 1 for c in cols], axis=0)  # +1 / -1 contrast
    eff = d.y[s_ == 1].mean() - d.y[s_ == -1].mean()
    se = np.sqrt(mse * (1 / (s_ == 1).sum() + 1 / (s_ == -1).sum()))
    t = stats.t.ppf(0.975, dfe)
    p = 2 * (1 - stats.t.cdf(abs(eff / se), dfe))
    rows.append({"effect": name, "estimate": eff, "SE": se, "95% CI": f"[{eff - t * se:.2f}, {eff + t * se:.2f}]", "p": p})
eff = pd.DataFrame(rows).set_index("effect")
log("\nEffects (average yield with the factor at the high level minus the low level, pounds per plot)")
log(eff.round(3).to_string(), "\n")
log("N increases the yield (about +5.6 lb) and K decreases it (about -4.0 lb); P and all interactions are indistinguishable from noise with 24 plots.")
log("A non-significant effect is not evidence of no effect: the confidence intervals for P and the interactions are wide.\n")

# ----------------------------------------------------------------------------
# 3. Blocking efficiency, random-effects version, diagnostics
# ----------------------------------------------------------------------------
m_nb = smf.ols("y ~ C(N) * C(P) + C(N) * C(K) + C(P) * C(K)", d).fit()
log(f"Without blocks the residual mean square would be {m_nb.mse_resid:.1f} instead of {mse:.1f}: blocking removed {(1 - mse / m_nb.mse_resid) * 100:.0f}% of the error variance "
    f"(relative efficiency {m_nb.mse_resid / mse:.1f}).")
dc = d.assign(Nc=2 * d.N - 1, Pc=2 * d.P - 1, Kc=2 * d.K - 1)  # +1/-1 coding so that 2 x coefficient = main effect
mm = smf.mixedlm("y ~ Nc + Pc + Kc + Nc:Pc + Nc:Kc + Pc:Kc", dc, groups=dc.block).fit(reml=True)
log(f"Mixed model with random block effects (+1/-1 coding; effect = 2 x coefficient): N {2 * mm.fe_params['Nc']:.2f}, P {2 * mm.fe_params['Pc']:.2f}, K {2 * mm.fe_params['Kc']:.2f}; "
    f"block SD {np.sqrt(mm.cov_re.iloc[0, 0]):.2f}, residual SD {np.sqrt(mm.scale):.2f}")
log("(the same effect estimates as the fixed-block ANOVA, as expected in an orthogonal design)\n")

r = m.resid
f = m.fittedvalues
lev = stats.levene(*[d.y[d.N == g] - d.y[d.N == g].mean() for g in (0, 1)])
log(f"Diagnostics: Shapiro-Wilk p = {stats.shapiro(r)[1]:.2f} (normal residuals not rejected); Levene test of equal variances between N levels p = {lev.pvalue:.2f}; "
    f"largest studentised residual {np.abs(m.get_influence().resid_studentized_external).max():.2f}")

# ----------------------------------------------------------------------------
# 4. Power / minimum detectable effect (two-level contrast, 12 vs 12 plots, residual SD from the model)
# ----------------------------------------------------------------------------
sd = np.sqrt(mse)
n1 = 12
tcrit = stats.t.ppf(0.975, dfe)
se_eff = sd * np.sqrt(2 / n1)
rows = []
for delta in [2, 3, 4, 5, 6, 8]:
    ncp = delta / se_eff
    power = 1 - stats.nct.cdf(tcrit, dfe, ncp) + stats.nct.cdf(-tcrit, dfe, ncp)
    rows.append({"true effect (lb)": delta, "power (5% test)": power})
mde = next(dl for dl in np.arange(0.5, 15, 0.1) if (1 - stats.nct.cdf(tcrit, dfe, dl / se_eff) + stats.nct.cdf(-tcrit, dfe, dl / se_eff)) >= 0.8)
log("\nPower of the test for a main effect (12 vs 12 plots, standard error of an effect {:.2f} lb)".format(se_eff))
log(pd.DataFrame(rows).round(2).to_string(index=False))
log(f"The experiment has 80% power only for effects of about {mde:.1f} lb or larger; a P effect of 1-2 lb would not have been detected.\n")

# ----------------------------------------------------------------------------
# 5. Permutation test (labels permuted within blocks) for the main effects
# ----------------------------------------------------------------------------
def effect_stat(df, f_):
    return df.y[df[f_] == 1].mean() - df.y[df[f_] == 0].mean()


rows = []
for f_ in ["N", "P", "K"]:
    obs = effect_stat(d, f_)
    cnt = 0
    B = 5000
    for _ in range(B):
        dp = d.copy()
        dp[f_] = d.groupby("block")[f_].transform(lambda s_: rng.permutation(s_.values))
        cnt += abs(effect_stat(dp, f_)) >= abs(obs)
    rows.append({"factor": f_, "effect": obs, "permutation p (within blocks, 5,000 permutations)": cnt / B, "ANOVA p": eff.loc[f_, "p"]})
log("Randomisation tests (the treatment labels are permuted within blocks) compared with the ANOVA p-values")
perm_tab = pd.DataFrame(rows)
log(perm_tab.round(3).to_string(index=False))
disagree = perm_tab[(perm_tab["ANOVA p"] < 0.05) & (perm_tab.iloc[:, 2] >= 0.05)]["factor"].tolist()
log((f"For {', '.join(disagree)} the ANOVA gives p < 0.05 but the permutation test does not, so that effect is less certain than the ANOVA suggests "
     "(within-block permutations of a 4-plot block allow only a few arrangements, and the permutation test makes no normality assumption).\n") if disagree else "The two approaches agree on which effects are significant.\n")

# ----------------------------------------------------------------------------
# 6. Figures: main effects, interaction plot, normal plot of effects
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for f_ in "NPK":
    mm_ = d.groupby(f_).y.mean()
    axes[0].plot([0, 1], mm_.values, "o-", label=f_)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["low", "high"])
axes[0].set_ylabel("mean yield")
axes[0].legend()
axes[0].set_title("Main effects")
for lvl in (0, 1):
    axes[1].plot([0, 1], d[d.K == lvl].groupby("N").y.mean().values, "o-", label=f"K = {lvl}")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["N low", "N high"])
axes[1].legend()
axes[1].set_title("N x K interaction plot (parallel = no interaction)")
e_ = np.sort(eff["estimate"].values)
q = stats.norm.ppf((np.arange(1, len(e_) + 1) - 0.5) / len(e_))
axes[2].plot(q, e_, "o")
for qq, ee, nm in zip(q, e_, eff.sort_values("estimate").index):
    axes[2].annotate(nm, (qq, ee), fontsize=8, xytext=(3, 3), textcoords="offset points")
axes[2].set_xlabel("normal quantile")
axes[2].set_ylabel("effect estimate")
axes[2].set_title("Normal plot of effects")
fig.tight_layout()
fig.savefig(HERE / "figures" / "effects.png", dpi=150)
plt.close(fig)

log("Caveats: only 24 plots (12 per factor level) and 6 blocks, so little power for interactions; the three-factor interaction cannot be estimated because it is")
log("confounded with blocks; agricultural yield data from 1937 may not resemble modern experiments; the p-values for six effects are not adjusted for multiple")
log("testing; the effect sizes are for these soils and weather conditions, not general laws.")
out.close()
print("\nDone. See results.txt and figures/.")
