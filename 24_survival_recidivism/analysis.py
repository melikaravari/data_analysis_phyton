"""
Project 24 - Survival analysis: does financial aid after release from prison delay re-arrest? (Rossi et al. 1980 experiment)

Data: `rossi` (from the lifelines package, Rossi, Berk & Lenihan 1980): 432 men released from Maryland state prisons and followed for 52 weeks.
Half were randomly assigned to receive financial aid (USD 60 per week for 52 weeks, `fin = 1`). Variables: weeks until first re-arrest (`week`) and
whether a re-arrest occurred (`arrest`; otherwise the follow-up is censored at 52 weeks), age at release, race (1 = black), prior work experience (`wexp`),
married (`mar`), released on parole (`paro`), number of prior convictions (`prio`).

Because the aid was randomly assigned, a simple comparison of the two groups estimates the causal effect of the offer of aid (intention to treat).

Questions
  1. How does the aid change the time to re-arrest (Kaplan-Meier curves, log-rank test, restricted mean survival time)?
  2. What is the hazard ratio in a Cox model, and how does it change when other covariates are included? Is the proportional-hazards assumption met?
  3. Does the effect differ by subgroup? How well does the model predict (cross-validated concordance)?

Run:  python analysis.py      (needs numpy pandas scipy lifelines matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.datasets import load_rossi
from lifelines.statistics import logrank_test, proportional_hazard_test
from lifelines.utils import k_fold_cross_validation, restricted_mean_survival_time

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")
rng = np.random.default_rng(5)


def log(*a):
    print(*a)
    print(*a, file=out)


d = load_rossi()
log(f"{len(d)} men; {int(d.arrest.sum())} re-arrested within 52 weeks ({d.arrest.mean() * 100:.0f}%), {int((1 - d.arrest).sum())} censored at 52 weeks ({(1 - d.arrest.mean()) * 100:.0f}%)")
log(f"Financial aid group: {int(d.fin.sum())}; control group: {int((1 - d.fin).sum())}\n")

# ----------------------------------------------------------------------------
# 1. Randomisation check and Kaplan-Meier
# ----------------------------------------------------------------------------
bal = d.groupby("fin")[["age", "race", "wexp", "mar", "paro", "prio"]].mean().round(2)
log("Balance of covariates by treatment group (randomisation should balance them)")
log(bal.to_string(), "\n")

kmf = {g: KaplanMeierFitter().fit(s.week, s.arrest, label="aid" if g else "no aid") for g, s in d.groupby("fin")}
lr = logrank_test(d.week[d.fin == 1], d.week[d.fin == 0], d.arrest[d.fin == 1], d.arrest[d.fin == 0])
log(f"Share still not re-arrested at 52 weeks: aid {kmf[1].predict(52) * 100:.1f}%, no aid {kmf[0].predict(52) * 100:.1f}% "
    f"(difference {(kmf[1].predict(52) - kmf[0].predict(52)) * 100:.1f} percentage points)")
log(f"Log-rank test: chi2 = {lr.test_statistic:.2f}, p = {lr.p_value:.3f}")

rm1, rm0 = restricted_mean_survival_time(kmf[1], t=52), restricted_mean_survival_time(kmf[0], t=52)
diffs = []
for _ in range(1000):
    b = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
    k1 = KaplanMeierFitter().fit(b.week[b.fin == 1], b.arrest[b.fin == 1])
    k0 = KaplanMeierFitter().fit(b.week[b.fin == 0], b.arrest[b.fin == 0])
    diffs.append(restricted_mean_survival_time(k1, t=52) - restricted_mean_survival_time(k0, t=52))
log(f"Restricted mean time without re-arrest over the 52 weeks: aid {rm1:.1f} weeks, no aid {rm0:.1f} weeks; difference {rm1 - rm0:.2f} weeks "
    f"(bootstrap 95% CI {np.percentile(diffs, 2.5):.2f} to {np.percentile(diffs, 97.5):.2f})\n")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for g in (0, 1):
    kmf[g].plot_survival_function(ax=ax, ci_show=True)
ax.set_xlabel("weeks since release")
ax.set_ylabel("share not yet re-arrested")
ax.set_ylim(0.6, 1.0)
ax.set_title("Kaplan-Meier curves (Rossi experiment)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "kaplan_meier.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 2. Cox proportional hazards models
# ----------------------------------------------------------------------------
cox0 = CoxPHFitter().fit(d[["week", "arrest", "fin"]], "week", "arrest")
cox1 = CoxPHFitter().fit(d, "week", "arrest")
rows = []
for name, m in [("Aid only", cox0), ("With covariates", cox1)]:
    s = m.summary.loc["fin"]
    rows.append({"model": name, "hazard ratio (aid)": s["exp(coef)"], "95% CI": f"[{s['exp(coef) lower 95%']:.2f}, {s['exp(coef) upper 95%']:.2f}]", "p": s["p"]})
log("Cox models: hazard ratio of re-arrest for the aid group (values below 1 = lower hazard)")
log(pd.DataFrame(rows).round({"hazard ratio (aid)": 2, "p": 3}).to_string(index=False))
s = cox1.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3)
s.columns = ["hazard ratio", "lower", "upper", "p"]
log("\nCox model with covariates")
log(s.to_string())
log(f"Concordance (in-sample) {cox1.concordance_index_:.3f}; likelihood-ratio test of the whole model: p < 0.001\n" if cox1.log_likelihood_ratio_test().p_value < 0.001 else
    f"Concordance (in-sample) {cox1.concordance_index_:.3f}\n")
log("Randomisation makes the two hazard ratios for aid similar: covariates do not confound the effect but add precision. Each additional prior conviction raises the hazard, ")
log("and being older at release lowers it.\n")

# Proportional-hazards assumption (scaled Schoenfeld residuals)
ph = proportional_hazard_test(cox1, d, time_transform="rank")
log("Test of the proportional-hazards assumption (Schoenfeld residuals; p < 0.05 = evidence the effect changes over time)")
log(ph.summary.round(3).to_string())
viol = [k for k, v in ph.summary["p"].items() if v < 0.05]
log(f"Covariates with evidence against proportional hazards (p < 0.05): {', '.join(viol) if viol else 'none'}. The treatment variable (fin) shows no sign of non-proportionality. "
    "I did not stratify or add time interactions for the others, so their hazard ratios are averages over the follow-up.\n")

# ----------------------------------------------------------------------------
# 3. Subgroup effects, parametric alternative, predictive performance
# ----------------------------------------------------------------------------
rows = []
for var, label in [("prio", "prior convictions (per conviction)"), ("age", "age at release (per year)"), ("wexp", "prior work experience"), ("paro", "on parole")]:
    di = d.copy()
    di["inter"] = di.fin * di[var]
    m = CoxPHFitter().fit(di, "week", "arrest")
    rows.append({"effect modifier": label, "interaction aid x modifier (log hazard)": m.params_["inter"], "p": m.summary.loc["inter", "p"]})
log("Does the effect of the aid differ by subgroup? Interactions added one at a time to the Cox model")
log(pd.DataFrame(rows).round(3).to_string(index=False))
log("Four tests were run; a single small p-value among them would not be convincing, and none is a pre-specified hypothesis.\n")

aft = WeibullAFTFitter().fit(d, "week", "arrest")
tr = np.exp(aft.params_.loc[("lambda_", "fin")])
ci = np.exp(aft.confidence_intervals_.loc[("lambda_", "fin")])
log(f"Weibull AFT model: time ratio for aid = {tr:.2f} (95% CI [{ci.iloc[0]:.2f}, {ci.iloc[1]:.2f}]): aid multiplies the time to re-arrest by this factor; "
    f"Weibull shape = {np.exp(aft.params_.loc[('rho_', 'Intercept')]):.2f} (> 1 means the hazard rises over time).")
log("(Only 26% of the men are re-arrested within the follow-up, so the model says little about the time after week 52.)\n")

cv = k_fold_cross_validation(CoxPHFitter(), d, duration_col="week", event_col="arrest", k=5, scoring_method="concordance_index", seed=1)
log(f"5-fold cross-validated concordance of the Cox model: {np.mean(cv):.3f} (std {np.std(cv):.3f}); 0.5 = no discrimination, 1 = perfect. The model ranks")
log("who is re-arrested earlier only modestly better than chance.\n")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
cox1.plot(ax=ax, hazard_ratios=True)
ax.set_title("Hazard ratios (95% CI)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "hazard_ratios.png", dpi=150)
plt.close(fig)

log("Caveats: the aid group received an offer, so these are intention-to-treat effects (not everyone may have received or used the aid); the follow-up is only")
log("one year, with 74% censored; re-arrest is a measure of detected crime, not of offending; the 1970s Maryland sample of men may not generalise; the original")
log("study also recorded weekly employment, which is not in this version and would allow an analysis of the mechanism (aid may reduce employment).")
out.close()
print("\nDone. See results.txt and figures/.")
