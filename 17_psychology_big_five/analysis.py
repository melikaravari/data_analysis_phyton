"""
Project 17 - Measuring personality: reliability and factor structure of a Big Five questionnaire (psychometrics)

Data: `bfi` from the R package `psych` (Revelle): 2,800 volunteers answered 25 personality items on a 1-6 scale
(5 items each for Agreeableness A, Conscientiousness C, Extraversion E, Neuroticism N, Openness O), plus gender
(1 = male, 2 = female), education (1-5) and age. Source file: data/bfi.csv (downloaded from the Rdatasets collection).

Questions
  1. Are the scales reliable (Cronbach's alpha), and do the reverse-keyed items behave as documented?
  2. How many factors does the data support (parallel analysis, BIC), and does the intended five-factor structure appear?
  3. How do the scale scores differ by gender, and relate to age and education, and how large are these effects?

Run:  python analysis.py      (needs numpy pandas scipy scikit-learn statsmodels matplotlib)
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
from sklearn.decomposition import FactorAnalysis
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")
rng = np.random.default_rng(7)


def log(*a):
    print(*a)
    print(*a, file=out)


raw = pd.read_csv(HERE.parent / "data" / "bfi.csv", index_col=0)
scales = {s: [f"{s}{i}" for i in range(1, 6)] for s in "ACENO"}
items = [c for s in scales.values() for c in s]
log(f"{len(raw)} respondents, {len(items)} items. Item missingness: min {raw[items].isna().mean().min() * 100:.1f}%, max {raw[items].isna().mean().max() * 100:.1f}%")
complete = raw.dropna(subset=items)
log(f"Respondents with all 25 items: {len(complete)} ({len(complete) / len(raw) * 100:.0f}%). Gender: {int((raw.gender == 1).sum())} men, {int((raw.gender == 2).sum())} women; age {raw.age.min()}-{raw.age.max()}, median {raw.age.median():.0f}\n")

# ----------------------------------------------------------------------------
# 1. Reverse-keyed items: detect from the data (negative item-rest correlation) and compare with the documentation
# ----------------------------------------------------------------------------
documented = {"A1", "C4", "C5", "E1", "E2", "O2", "O5"}
detected = set()
for s_, cols in scales.items():
    Rm = raw[cols].corr().values
    w, v = np.linalg.eigh(Rm)
    pc1 = v[:, -1]
    if (pc1 > 0).sum() < (pc1 < 0).sum():  # orient so that the majority of items load positively
        pc1 = -pc1
    detected |= {c for c, l in zip(cols, pc1) if l < 0}
log(f"Items loading in the minority direction on the first principal component of their scale (data-driven): {sorted(detected)}")
log(f"Documented reverse-keyed items:                                         {sorted(documented)}")
log(f"Identical: {detected == documented}  (the first component of each scale separates forward- from reverse-keyed items)\n")
df = raw.copy()
for c in documented:
    df[c] = 7 - df[c]  # scale 1-6

# ----------------------------------------------------------------------------
# 2. Reliability: Cronbach's alpha, mean inter-item correlation, alpha if item dropped
# ----------------------------------------------------------------------------
def alpha(X):
    X = X.dropna()
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))


rows = []
for s, cols in scales.items():
    X = df[cols].dropna()
    cm = X.corr().values[np.triu_indices(5, 1)].mean()
    drop = {c: alpha(X.drop(columns=c)) for c in cols}
    best = max(drop, key=drop.get)
    rows.append({"scale": s, "n": len(X), "alpha": round(alpha(X), 2), "mean inter-item r": round(cm, 2),
                 "alpha if worst item dropped": round(drop[best], 2), "item": best})
rel = pd.DataFrame(rows)
log("Reliability of the five scales (alpha around 0.7+ is conventionally acceptable for research use)")
log(rel.to_string(index=False), "\n")

# ----------------------------------------------------------------------------
# 3. Dimensionality: parallel analysis and BIC of ML factor models
# ----------------------------------------------------------------------------
Xc = complete[items].copy()
for c in documented:
    Xc[c] = 7 - Xc[c]
R = np.corrcoef(Xc.values, rowvar=False)
ev = np.linalg.eigvalsh(R)[::-1]
n, p = Xc.shape
sim = np.array([np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False))[::-1] for _ in range(200)])
thr = np.percentile(sim, 95, axis=0)
n_pa = int((ev > thr).sum())
log(f"Parallel analysis (Horn): {n_pa} eigenvalues exceed the 95th percentile of random data. First eigenvalues: " + ", ".join(f"{v:.2f}" for v in ev[:8]))

Z = (Xc - Xc.mean()) / Xc.std()
rows = []
for k in range(1, 9):
    fa = FactorAnalysis(k, random_state=0, max_iter=2000).fit(Z)
    ll = fa.score(Z) * n
    npar = p * k - k * (k - 1) / 2 + p
    rows.append({"factors": k, "log-likelihood": round(ll, 0), "BIC": round(-2 * ll + npar * np.log(n), 0)})
bic = pd.DataFrame(rows)
log("Maximum-likelihood factor models (lower BIC is better)")
log(bic.to_string(index=False))
log(f"BIC picks {int(bic.loc[bic.BIC.idxmin(), 'factors'])} factors; parallel analysis picks {n_pa}. BIC tends to favour more factors in large samples.\n")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(range(1, p + 1), ev, "o-", label="observed eigenvalues")
ax.plot(range(1, p + 1), thr, "k--", label="95th percentile, random data")
ax.axhline(1, color="grey", lw=0.6)
ax.set_xlabel("component")
ax.set_ylabel("eigenvalue")
ax.set_xlim(0, 15)
ax.legend()
ax.set_title("Scree plot with parallel analysis")
fig.tight_layout()
fig.savefig(HERE / "figures" / "scree_parallel.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4. Five-factor model: does the intended structure appear? (varimax rotation)
# ----------------------------------------------------------------------------
fa5 = FactorAnalysis(5, rotation="varimax", random_state=0, max_iter=2000).fit(Z)
L = pd.DataFrame(fa5.components_.T, index=items, columns=[f"F{i + 1}" for i in range(5)])
dom = L.abs().idxmax(axis=1)
scale_factor = {s: dom[cols].mode().iloc[0] for s, cols in scales.items()}
correct = sum(dom[c] == scale_factor[s] for s, cols in scales.items() for c in cols)
log("Five-factor solution (ML, varimax): factor that carries most of each scale's items")
log("  " + ", ".join(f"{s} -> {scale_factor[s]}" for s in scales) + f"; distinct factors used: {len(set(scale_factor.values()))}")
log(f"Items whose largest loading is on the factor of their intended scale: {correct} of 25")
cross = [(c, f, round(float(L.loc[c, f]), 2)) for c in items for f in L.columns if f != dom[c] and abs(L.loc[c, f]) > 0.3]
log(f"Cross-loadings above 0.30 in absolute value: {cross if cross else 'none'}")
mis = [(c, dom[c], round(float(L.loc[c, dom[c]]), 2)) for s, cols in scales.items() for c in cols if dom[c] != scale_factor[s]]
log(f"Items not on their scale's factor: {mis if mis else 'none'}")
log("Share of variance explained by the five factors: %.0f%%\n" % (100 * (L ** 2).sum().sum() / p))

fig, ax = plt.subplots(figsize=(5.5, 7))
order = [c for s in scales.values() for c in s]
im = ax.imshow(L.loc[order].values, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
ax.set_yticks(range(25))
ax.set_yticklabels(order, fontsize=7)
ax.set_xticks(range(5))
ax.set_xticklabels(L.columns)
fig.colorbar(im, ax=ax)
ax.set_title("Factor loadings (varimax)")
fig.tight_layout()
fig.savefig(HERE / "figures" / "loadings.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. Scale scores, group differences and associations
# ----------------------------------------------------------------------------
for s, cols in scales.items():
    ok = df[cols].notna().sum(axis=1) >= 4
    df[f"score_{s}"] = df[cols].mean(axis=1).where(ok)
sc = [f"score_{s}" for s in scales]
log("Correlations between scale scores (traits are not independent; this is why an oblique rotation would be more natural):")
log(df[sc].corr().round(2).rename(index=lambda x: x[-1], columns=lambda x: x[-1]).to_string(), "\n")

rows = []
for s in scales:
    a = df.loc[df.gender == 1, f"score_{s}"].dropna()
    b = df.loc[df.gender == 2, f"score_{s}"].dropna()
    t, pval = stats.ttest_ind(b, a, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    d = (b.mean() - a.mean()) / sp
    se = np.sqrt((len(a) + len(b)) / (len(a) * len(b)) + d ** 2 / (2 * (len(a) + len(b))))
    rows.append({"scale": s, "mean men": a.mean(), "mean women": b.mean(), "Cohen d (women - men)": d, "95% CI": f"[{d - 1.96 * se:.2f}, {d + 1.96 * se:.2f}]", "p (Welch)": pval})
g = pd.DataFrame(rows)
g["p (Holm)"] = multipletests(g["p (Welch)"], method="holm")[1]
g = g.round({"mean men": 2, "mean women": 2, "Cohen d (women - men)": 2})
g["p (Welch)"] = g["p (Welch)"].map(lambda v: f"{v:.2g}")
g["p (Holm)"] = g["p (Holm)"].map(lambda v: f"{v:.2g}")
log("Gender differences (Cohen's d: 0.2 small, 0.5 medium, 0.8 large)")
log(g.to_string(index=False), "\n")

rows = []
for s in scales:
    for v in ("age", "education"):
        d_ = df[[f"score_{s}", v]].dropna()
        r, pv = stats.spearmanr(d_[f"score_{s}"], d_[v])
        rows.append({"scale": s, "variable": v, "Spearman r": round(r, 2), "p": f"{pv:.2g}", "n": len(d_)})
log("Associations of scale scores with age and education (Spearman)")
log(pd.DataFrame(rows).pivot(index="scale", columns="variable", values="Spearman r").to_string(), "\n")

log("Caveats: volunteers in an online survey are not a random sample; items are ordinal (1-6) and treated as continuous (polychoric")
log("correlations would be better); varimax forces uncorrelated factors although traits correlate; no confirmatory factor analysis")
log("and no test of measurement invariance across gender, so mean differences between groups assume the scales measure the same")
log("thing in both groups; the Openness scale is known to be weaker; cross-sectional data, so age differences are not development.")
out.close()
print("\nDone. See results.txt and figures/.")
