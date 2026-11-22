"""
Project 16 - Two typical chemistry data analyses: (A) enzyme kinetics and (B) chemometrics of wine

(A) Enzyme kinetics. Data: R's `Puromycin` data (Treloar 1974): reaction velocity (counts/min^2) of an enzymatic reaction at 6-7
    substrate concentrations (ppm), in cells treated with puromycin and untreated cells; 23 observations.
    Questions: estimate the Michaelis-Menten parameters Vmax and Km, with honest uncertainty; does puromycin change Vmax, Km, or both?
    Is the old Lineweaver-Burk linearisation as good as direct nonlinear least squares?
(B) Chemometrics. Data: the classic wine data (Forina et al.): 178 wines from 3 cultivars of one Italian region, 13 chemical
    measurements (alcohol, flavanoids, proline, ...); loaded from scikit-learn.
    Questions: how many principal components describe the chemistry? Which wines are outliers (Hotelling T2, Q residuals)?
    Can cultivars be classified (LDA, PLS-DA, SVM, kNN) with honest cross-validation, and which variables matter (PLS VIP)?

Run:  python analysis.py      (needs numpy pandas scipy scikit-learn matplotlib; Puromycin CSV in ../data/puromycin.csv)
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
from scipy.optimize import curve_fit
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")
rng = np.random.default_rng(1)


def log(*a):
    print(*a)
    print(*a, file=out)


# =============================================================================
# PART A: Michaelis-Menten kinetics
# =============================================================================
log("=" * 70)
log("PART A: enzyme kinetics (Puromycin data)")
log("=" * 70)
pu = pd.read_csv(HERE.parent / "data" / "puromycin.csv")
pu["trt"] = (pu.state == "treated").astype(int)
log(f"{len(pu)} observations: treated {pu.trt.sum()}, untreated {(1 - pu.trt).sum()}; concentrations {[float(c) for c in sorted(pu.conc.unique())]} ppm\n")


def mm(c, vmax, km):
    return vmax * c / (km + c)


def fit_group(d):
    p, cov = curve_fit(mm, d.conc, d.rate, p0=[d.rate.max(), 0.1])
    res = d.rate - mm(d.conc, *p)
    return p, cov, (res ** 2).sum(), len(d) - 2


rows = []
fits = {}
for name, d in pu.groupby("state"):
    p, cov, rss, dof = fit_group(d)
    se = np.sqrt(np.diag(cov))
    t = stats.t.ppf(0.975, dof)
    fits[name] = (p, rss, dof, d)
    # residual bootstrap for the confidence intervals (small samples: asymptotic intervals can be too narrow)
    fitted, resid = mm(d.conc.values, *p), (d.rate - mm(d.conc, *p)).values
    boot = []
    for _ in range(2000):
        yb = fitted + rng.choice(resid, len(resid), replace=True)
        try:
            boot.append(curve_fit(mm, d.conc.values, yb, p0=p, maxfev=2000)[0])
        except Exception:
            pass
    boot = np.array(boot)
    rows.append({"state": name, "n": len(d), "Vmax": p[0], "Vmax 95% CI (asymptotic)": f"[{p[0] - t * se[0]:.0f}, {p[0] + t * se[0]:.0f}]",
                 "Vmax 95% CI (bootstrap)": f"[{np.percentile(boot[:, 0], 2.5):.0f}, {np.percentile(boot[:, 0], 97.5):.0f}]",
                 "Km": p[1], "Km 95% CI (asymptotic)": f"[{p[1] - t * se[1]:.3f}, {p[1] + t * se[1]:.3f}]",
                 "Km 95% CI (bootstrap)": f"[{np.percentile(boot[:, 1], 2.5):.3f}, {np.percentile(boot[:, 1], 97.5):.3f}]"})
tab = pd.DataFrame(rows).set_index("state")
tab["Vmax"] = tab["Vmax"].round(1)
tab["Km"] = tab["Km"].round(3)
log("Separate Michaelis-Menten fits (rate = Vmax * conc / (Km + conc)); Vmax in counts/min^2, Km in ppm")
log(tab.T.to_string(), "\n")

# Nested models fitted jointly: does puromycin change Vmax, Km, or both?
t_ = pu.trt.values
c_ = pu.conc.values
y_ = pu.rate.values
models = {
    "M0: no effect (common Vmax, Km)": (lambda c, v, k: mm(c, v, k), [200, 0.1], 2),
    "M1: only Vmax differs": (lambda X, v0, dv, k: mm(X[0], v0 + dv * X[1], k), [200, 50, 0.1], 3),
    "M2: only Km differs": (lambda X, v, k0, dk: mm(X[0], v, k0 + dk * X[1]), [200, 0.1, 0.0], 3),
    "M3: both differ": (lambda X, v0, dv, k0, dk: mm(X[0], v0 + dv * X[1], k0 + dk * X[1]), [200, 50, 0.1, 0.0], 4),
}
rss = {}
for name, (f, p0, k) in models.items():
    X = c_ if name.startswith("M0") else (c_, t_)
    p, _ = curve_fit(f, X, y_, p0=p0, maxfev=20000)
    rss[name] = (((y_ - f(X, *p)) ** 2).sum(), k, p)
n = len(y_)
rows = []
for name, (r, k, p) in rss.items():
    rows.append({"model": name, "parameters": k, "RSS": round(r, 0), "AIC": round(n * np.log(r / n) + 2 * k, 1)})
log("Joint models (all 23 observations)")
log(pd.DataFrame(rows).to_string(index=False))


def f_test(small, big):
    r0, k0, _ = rss[small]
    r1, k1, _ = rss[big]
    F = ((r0 - r1) / (k1 - k0)) / (r1 / (n - k1))
    return F, 1 - stats.f.cdf(F, k1 - k0, n - k1)


for a, b in [("M0: no effect (common Vmax, Km)", "M1: only Vmax differs"), ("M1: only Vmax differs", "M3: both differ"), ("M2: only Km differs", "M3: both differ")]:
    F, p = f_test(a, b)
    log(f"F-test {a.split(':')[0]} vs {b.split(':')[0]}: F = {F:.2f}, p = {p:.4f}")
log("")

# Lineweaver-Burk (double reciprocal) for comparison
rows = []
for name, d in pu.groupby("state"):
    sl, ic, *_ = stats.linregress(1 / d.conc, 1 / d.rate)
    vmax_lb, km_lb = 1 / ic, sl / ic
    rows.append({"state": name, "Vmax (Lineweaver-Burk)": round(vmax_lb, 1), "Km (Lineweaver-Burk)": round(km_lb, 3),
                 "Vmax (nonlinear LS)": round(fits[name][0][0], 1), "Km (nonlinear LS)": round(fits[name][0][1], 3)})
log("Lineweaver-Burk linearisation versus direct nonlinear least squares")
log(pd.DataFrame(rows).to_string(index=False))
log("The double-reciprocal transform magnifies the error at low concentrations, so its estimates are less reliable; nonlinear LS is preferred.\n")

fig, ax = plt.subplots(figsize=(6, 4))
cg = np.linspace(0, 1.15, 200)
for name, color in [("treated", "C0"), ("untreated", "C1")]:
    p, _, _, d = fits[name]
    ax.plot(d.conc, d.rate, "o", color=color, label=f"{name} (data)")
    ax.plot(cg, mm(cg, *p), "-", color=color, label=f"{name} fit")
ax.set_xlabel("substrate concentration (ppm)")
ax.set_ylabel("rate (counts/min$^2$)")
ax.legend(fontsize=8)
ax.set_title("Michaelis-Menten fits")
fig.tight_layout()
fig.savefig(HERE / "figures" / "michaelis_menten.png", dpi=150)
plt.close(fig)

# =============================================================================
# PART B: chemometrics of wine
# =============================================================================
log("=" * 70)
log("PART B: chemometrics (wine data)")
log("=" * 70)
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
names = list(wine.target_names)
log(f"{X.shape[0]} wines, {X.shape[1]} chemical variables, cultivars: {', '.join(f'{n_}: {int(c)}' for n_, c in zip(names, np.bincount(y)))}")
Z = StandardScaler().fit_transform(X)  # autoscaling is essential: variables have very different units (proline ~ 700, hue ~ 1)
log(f"Variance ranges over a factor of {X.var().max() / X.var().min():.0f} before scaling (proline dominates unscaled data).\n")

pca = PCA().fit(Z)
cum = np.cumsum(pca.explained_variance_ratio_) * 100
log("PCA on autoscaled data: cumulative explained variance (%) for 1..5 components: " + ", ".join(f"{v:.1f}" for v in cum[:5]))
k = 2
scores = pca.transform(Z)[:, :k]
load = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"]).round(2)
log("Largest loadings: PC1 " + ", ".join(load.PC1.abs().sort_values(ascending=False).index[:3]) + "; PC2 " + ", ".join(load.PC2.abs().sort_values(ascending=False).index[:3]))
log(f"Adjusted Rand index of cultivar vs 3-cluster k-means in the full space: {adjusted_rand_score(y, KMeans(3, n_init=20, random_state=0).fit_predict(Z)):.3f}; "
    f"Ward hierarchical: {adjusted_rand_score(y, AgglomerativeClustering(3).fit_predict(Z)):.3f}\n")

# Outlier diagnostics with a 3-component PCA model: Hotelling T2 (inside the model) and Q (distance from the model)
K = 3
pc = PCA(K).fit(Z)
T = pc.transform(Z)
t2 = ((T / np.sqrt(pc.explained_variance_)) ** 2).sum(1)
q = ((Z - pc.inverse_transform(T)) ** 2).sum(1)
nn = len(Z)
t2_lim = K * (nn - 1) / (nn - K) * stats.f.ppf(0.95, K, nn - K)
q_lim = np.percentile(q, 95)  # empirical limit (no distributional assumption)
out_idx = np.where((t2 > t2_lim) | (q > q_lim))[0]
log(f"Outlier diagnostics (3-component PCA model): T2 limit (95%, F-distribution) = {t2_lim:.1f}, Q limit (empirical 95th percentile) = {q_lim:.1f}")
log(f"Wines above the T2 limit: {(t2 > t2_lim).sum()}, above the Q limit: {(q > q_lim).sum()}, either: {len(out_idx)} (about 5% each is expected by construction)")
extreme = np.argsort(-t2)[:3]
log("Most extreme by T2 (row, T2, cultivar): " + ", ".join(f"({i}, {t2[i]:.1f}, {names[y[i]]})" for i in extreme) + "\n")

# Classification with repeated stratified cross-validation; scaling inside the pipeline
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)


class PLSDA(ClassifierMixin, BaseEstimator):
    """PLS-DA: PLS regression on one-hot class indicators, predicted class = largest fitted value."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X_, y_):
        self.classes_ = np.unique(y_)
        self.pls_ = PLSRegression(self.n_components, scale=False).fit(X_, (np.asarray(y_)[:, None] == self.classes_).astype(float))
        return self

    def predict(self, X_):
        return self.classes_[self.pls_.predict(X_).argmax(1)]


rows = []
for nm, est in [("LDA", make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
                ("PLS-DA (2 components)", make_pipeline(StandardScaler(), PLSDA(2))),
                ("SVM (RBF)", make_pipeline(StandardScaler(), SVC())),
                ("kNN (k = 5)", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
                ("kNN (k = 5) WITHOUT scaling", KNeighborsClassifier(5))]:
    s = cross_val_score(est, X, y, cv=cv)
    rows.append({"classifier": nm, "mean CV accuracy": round(s.mean(), 3), "std over folds": round(s.std(), 3)})
log("Cultivar classification, 5-fold cross-validation repeated 10 times")
log(pd.DataFrame(rows).to_string(index=False))
log("(The unscaled kNN shows why autoscaling matters in chemometrics: proline's large numbers dominate the distance.)\n")

# PLS-DA VIP scores (variable importance in projection), fitted on all data for interpretation
pls = PLSRegression(2, scale=False).fit(Z, np.eye(3)[y])
W, Tm, Qm = pls.x_weights_, pls.x_scores_, pls.y_loadings_
ss = (Qm ** 2).sum(0) * (Tm ** 2).sum(0)
vip = np.sqrt(Z.shape[1] * ((W / np.linalg.norm(W, axis=0)) ** 2 @ ss) / ss.sum())
vip = pd.Series(vip, index=X.columns).sort_values(ascending=False)
log("PLS-DA VIP scores (VIP > 1 = more important than average): top 6")
log(vip.head(6).round(2).to_string())
log(f"{(vip > 1).sum()} of {len(vip)} variables have VIP > 1.\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for c, nm in enumerate(names):
    axes[0].scatter(scores[y == c, 0], scores[y == c, 1], s=14, label=nm)
axes[0].scatter(scores[out_idx, 0], scores[out_idx, 1], s=70, facecolors="none", edgecolors="k", label="T2/Q outlier")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f}%)")
axes[0].legend(fontsize=7)
axes[0].set_title("PCA scores, autoscaled wine data")
vip.sort_values().plot.barh(ax=axes[1])
axes[1].axvline(1, color="k", ls="--", lw=0.8)
axes[1].set_title("PLS-DA VIP scores")
fig.tight_layout()
fig.savefig(HERE / "figures" / "pca_and_vip.png", dpi=150)
plt.close(fig)

log("Caveats: (A) only 23 observations and no replicate structure; a heteroskedastic error (variance growing with rate) is plausible and was")
log("not modelled; confidence intervals come from asymptotic and residual-bootstrap methods that can be inaccurate at this sample size.")
log("(B) the wines all come from one region and three cultivars, so classification accuracy will not transfer to other settings; the outlier")
log("limits are approximate; PLS components were fixed at 2 rather than tuned; VIP is a heuristic.")
out.close()
print("\nDone. See results.txt and figures/.")
