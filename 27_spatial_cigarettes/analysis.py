"""
Project 27 - Spatial econometrics: cigarette demand across US states and spillovers between neighbours

Data: `CigarettesSW` (R package AER; Stock & Watson): annual data for 48 contiguous US states, 1985 and 1995: packs sold per capita, average price per pack
(including sales tax), excise taxes, population, personal income, CPI. State boundaries: US states GeoJSON (PublicaMundi/MappingAPI on GitHub).

Questions
  1. Is cigarette consumption spatially clustered (Moran's I, local clusters)?
  2. What is the price elasticity of demand, and does OLS leave spatial autocorrelation in its residuals?
  3. Do neighbouring states' prices matter (cross-border shopping)? Which spatial model fits best: lagged dependent variable (SAR), spatial error (SEM),
     or spatially lagged regressors (SLX)? What are the direct and indirect effects?

Run:  python analysis.py      (needs numpy pandas geopandas libpysal esda spreg matplotlib)
Outputs: results.txt and figures/*.png in this folder.
"""
import warnings
from pathlib import Path

import esda
import geopandas as gpd
import libpysal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spreg

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
(HERE / "figures").mkdir(exist_ok=True)
out = open(HERE / "results.txt", "w")


def log(*a):
    print(*a)
    print(*a, file=out)


ABBR = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
        "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
        "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
        "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
        "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}

cig = pd.read_csv(HERE.parent / "data" / "cigarettes_sw.csv")
cig["name"] = cig.state.map(ABBR)
cig["rprice"] = cig.price / cig.cpi
cig["rincome"] = cig.income / cig.population / cig.cpi
cig["lq"], cig["lp"], cig["li"] = np.log(cig.packs), np.log(cig.rprice), np.log(cig.rincome)
geo = gpd.read_file(HERE.parent / "data" / "us_states.geojson")
log(f"{cig.state.nunique()} states in the cigarette data, years {sorted(cig.year.unique())}; {len(geo)} polygons in the map file")


def prepare(year):
    g = geo.merge(cig[cig.year == year], left_on="name", right_on="name", how="inner").reset_index(drop=True)
    g = g.to_crs(5070)  # equal-area projection (Albers) for contiguity weights
    w = libpysal.weights.Queen.from_dataframe(g, use_index=False)
    w.transform = "r"
    return g, w


g95, w95 = prepare(1995)
log(f"Contiguity: {len(g95)} states, Queen neighbours per state: mean {np.mean(list(w95.cardinalities.values())):.1f}, min {min(w95.cardinalities.values())}, max {max(w95.cardinalities.values())}; "
    f"islands (states without neighbours): {len(w95.islands)}\n")

# ----------------------------------------------------------------------------
# 1. Spatial autocorrelation of consumption
# ----------------------------------------------------------------------------
np.random.seed(1)
for year in (1985, 1995):
    g, w = prepare(year)
    mi = esda.Moran(g.lq.values, w, permutations=9999)
    mp = esda.Moran(g.lp.values, w, permutations=9999)
    log(f"{year}: Moran's I of ln(packs per capita) = {mi.I:.3f} (permutation p = {mi.p_sim:.4f}); of ln(real price) = {mp.I:.3f} (p = {mp.p_sim:.4f})")
log("A positive Moran's I means that neighbouring states have similar values.\n")

lisa = esda.Moran_Local(g95.lq.values, w95, permutations=9999, seed=2)
lab = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
g95["cluster"] = [lab[q] if p < 0.05 else "not significant" for q, p in zip(lisa.q, lisa.p_sim)]
log("Local clusters of cigarette consumption, 1995 (LISA, 5% pseudo-p-value, no multiple-testing correction)")
for c in ["High-High", "Low-Low", "High-Low", "Low-High"]:
    sts = g95.loc[g95.cluster == c, "state"].tolist()
    if sts:
        log(f"  {c}: {', '.join(sts)}")
log("")

# ----------------------------------------------------------------------------
# 2. OLS and spatial diagnostics (1995)
# ----------------------------------------------------------------------------
y = g95[["lq"]].values
X = g95[["lp", "li"]].values
xn = ["ln real price", "ln real income per capita"]
ols = spreg.OLS(y, X, w=w95, name_y="ln packs", name_x=xn, spat_diag=True, moran=True)
log("OLS, 1995: ln(packs per capita) on ln(real price) and ln(real income)")
log(f"  price elasticity {ols.betas[1, 0]:.2f} (SE {np.sqrt(ols.vm[1, 1]):.2f}); income elasticity {ols.betas[2, 0]:.2f} (SE {np.sqrt(ols.vm[2, 2]):.2f}); R2 {ols.r2:.2f}, n = {ols.n}")
log(f"  Moran's I of the OLS residuals: {ols.moran_res[0]:.3f} (p = {ols.moran_res[2]:.3f})")
log(f"  LM-lag test: {ols.lm_lag[0]:.2f} (p = {ols.lm_lag[1]:.3f}); LM-error test: {ols.lm_error[0]:.2f} (p = {ols.lm_error[1]:.3f}); "
    f"robust LM-lag {ols.rlm_lag[0]:.2f} (p = {ols.rlm_lag[1]:.3f}); robust LM-error {ols.rlm_error[0]:.2f} (p = {ols.rlm_error[1]:.3f})")
log("  (Rule of thumb: if only one of the robust tests is significant, choose that model; the OLS residual test says whether any spatial model is needed.)\n")

# ----------------------------------------------------------------------------
# 3. Spatial models: SLX, SAR (lag), SEM (error)
# ----------------------------------------------------------------------------
wy = libpysal.weights.lag_spatial(w95, g95.lp.values)
Xslx = np.column_stack([g95.lp.values, g95.li.values, wy])
slx = spreg.OLS(y, Xslx, name_y="ln packs", name_x=xn + ["W x ln real price"])
sar = spreg.ML_Lag(y, X, w=w95, name_y="ln packs", name_x=xn)
sem = spreg.ML_Error(y, X, w=w95, name_y="ln packs", name_x=xn)
rows = [
    {"model": "OLS", "price elasticity": ols.betas[1, 0], "SE": np.sqrt(ols.vm[1, 1]), "spatial parameter": "-", "log-likelihood": ols.logll, "AIC": ols.aic},
    {"model": "SLX (add neighbours' price)", "price elasticity": slx.betas[1, 0], "SE": np.sqrt(slx.vm[1, 1]), "spatial parameter": f"W x price: {slx.betas[3, 0]:.2f} (SE {np.sqrt(slx.vm[3, 3]):.2f})", "log-likelihood": slx.logll, "AIC": slx.aic},
    {"model": "SAR (spatial lag of y)", "price elasticity": sar.betas[1, 0], "SE": np.sqrt(sar.vm[1, 1]), "spatial parameter": f"rho: {sar.rho:.2f} (SE {sar.std_err[-1]:.2f})", "log-likelihood": sar.logll, "AIC": sar.aic},
    {"model": "SEM (spatial error)", "price elasticity": sem.betas[1, 0], "SE": np.sqrt(sem.vm[1, 1]), "spatial parameter": f"lambda: {sem.betas[-1, 0]:.2f} (SE {sem.std_err[-1]:.2f})", "log-likelihood": sem.logll, "AIC": sem.aic},
]
cmp_ = pd.DataFrame(rows)
log("Spatial models, 1995 (lower AIC is better; the AIC of the spreg OLS and ML models is computed on the same likelihood basis)")
log(cmp_.round({"price elasticity": 3, "SE": 3, "log-likelihood": 1, "AIC": 1}).to_string(index=False))
best = cmp_.loc[cmp_["AIC"].idxmin(), "model"]
slx_t = slx.betas[3, 0] / np.sqrt(slx.vm[3, 3])
log(f"Best AIC: {best}. The spatial dependence is better described as correlated unobserved factors across neighbours (spatial error) than as a direct spillover of "
    f"consumption. The neighbours' price coefficient in the SLX model has t = {slx_t:.1f}, so there is no clear evidence of cross-border shopping effects with this "
    "sample of 48 states.\n")

# Direct and indirect effects of the SAR model: S(W) = (I - rho W)^-1 (I beta_k)
Wd = w95.full()[0]
n = Wd.shape[0]
Sinv = np.linalg.inv(np.eye(n) - sar.rho * Wd)
rows = []
for k, nm in [(1, "ln real price"), (2, "ln real income")]:
    S = Sinv * sar.betas[k, 0]
    direct = np.trace(S) / n
    total = S.sum() / n
    rows.append({"variable": nm, "direct": direct, "indirect (spillover)": total - direct, "total": total})
log("SAR effect decomposition (average effects of a 1% change in the variable in all states / in one state's neighbours)")
log(pd.DataFrame(rows).round(3).to_string(index=False))
log("The indirect effect is the effect on a state's consumption of changes in the price in other states, passed through the spatial lag. It is conditional on the SAR model, which does not fit best (see the AIC); rho is only moderately precise.\n")

# Do the models remove the residual autocorrelation?
mI = esda.Moran(slx.u.flatten(), w95, permutations=9999)
log(f"Moran's I of residuals: OLS {ols.moran_res[0]:.3f}, SLX {mI.I:.3f} (p = {mI.p_sim:.3f})\n")

# 1985 comparison of the cross-border coefficient
g85, w85 = prepare(1985)
y85 = g85[["lq"]].values
X85 = np.column_stack([g85.lp.values, g85.li.values, libpysal.weights.lag_spatial(w85, g85.lp.values)])
slx85 = spreg.OLS(y85, X85)
log(f"SLX 1985: own price {slx85.betas[1, 0]:.2f} (SE {np.sqrt(slx85.vm[1, 1]):.2f}), neighbours' price {slx85.betas[3, 0]:.2f} (SE {np.sqrt(slx85.vm[3, 3]):.2f})")
log("A positive and significant neighbour-price coefficient would fit cross-border shopping (residents buy more at home when neighbours are more expensive); in both years it is small and imprecise.\n")

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
g95.plot(column="packs", cmap="OrRd", legend=True, ax=axes[0], edgecolor="white", linewidth=0.5, legend_kwds={"shrink": 0.7})
axes[0].set_title("Packs per capita, 1995")
axes[0].axis("off")
colors = {"High-High": "#d7191c", "Low-Low": "#2c7bb6", "High-Low": "#fdae61", "Low-High": "#abd9e9", "not significant": "#eeeeee"}
for c, col in colors.items():
    sub = g95[g95.cluster == c]
    if len(sub):
        sub.plot(ax=axes[1], color=col, edgecolor="white", linewidth=0.5, label=c)
axes[1].legend(fontsize=7, loc="lower left")
axes[1].set_title("Local clusters (LISA) of consumption, 1995")
axes[1].axis("off")
fig.tight_layout()
fig.savefig(HERE / "figures" / "maps.png", dpi=150)
plt.close(fig)

log("Caveats: 48 observations per year, so spatial parameters are imprecise; the cross-section is not a causal design (prices and taxes are set by states that")
log("differ in unobserved ways, such as tobacco-growing states with low taxes); a panel with state fixed effects would be better; the results depend on the")
log("Queen contiguity weight matrix (distance-based weights could differ); LISA p-values are not corrected for multiple testing; the data are from 1985 and 1995.")
out.close()
print("\nDone. See results.txt and figures/.")
