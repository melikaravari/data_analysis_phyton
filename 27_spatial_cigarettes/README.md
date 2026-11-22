# Project 27: Spatial econometrics - cigarette demand across US states

**Data:** `CigarettesSW` (R package AER; Stock and Watson): 48 contiguous US states, 1985 and 1995: packs sold per capita, price, taxes, population, income, CPI; state boundaries from
a US states GeoJSON file (PublicaMundi/MappingAPI on GitHub). Download with `download_public_data.py`.

## Questions
1. Is cigarette consumption spatially clustered?
2. What is the price elasticity, and does OLS leave spatial autocorrelation in the residuals?
3. Do neighbours' prices matter (cross-border shopping)? Which spatial model fits best?

## Method
Queen-contiguity weights (libpysal); Moran's I with permutation inference and local clusters (LISA, esda); OLS with spatial diagnostics (LM tests); SLX, spatial lag (SAR) and spatial error
(SEM) models (spreg); AIC comparison; direct and indirect effects of the SAR model; maps.

## Results
- **Consumption is clustered:** Moran's I of log packs per capita is 0.26 in 1985 (p = 0.003) and 0.29 in 1995 (p = 0.001). High-high clusters in 1995: KY, ME, OH, TN, VA; low-low clusters: the
  western states AZ, CO, ID, NV, NM, OR, UT.
- **OLS price elasticity -1.41** (SE 0.25), income elasticity 0.34; the OLS residuals are still spatially correlated (Moran's I 0.29, p = 0.001), so OLS standard errors are unreliable.
- **Spatial error model fits best** (AIC -29.0 vs -21.7 OLS, -23.9 SAR, -20.1 SLX; lambda = 0.47, SE 0.15), in line with the robust LM tests (error p = 0.10, lag p = 0.95).
  Spatial dependence therefore looks like correlated unobserved factors across neighbouring states.
- **No clear evidence of cross-border shopping:** the neighbours' price coefficient in the SLX model is 0.24 (SE 0.40) in 1995 and -0.06 (SE 0.48) in 1985.
- The SAR model (rho = 0.29, SE 0.15) implies a direct price effect of -1.32 and an indirect (spillover) effect of -0.50; this is conditional on a model that does not fit best.

## Limitations
- 48 observations per year; a cross-section is not a causal design (states differ in unobservables such as tobacco production); a panel with state fixed effects would be better.
- Results depend on the Queen weights; LISA p-values are not corrected for multiple testing; 1985 and 1995 data only.

## Possible extensions
Spatial panel models, distance-based weights, a spatial Durbin model, regional data for Germany (for example Eurostat NUTS-3 data) with the same methods.
