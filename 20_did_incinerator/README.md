# Project 20: Difference-in-differences - did a waste incinerator lower nearby house prices?

**Data:** `kielmc` (Kiel and McClain 1995, in Wooldridge's textbook data): 321 house sales in North Andover, Massachusetts, 179 in 1978 and 142 in 1981 (different
houses in each year: repeated cross-sections). 96 houses are within 3 miles of the site of an incinerator that was being planned and built in this period (treated
group). Prices are in 1978 dollars. Download with `download_public_data.py`.

## Questions
1. What is the difference-in-differences (DiD) estimate of the incinerator's effect on nearby prices?
2. Is it robust to controls, to the functional form, and to the method of inference?
3. How credible is the identifying assumption with only two periods?

## Method
2x2 table of mean prices; OLS DiD regressions in levels and logs, with and without house characteristics (age, age squared, distance to the interstate, land area,
house area, rooms, baths) and HC3 standard errors; cluster-robust errors by neighbourhood (only 7 clusters); a randomisation test (2,000 random reassignments of
the "near" label); a continuous-distance version; checks of sample composition.

## Results
- Mean price rose by about USD 18.8 thousand for far houses (82.5 to 101.3) but only 6.9 for near houses (63.7 to 70.6): a **DiD of -USD 11.9 thousand**.
- **All four specifications are negative but imprecise:** -USD 11,900 (p = 0.18) without controls, **-USD 14,200 (p = 0.0496) with controls**, and in logs -6.3% (p = 0.51)
  and -9.3% (p = 0.15; 95% CI -0.22 to +0.03 log points). Only one of four reaches p < 0.05, and only barely.
- Other inference methods agree that the evidence is moderate: randomisation p = 0.067; clustered by neighbourhood p = 0.14 (7 clusters, not reliable).
- The continuous distance version shows no clear gradient (p = 0.54).
- **Composition changed:** near houses were already USD 18.8 thousand cheaper in 1978 and are much older (40 vs 13 years in 1978) with smaller plots; the age of houses
  sold also fell between years (21 to 14 years). This is why controls matter, and why this is a repeated cross-section, not a panel.

My honest reading: the data are consistent with a price drop of roughly 5-10% near the incinerator, but they cannot rule out zero.

## Limitations
- Only two periods, so parallel pre-trends cannot be tested; the 1978 "before" year may already reflect anticipation of the incinerator.
- The treatment is defined by an arbitrary 3-mile threshold; 321 observations; no spillover analysis; too few neighbourhoods for cluster-robust inference.
- Different houses are sold each year, so selection into sale could differ by group.

## Possible extensions
Wild-cluster bootstrap, matching or reweighting on house characteristics, a synthetic control or event-study design with more periods (not available in this data set),
a sensitivity analysis for violations of parallel trends (Rambachan-Roth).
