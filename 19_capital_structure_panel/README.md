# Project 19: What determines corporate leverage? Panel-data econometrics with Compustat

**Data:** Compustat North America annual fundamentals from WRDS, fiscal years 2000-2024. The data are licensed and are **not included**.
To reproduce: download the same variables from WRDS (screens `consol = C`, `indfmt = INDL`, `datafmt = STD`, `popsrc = D`, whole database) and save the
file as `data/compustat_funda.csv`. Variables: gvkey, datadate, fyear, curcd, fic, sic, at, ceq, dlc, dltt, ppent, oibdp, xrd, csho, prcc_f.

## Questions
1. Which firm characteristics are related to book leverage (debt / total assets): size, profitability, asset tangibility, market-to-book, R&D?
   Do the signs fit the trade-off and pecking-order theories?
2. How much do the estimates change with firm fixed effects?
3. Which specification is supported by a test, and how robust are the results?

## Sample
Starting from 288,127 firm-years (31,611 firms): fiscal years 2000-2024; US-incorporated firms reporting in USD; utilities and financial firms
excluded (SIC 49xx, 6xxx); total assets of at least USD 10 million; complete core variables; at least 3 consecutive-lag observations per firm.
**Final sample: 71,705 firm-years, 6,722 firms, 2001-2024 (10.7 years per firm).** Ratios are winsorised at the 1st and 99th percentile.
All regressors are lagged by one fiscal year. Every filter and its effect on the sample size is printed in `results.txt`.

## Method
Book leverage = (short-term + long-term debt) / total assets. Regressors: size (ln assets), profitability (EBITDA / assets), tangibility
(net PPE / assets), market-to-book (assets) and R&D / assets (missing R&D set to zero with a dummy). Four specifications: (1) pooled OLS,
(2) pooled OLS with year and 2-digit industry effects, (3) firm fixed effects, (4) firm and year fixed effects, all with firm-clustered
standard errors (`linearmodels`). A Mundlak test (the cluster-robust version of the Hausman test), an F-test of the firm effects, and
robustness checks.

## Results
| Coefficient on lagged variable | (1) Pooled | (2) + year, industry | (3) Firm FE | (4) Firm + year FE |
|---|---|---|---|---|
| Size (ln assets) | 0.029*** | 0.023*** | 0.050*** | 0.027*** |
| Profitability | -0.177*** | -0.139*** | -0.137*** | -0.115*** |
| Tangibility | 0.224*** | 0.247*** | 0.225*** | 0.209*** |
| Market-to-book | 0.003* | 0.002 | 0.000 | -0.001 |
| R&D / assets | -0.174*** | -0.148*** | -0.022 | -0.031 |
| R-squared | 0.128 | 0.194 | 0.062 (within) | 0.054 (within) |

*** p < 0.01, * p < 0.10; firm-clustered standard errors (see `results.txt`).

- **Three results are robust to every specification:** larger firms and firms with more tangible assets have more debt, and more profitable firms
  have less debt. The negative profitability coefficient (-0.115 to -0.177) is the classic result that fits the pecking-order theory and is hard to
  reconcile with the simplest trade-off theory.
- **Fixed effects matter for some coefficients, not others.** R&D is strongly negative in the pooled models but insignificant with firm fixed effects,
  so it reflects differences between firms rather than changes within a firm. The size coefficient moves from 0.029 to 0.050 with firm fixed effects
  and back to 0.027 once year effects are added.
- **The specification test favours fixed effects:** Mundlak test chi-squared(5) = 39.9, p < 0.001; the firm effects are jointly significant
  (F = 17.7, p < 0.001).
- **Economic size is modest.** A one-standard-deviation change within a firm changes leverage by about 1.6 percentage points for size (+), 1.4 for
  profitability (-) and 1.5 for tangibility (+), against a within-firm standard deviation of leverage of 14.4 points. The models explain only about 5%
  of the within-firm variation.
- **Robustness:** the signs for size, profitability and tangibility hold with market leverage as dependent variable, two-way clustered standard errors
  (firm and year), firms with assets of at least USD 100 million, and three subperiods (2000-2007, 2008-2015, 2016-2024). The tangibility
  coefficient is largest in 2016-2024 (0.297 vs 0.141-0.152 before); I did not test why. Market-to-book is not robustly related to book leverage
  (insignificant in the baseline), but negative with market leverage (-0.018, p < 0.01); that comparison is partly mechanical because market
  value of equity appears in both variables.

## Limitations
- Associations, not causal effects: leverage and profitability are determined jointly, time-varying omitted factors remain, and lagging the regressors
  only reduces simultaneity. Fixed effects remove only time-invariant firm differences.
- Static model: leverage adjusts only slowly to a target, so a partial-adjustment model would be natural, but fixed effects with a lagged dependent
  variable are biased in this setting (Nickell bias); that needs GMM or a bias correction.
- Compustat covers listed firms only; the filters (USD 10 million of assets, at least three observations per firm, winsorising) are choices and
  survivorship in the sample is not fully removed. Accounting changes (e.g. leases after 2019) affect book debt and book assets.

## Possible extensions
Partial-adjustment model with system GMM (Blundell-Bond), industry-median leverage as an additional determinant, quantile regression, effects of the
2008 crisis through interactions, and CRSP-Compustat merging for market-based measures.
