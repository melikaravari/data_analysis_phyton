# Project 4: S&P 500 panel - cleaning, PCA and minimum-variance portfolio

**Data:** daily prices for 708 tickers, 2019-01-02 to 2020-11-12 (479 days), 9.9% missing cells.

## Questions
1. How to clean a messy wide price panel before computing returns?
2. How much co-movement is explained by a few common factors?
3. Does a minimum-variance portfolio have lower out-of-sample risk than equal weighting, including the 2020 crash?

## Method
**Cleaning:** drop tickers with "." in the name (176: preferred shares, warrants, share classes), drop tickers
with more than 5% missing prices (37), forward-fill gaps of up to 3 days, compute log returns, check for
implausible returns. Result: 495 tickers, 478 returns.
**PCA** on standardised returns. **Portfolio:** long-only minimum variance with a 5% cap per stock and a
Ledoit-Wolf shrinkage covariance matrix, estimated on 2019 (255 days), evaluated on 2020-01 to 2020-11 (223 days)
with static weights.

## Results
- **PC1 explains 49% of return variance** and has a correlation of 0.998 with the equally weighted market
  return, so it is the market factor. The first five components together explain 63%.
- **Correlations rise in crises:** PC1 explains 26.8% of the variance before the COVID crash (to 2020-02-18) and
  66.5% between 2020-02-19 and 2020-04-30, so diversification weakens exactly when it is needed.
- **Minimum-variance portfolio** (62 stocks with non-trivial weight): annualised volatility 29.0% versus
  41.4% for equal weight, max drawdown -31.3% versus -41.5%, worst day -9.6% versus -12.8%.
  The risk reduction did not come with a better return: annualised return was -4.9% versus -4.2%, so return/vol
  is slightly worse (-0.17 versus -0.10). The result is a risk statement, not a performance claim.
- Cleaning check: six returns above 50% remain, mostly energy stocks on 2020-03-09 (real: oil price crash) and
  one odd security (BMYMP, a non-standard instrument) that the ticker rules did not catch.

## Limitations
- Survivorship bias (only stocks present in the file), a single stress event in the test window, no
  transaction costs, static weights and a single estimation window.
- The 5% cap and the 1-year window are arbitrary choices; I did not tune or test sensitivity to them.
- Ticker rules are heuristic; a proper security master would identify share classes and instrument types.

## Possible extensions
Rolling re-estimation and rebalancing, factor models with Fama-French data, sensitivity analysis of the cap and window.
