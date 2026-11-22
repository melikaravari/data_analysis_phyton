# Data files expected by projects 1-12 and 19


| File | Content | Rows | Period | Projects | Source |
|---|---|---|---|---|---|
| `data/brent.csv` | Daily Brent crude oil price (columns `date`, `BRENT`) | 4,317 | 2010-01-01 to 2026-07-20 | 1, 8, 10 | TODO |
| `data/energy.csv` | Daily WTI, Henry Hub gas and jet fuel prices (columns `date`, `WTI`, `HH`, `JETFUEL`) | 4,317 | 2010-01-01 to 2026-07-20 | 1, 8, 10 | TODO |
| `data/stock_indices.csv` | Daily levels of four US equity indices (columns `date`, `SPX`, `DJI`, `NDX`, `RUT`) | 4,945 | 2000-01-03 to 2019-08-28 | 2, 6, 7, 12 | TODO |
| `data/vix.csv` | Daily VIX index (columns `date`, `vix`) | 7,767 | 1990-01-02 to 2020-10-30 | 2, 7, 12 | TODO |
| `data/power_DE.csv` | Hourly German electricity price in EUR/MWh (columns `timestamp`, `price`) | 66,232 | 2019-01-01 to 2026-07-22 | 3, 9 | TODO |
| `data/sp500_prices.csv` | Daily prices, one column per ticker (about 700), first column `datadate` | 479 | 2019-01-02 to 2020-11-12 | 4, 11 | TODO |
| `data/x_sim.csv`, `data/y_sim.csv` | Simulated trajectory (6 columns) and noisy position measurements (2 columns); whitespace-separated, no header | 69 | n/a | 5 | TODO |
| `data/compustat_funda.csv` | Compustat North America annual fundamentals from WRDS (see project 19 README for the variables and screens) | about 288,000 | fiscal years 2000-2024 | 19 | WRDS / Compustat (licensed, do not publish) |
