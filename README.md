# Systematic Option Strategies

Backtesting framework for systematic option strategies on SPY (2020 to 2022): four classic overlays plus variance risk premium (VRP) harvesting.

> **Data.** The real market option data (SPY, QQQ, AAPL end-of-day chains) is **not included in this repository**. To obtain it, please contact me by opening an issue on this repository. A **synthetic** SPY option chain (`data/spy_simulated_2020_2022.csv`) is provided so that everything runs offline; see [`data/README.md`](data/README.md).

## Strategies

| Strategy | Type | Bias | Cost |
|---|---|---|---|
| **Buy-Write** | Income | Long, capped upside | Credit (call premium) |
| **Protective Put** | Hedging | Long, protected | Debit (put premium) |
| **Collar** | Hedging | Neutral to long | Near zero-cost |
| **Long Straddle** | Volatility | Direction-neutral | Debit (call + put) |
| **Short Straddle (VRP)** | Short volatility | Direction-neutral | Credit |
| **Synthetic variance swap (VRP)** | Short variance | Direction-neutral | Replicated with OTM options weighted by $1/K^2$ |

The variance swap leg follows the log-contract replication (Demeterfi et al., 1999; CBOE VIX methodology):

$$K_{\text{var}} = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2}\, e^{rT} O(K_i), \qquad \text{P\&L}_T = N\left(\sigma_{\text{RV}}^2 - K_{\text{var}}\right).$$

## Structure

```
├── data/
│   ├── spy_simulated_2020_2022.csv   # synthetic option chain (included)
│   └── README.md                     # real datasets: not included, available on request
├── src/
│   ├── data_loader.py                # loading & cleaning
│   ├── generate_simulated_data.py    # synthetic option chain generator
│   ├── strategy_buywrite.py
│   ├── strategy_protective_put.py
│   ├── strategy_collar.py
│   ├── strategy_straddle.py
│   ├── strategy_vrp.py               # short straddle + synthetic variance swap
│   ├── metrics.py
│   └── plotting.py
├── notebooks/                        # 01 buy-write … 05 comparison, 06 VRP
├── docs/
│   ├── Option-Strat.pdf              # course notes (FR): pricing, Greeks, structures, VRP (synthetic data)
│   ├── SystematicOptionStrat.tex     # strategy report source
│   └── SystematicOptionStrat.pdf     # strategy report
├── outputs/                          # generated charts
├── build_notebooks.py, build_vrp_notebook.py
└── run_all.py
```

## Quickstart

```bash
pip install -r requirements.txt

# synthetic data (included)
python run_all.py data/spy_simulated_2020_2022.csv

# regenerate the synthetic chain
python src/generate_simulated_data.py

# real data (not included, available on request), once placed in data/
python run_all.py data/spy_2020_2022.csv
```

In the notebooks, switch the source with the `DATA_PATH` variable in the first cell.

## Parameters

| Parameter | Buy-Write | Protective Put | Collar | Straddle |
|---|---|---|---|---|
| DTE target | ~7 | ~30 | ~30 | ~30 |
| Strike (call) | ATM ($\Delta \approx 0.50$) | | OTM ($\Delta \approx 0.25$) | ATM ($\Delta \approx 0.50$) |
| Strike (put) | | OTM ($\Delta \approx -0.30$) | OTM ($\Delta \approx -0.25$) | ATM ($\Delta \approx -0.50$) |
| Execution | Mid | Mid | Mid | Mid |

## Data format

Expected CSV columns (bracketed headers are stripped automatically):

```
QUOTE_DATE, QUOTE_TIME_HOURS, UNDERLYING_LAST, EXPIRE_DATE, DTE,
C_DELTA, C_BID, C_ASK, C_IV, STRIKE,
P_DELTA, P_BID, P_ASK, P_IV
```

## References

- CBOE BXM Index methodology (buy-write benchmark)
- Black & Scholes (1973), option pricing
- Whaley (2002), return and risk of the CBOE buy-write monthly index
- Demeterfi, Derman, Kamal & Zou (1999), more than you ever wanted to know about volatility swaps
- Dörries, Korn & Power (2022), how should the long-term investor harvest variance risk premiums?
- Lhabitant (2004), hedge funds: quantitative insights

## License

MIT, see [LICENSE](LICENSE).
