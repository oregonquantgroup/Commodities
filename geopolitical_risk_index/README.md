[GPRD_Gold_Strategy_README.md](https://github.com/user-attachments/files/27327190/GPRD_Gold_Strategy_README.md)
# GPRD-Gold Trading Strategy

---

## Overview

This repository implements a quantitative trading strategy that uses the **Geopolitical Risk Daily (GPRD) index** as a signal to time long/flat positions in gold. The pipeline ingests GPRD and price data, constructs a release-aware rolling Z-score signal, runs a state machine to generate positions, backtests against gold and S&P 500 benchmarks, and produces statistical validation outputs and Excel reporting artifacts.

---

## Current Capabilities

- Load GPRD index, gold price, and S&P 500 data from a structured Excel workbook.
- Aggregate daily GPRD readings to weekly averages (Monday-anchored) to reduce noise.
- Build a **release-aware 2-year rolling Z-score signal** that strictly avoids look-ahead bias by anchoring each week's calculation to data available through the prior Wednesday.
- Apply configurable Buy/Sell/Hold thresholds to raw Z-scores.
- Pass raw signals through a **state machine** that maintains position continuity and prevents whipsaw.
- Backtest strategy returns against gold buy-and-hold and S&P 500 buy-and-hold, indexed to a base of 100.
- Compute full-period and sub-period performance metrics: total return, CAGR, Sharpe ratio, max drawdown, and time-in-market.
- Run **Welch t-tests** on 5-day forward log returns grouped by signal type (Buy / Sell / Hold).
- Compute directional accuracy (one-tailed binomial test) and Cohen's d power analysis per signal pair.
- Generate a 3-panel matplotlib chart: rolling Sharpe, cumulative log returns, and strategy value vs. GPRD moving average.
- Write a rolling 1-year Sharpe Excel sheet with an embedded line chart via openpyxl.
- Export a full daily results CSV covering all signals, returns, positions, and portfolio values.

---

## Repository Layout

```
gprd_gold_strategy.py   # Single-file pipeline: data loading through output generation
```

### Function Reference

```
load_data()              # Reads GPRD, gold, and S&P 500 sheets from Excel workbook
weekly_gprd()            # Aggregates daily GPRD to Monday-anchored weekly averages
build_signals()          # Constructs release-aware rolling Z-score and raw signals
state_machine()          # Converts raw signals to held positions via Buy/Sell logic
backtest()               # Merges positions with returns; computes strategy and benchmark series
metrics()                # Calculates performance statistics for a given result window
subperiod()              # Slices results for in-sample and out-of-sample period analysis
stat_tests()             # Welch t-tests, directional accuracy, and power analysis
plot_charts()            # Generates 3-panel matplotlib figure
rolling_sharpe_sheet()   # Writes rolling Sharpe Excel output with embedded chart
main()                   # Orchestrates full pipeline end-to-end
```

---

## Data Pipeline

1. **Data loading:** `load_data()` reads three sheets from `The_Finale_v2.xlsx` — GPRD index, gold closing prices, and S&P 500 levels. Raw data downloadable from Yahoo Finance and matteoiacoviello.com.
2. **Weekly aggregation:** `weekly_gprd()` averages daily GPRD readings to the week to smooth noise before signal construction.
3. **Signal construction:** `build_signals()` computes a 504-day rolling Z-score for each trading day using only data available through the Wednesday prior to that week, then applies Buy/Sell/Hold thresholds.
4. **State machine:** `state_machine()` converts raw weekly signals into daily held positions — Buy opens long, Sell closes to flat, Hold maintains current position.
5. **Backtest:** `backtest()` merges positions with gold and S&P 500 returns, computes daily strategy returns, and builds cumulative value series indexed to 100.
6. **Statistical validation:** `stat_tests()` runs Welch t-tests and directional accuracy tests on 5-day forward log returns by signal group.

### Data Artifacts

| File | Description |
|---|---|
| `The_Finale_v2.xlsx` | Source workbook (required input) |
| `strategy_results.csv` | Full daily dataset: signals, returns, positions, values |
| `The_Finale_v2_output.xlsx` | Rolling 1-year Sharpe sheet with embedded line chart |
| `The_Finale_v2_final.xlsx` | Copy of output Excel file |
| `strategy_charts.png` | 3-panel performance figure |

---

## Setup

Install dependencies:

```bash
pip install pandas numpy scipy matplotlib openpyxl
```

Ensure the source workbook is present at one of:

```
/mnt/user-data/uploads/The_Finale_v2.xlsx
~/Downloads/The_Finale_v2.xlsx
```

The workbook must contain three sheets:

| Sheet | Required Columns | Description |
|---|---|---|
| `GPRD2008-Present` | Date, DOW, GPRD | Daily GPRD index readings |
| `Gold_data` | Date, Gold_Close | Daily gold closing prices |
| `S&P500` | Date, SP500_Level, SP500_1d_return | Daily S&P 500 data |

---

## Usage

```bash
python gprd_gold_strategy.py
```

All outputs are written to the same directory as the source Excel file. No CLI arguments are required — signal parameters can be adjusted directly in `build_signals()`:

| Parameter | Default | Description |
|---|---|---|
| `rw` | `504` | Rolling window length in trading days (~2 years) |
| `buy_t` | `-0.2` | Z-score threshold below which a Buy signal fires |
| `sell_t` | `1.0` | Z-score threshold above which a Sell signal fires |
| `rf` | `0.0274` | Annual risk-free rate used in Sharpe calculations |

---

## Known Limitations

- Signal fires weekly due to lack of live GPRD data, limiting the signal's capability to respond to sudden mid-week shocks.
- Signal generation requires a minimum of 504 trading days of GPRD history before the first valid Z-score, meaning the live strategy period begins around 2010 regardless of earlier data availability.
- Buy vs. Hold has not reached statistical significance in testing; results should be interpreted with caution and not treated as validation of the Buy signal in isolation.
- The strategy trades only long/flat in gold with no short exposure, which limits the actionability of Sell signals beyond exiting positions.
