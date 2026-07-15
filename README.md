# Global Probabilistic Demand Forecasting Engine

Global (multi-series) probabilistic deep learning forecaster on the M5 / Walmart
retail dataset. Trains a shared DeepAR model across ~42,840 item×store series,
outputs P10/P50/P90 quantile forecasts, evaluates cold-start items with a
leakage-checked holdout, and translates forecast quality into a business
(inventory cost) metric.

Full spec: [`demand_forecasting_project_spec.md`](demand_forecasting_project_spec.md).
Build status / phase log: [`PROGRESS.md`](PROGRESS.md).

> **Status:** Phase 0 (scaffold) complete. Results table and architecture
> diagram polish land in Phase 7 — this README is a working draft until then.

## Architecture

```
Raw Data (M5 CSVs)
   -> Ingestion & Preprocessing (schema normalization, missing values, time split)
   -> Feature Engineering (calendar, lags, static embeddings, cold-start flag)
   -> Model Layer (baselines: naive/ETS/ARIMA/Prophet | global: DeepAR, quantile output)
   -> Backtesting Harness (rolling-origin, segmented metrics)
   -> Business Translation Layer (stockout/holding cost simulation)
   -> Reporting (Streamlit dashboard + technical write-up)
```

## Setup

Requires Python 3.11 (pinned — see §3 of the spec and the Phase 0 deviation
note in `PROGRESS.md` if your system default differs).

```powershell
py -3.11 -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### Data download (manual, one-time)

The M5 dataset requires Kaggle authentication, so this is a documented manual
step rather than part of `run.sh`:

```powershell
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/
Expand-Archive data/raw/m5-forecasting-accuracy.zip -DestinationPath data/raw/
```

Files needed in `data/raw/`: `sales_train_evaluation.csv`, `calendar.csv`,
`sell_prices.csv`.

### Run the pipeline

```bash
./run.sh
```

(Currently a stub — stages are filled in phase by phase; see `PROGRESS.md`.)

## Repo structure

```
├── run.sh                   # single command, end to end (stub for now)
├── requirements.txt
├── data/
│   ├── raw/                 # gitignored — populate via the manual step above
│   └── processed/           # gitignored, parquet
├── src/
│   ├── ingestion/
│   ├── features/
│   ├── models/
│   │   ├── baselines.py
│   │   └── global_model.py  # DeepAR
│   ├── backtest/
│   │   └── harness.py
│   └── business_sim/
├── notebooks/                # exploration only, never the source of truth
├── dashboard/                # Streamlit app
├── reports/
│   └── technical_writeup.md
└── tests/
    └── test_no_leakage.py
```

## Results

_To be filled in as baselines (Phase 2) and the global model (Phase 3) land._

## Defend this decision

_To be written in Phase 7 (spec §10.7) — will cover WQL vs. RMSE, DeepAR vs.
TFT, and the cold-start leakage-prevention methodology._
