# Global Probabilistic Demand Forecasting Engine

Global (multi-series) probabilistic deep learning forecaster on the M5 /
Walmart retail dataset. Trains a shared DeepAR model across a ~2,001-series
subset of the 30,490 item x store series, outputs P10/P50/P90 quantile
forecasts, evaluates cold-start items with a leakage-checked holdout, and
translates forecast quality into a business (inventory cost) metric.

Full spec: [`demand_forecasting_project_spec.md`](demand_forecasting_project_spec.md).
Build status / phase log, every deviation and its reasoning:
[`PROGRESS.md`](PROGRESS.md). Full write-up:
[`reports/technical_writeup.md`](reports/technical_writeup.md).

> **Status:** All 7 phases complete (Phase 0 through Phase 6 — dashboard —
> plus this write-up/README pass). See `PROGRESS.md` for the phase-by-phase
> log and every deviation from the spec, with reasoning.

## Architecture

```
Raw Data (M5 CSVs)
   -> Ingestion & Preprocessing (schema normalization, missing values, time split)
   -> Feature Engineering (calendar, lags, static embeddings, cold-start flag)
   -> Model Layer (baselines: naive/ETS/Prophet | global: DeepAR, quantile output)
   -> Backtesting Harness (rolling-origin, segmented metrics)
   -> Cold-Start Holdout (leakage-tested, evaluated separately from warm-start)
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

Runs ingestion -> baselines -> DeepAR (Phase 3) -> cold-start retrain +
leakage test (Phase 4) -> business simulation (Phase 5) -> dashboard data
refresh (Phase 6), in order, writing every report to `reports/`. Each of the
three DeepAR training runs takes a few minutes on a single consumer GPU
(RTX 2070, 8GB) and early-stops around epoch 4 — see `PROGRESS.md` for the
scope-reduction reasoning (subset size, history window) behind why this
runs in minutes rather than hours.

### Dashboard

The dashboard is interactive, so it's started separately from `run.sh`:

```bash
streamlit run dashboard/app.py
```

Reads only already-committed reports (`reports/dashboard_series_forecasts.csv`,
`reports/phase4_coldstart_results.csv`, `reports/business_sim_results.csv`) —
runs from a fresh clone with no GPU or Kaggle download needed, as long as
those reports exist (they're committed, unlike `data/raw`/`data/processed`).

## Repo structure

```
├── run.sh                   # single command, end to end (excl. raw data download)
├── requirements.txt
├── data/
│   ├── raw/                 # gitignored — populate via the manual step above
│   └── processed/           # gitignored, parquet
├── src/
│   ├── ingestion/            # Phase 1: load + build train/val/test parquet
│   ├── features/              # calendar, price, lags, static, cold-start
│   ├── models/
│   │   ├── baselines.py      # seasonal naive / ETS / Prophet (Phase 2)
│   │   └── global_model.py   # DeepAR feature pipeline + dataset builder
│   ├── backtest/
│   │   ├── harness.py            # rolling-origin backtester (Phase 2)
│   │   ├── metrics.py            # WQL, MASE
│   │   ├── run_baselines.py      # Phase 2 orchestration
│   │   ├── run_deepar.py         # Phase 3 orchestration
│   │   ├── run_deepar_coldstart.py  # Phase 4 orchestration
│   │   └── run_business_sim.py      # Phase 5 orchestration
│   └── business_sim/
│       └── simulate.py       # order-up-to-P90/P50 policy simulation
├── notebooks/                # exploration only, never the source of truth
├── dashboard/
│   ├── app.py                # Streamlit app (3 required views)
│   └── prepare_data.py       # consolidates reports for the dashboard
├── reports/                  # committed — results tables + write-up
│   └── technical_writeup.md
└── tests/
    ├── test_no_leakage.py    # cold-start leakage test (Phase 4)
    └── ...                   # 44 tests total across the pipeline
```

## Results

**Baselines vs. global model, warm-start (val split, days 1886-1913):**

| model | n_series | population | WQL | MASE |
|---|---|---|---|---|
| seasonal_naive | 30,490 | full universe | 0.582 | 1.189 |
| ets | 100 | stratified subset | **0.472** | **0.976** |
| prophet | 100 | same subset | 1.035 | 1.387 |
| deepar (global) | 2,001 | stratified subset | 0.482 | 1.214 |
| lightgbm (global, bonus) | 2,001 | same subset as deepar | **0.467** | **0.862** |

DeepAR essentially ties ETS on WQL and doesn't clearly beat the baselines
on warm-start series — documented as an honest result, not hidden (see
`reports/technical_writeup.md` §4 for the full discussion of why, and where
a global model's advantage is expected to show up instead).

**Bonus, post-completion:** a global LightGBM model (`src/models/lightgbm_model.py`),
evaluated on the exact same rolling-origin harness and 2,001-series
population as DeepAR, comes out ahead on both WQL and MASE — consistent
with M5's own competition results (the winning solutions were all
LightGBM-based). Not wired into `run.sh`; full comparison and the
"horizon-as-feature" design in `PROGRESS.md`'s Bonus section and
`reports/phase3_comparison_with_lightgbm.csv`.

**Cold-start vs. warm-start (same DeepAR model, retrained with a
leakage-tested cold-start holdout — 100 series with <28 days of visible
history):**

| segment | n_series | WQL | MASE |
|---|---|---|---|
| overall | 2,001 | 0.466 | 1.196 |
| cold_start | 100 | 0.460 | 0.867 |
| warm_start | 1,901 | 0.466 | 1.213 |

Cold-start WQL is essentially tied with warm-start — evidence the model's
static/category embeddings carry real signal for brand-new items. See
`reports/technical_writeup.md` §5 for the honest caveat on the MASE
comparison specifically.

**Business simulation (order-up-to-P90 vs. order-up-to-P50, test split,
same DeepAR forecast for both policies):**

| policy | stockout rate | total holding cost |
|---|---|---|
| P90 | 1.90% | $1,542 |
| P50 | 23.86% | $259 |

P90 cuts stockouts ~12.5x at ~6x the holding cost — the safety-stock
trade-off, measured on real forecasts. See `reports/technical_writeup.md`
§6 for the holding-cost assumption and full discussion.

## Defend this decision

**Why WQL over RMSE?** RMSE evaluates a point forecast and implicitly
assumes symmetric, equally-costly errors — exactly the assumption an
inventory decision violates (a stockout and excess stock are not
equally costly, and the "right" order quantity is a quantile of the demand
distribution, not its mean). Weighted Quantile Loss (pinball loss, averaged
across P10/P50/P90) scores the whole predicted distribution against
realized demand at each quantile level, which is both the metric this
project's models are actually optimized to produce output for, and the one
that connects directly to the business layer: the P90 forecast fed into the
inventory simulation *is* the same P90 WQL is scoring.

**Why DeepAR over TFT?** Both are legitimate choices (TFT is cited in
`reports/technical_writeup.md` and considered explicitly, not omitted).
DeepAR was chosen because it trains faster on this hardware, its
autoregressive quantile output is more directly comparable to the
classical baselines (all four models here produce the same
`[id, d_num, q0.1, q0.5, q0.9]` shape), and it doesn't require the
additional architectural complexity (attention, variable selection
networks) TFT needs for its interpretability advantage — which this
project's Definition of Done doesn't require. TFT's asymmetric
encoder/decoder feature support (it can use encoder-only features DeepAR
architecturally cannot) is a real advantage TFT has and DeepAR doesn't;
noted as a concrete follow-up in the write-up's Limitations section rather
than downplayed.

**How was the cold-start holdout done without leakage?** ~5% of the DeepAR
training population (100 series, stratified by category) have every row
before a cutoff day deleted *before* any feature engineering or dataset
windowing happens — not filtered only at the final evaluation step. Because
the deletion happens upstream of everything else, no downstream training or
evaluation window can ever contain a pre-cutoff timestep for those series,
by construction. This is verified programmatically, not just by design:
`tests/test_no_leakage.py` builds an actual `TimeSeriesDataSet` (the real
class the model trains on), iterates every batch its training dataloader
can produce, and asserts the reconstructed encoder start day for each
cold-start series never precedes its cutoff. Two non-obvious library
behaviors that would have silently broken this (a min-encoder-length filter
that silently drops short series, and a target-lag feature that deletes a
cold-start series' entire visible history) were found by testing before the
real run and are documented with the exact fix in `PROGRESS.md` Phase 4.
