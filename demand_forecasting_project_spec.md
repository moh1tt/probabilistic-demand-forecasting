# PROJECT SPEC: Global Probabilistic Demand Forecasting Engine (Retail / M5)

> **Instructions for Claude Code (or any AI coding agent):** This is the single source of truth for this project. Before starting any phase, re-read the relevant section below. Do not skip acceptance criteria. Do not start Phase N+1 until Phase N's acceptance criteria are met and checked off. If you are unsure which phase you're on, check `PROGRESS.md` in the repo root (create it in Phase 0 and update it after every phase).

---

## 0. Project Identity

- **Name:** `global-demand-forecaster`
- **Owner:** Mohit Appari
- **Purpose:** Portfolio project for Applied/Data Scientist roles at Amazon, Walmart, Target, Costco, NVIDIA, Microsoft, Google.
- **Domain:** Retail demand forecasting (M5 / Walmart dataset). Single domain — no cross-domain energy port in v1 (cut per timeline risk; see §11 Stretch Goals if time allows).
- **Core claim this project must prove:** "I can build a global (multi-series) probabilistic deep learning forecaster that beats per-series statistical baselines, explicitly handles cold-start items, and translates forecast quality into a business (inventory cost) metric — not just RMSE."

---

## 1. Non-Negotiable Success Criteria (Definition of Done)

The project is NOT done until all of these are true. Treat this as the checklist to return to when scope-creep tempts a detour:

1. [ ] Baselines (naive, ETS, per-series ARIMA/Prophet) run on M5 with rolling-origin backtesting and reported metrics.
2. [ ] Global deep model (DeepAR or TFT) trained jointly across all series, outputs P10/P50/P90 quantiles.
3. [ ] Global model beats baselines on WQL for warm-start series (or, if it doesn't, this is documented as an honest negative result with analysis — NOT hidden).
4. [ ] Cold-start holdout set exists with **zero leakage** (verified programmatically, not just by design) and cold-start accuracy is reported **separately** from warm-start.
5. [ ] Business simulation (order-up-to-P90 policy vs. naive point-forecast policy) run, with stockout rate and holding cost reported for both.
6. [ ] Streamlit dashboard runs locally via one command and shows: per-series forecast + quantile bands, cold-start vs warm-start breakdown, business sim results.
7. [ ] Technical write-up (3-5 pages) complete, following the outline in §10.
8. [ ] Full pipeline reproducible via a single `make run` or `./run.sh` from a clean clone (excluding raw data download, which can be a separate documented step).
9. [ ] README has an architecture diagram and a results table.
10. [ ] "Defend this decision" note written (§10.7).

---

## 2. Explicit Non-Goals (do not build these — if tempted, check back here)

- No real-time/low-latency serving API. Batch inference only.
- No novel model architecture. Use DeepAR or TFT via an existing library (PyTorch Forecasting or GluonTS), correctly and rigorously.
- No cross-domain energy dataset in v1 (moved to stretch goals).
- No Kaggle leaderboard chasing — no hyperparameter search purely to shave 0.01 off a metric with no methodological point.
- No frontend beyond a functional Streamlit app — no custom React dashboard, no auth, no multi-user features.

---

## 3. Tech Stack (pin these — do not swap mid-project)

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| Modeling | PyTorch + PyTorch Forecasting | Has DeepAR and TFT implementations built in |
| Data processing | pandas + polars (polars for the large joins, pandas elsewhere) | |
| Fast local querying | DuckDB | For querying processed parquet files without loading everything into memory |
| Experiment tracking | MLflow | Reuse existing setup if available |
| Dashboard | Streamlit | |
| Env management | `venv` + `requirements.txt` (or `uv` if available) | |
| Data format (processed) | Parquet | Not CSV — CSV is only the raw input format |

**Model decision:** Default to **DeepAR** as the primary global model. Rationale to document in the write-up: DeepAR is simpler, faster to train, and its autoregressive quantile output is more directly comparable to the baselines. TFT is a stretch goal (§11) if time allows, since its attention-based interpretability is a nice-to-have, not core to the DoD.

---

## 4. Data

### 4.1 Source
- **M5 Forecasting - Accuracy** dataset (Walmart), available via Kaggle: `m5-forecasting-accuracy`.
- Files needed: `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.
- ~42,840 series: item × store combinations across 3 states, 10 stores, 3,049 products, hierarchical (item → dept → category → store → state), 5+ years of daily sales.

### 4.2 Download step
Document in README as a manual step (Kaggle requires auth):
```
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/
unzip data/raw/m5-forecasting-accuracy.zip -d data/raw/
```

### 4.3 Train/Val/Test split (time-based, not random)
- **Train:** all days except the last 56 days.
- **Validation:** the 28 days before the final 28.
- **Test:** the final 28 days (matches M5's native evaluation horizon).
- Rolling-origin backtesting (§7) uses multiple origins within the train/val range — the test set is touched only once, at the end.

### 4.4 Cold-start holdout definition (be precise — this is the part most likely to leak)
- Select ~5% of series (stratified by category, to keep the holdout representative) to be "simulated new products."
- For these series: **remove all rows before day 1400** (or an appropriate cutoff giving them <28 days of history) from every training fold — not just the final one.
- At inference time for these series, the model must rely only on static/category embeddings — verify this by checking that the model's input tensors for these series contain no historical target values before the cutoff.
- Write an automated test (`tests/test_no_leakage.py`) that asserts: for every cold-start series ID, no timestep before the cutoff appears in any training batch.

---

## 5. Feature Engineering

| Feature type | Examples | Source |
|---|---|---|
| Calendar/event | day of week, month, SNAP flags, event name/type | `calendar.csv` |
| Price | sell price, price change flag | `sell_prices.csv` |
| Lag/rolling | lag-7, lag-28, rolling mean/std (7/28 day windows) | derived |
| Static covariates | category, department, store, state | `sales_train_evaluation.csv` → embedded via learned embedding table |
| Cold-start flag | boolean, whether series has <28 days history at time t | derived |

All feature engineering code lives in `src/features/`, one function per feature group, each independently testable.

---

## 6. Model Layer

### 6.1 Baselines (build first — Phase 2)
- Seasonal naive (lag-7)
- ETS / exponential smoothing (statsmodels)
- Per-series ARIMA or Prophet (use Prophet — faster to fit at scale, well documented)
- Run on a **subset** of series for ARIMA/Prophet if full-scale fitting is too slow (document the subset size and selection method — don't silently subsample without noting it).

### 6.2 Global model (Phase 3)
- DeepAR via PyTorch Forecasting.
- Shared weights across all series.
- Static embedding table for category/store/state.
- Quantile output head: P10/P50/P90 via pinball loss.
- Config to start with (adjust based on compute): hidden size 32, 2 LSTM layers, embedding dim 8 per static covariate, batch size 128, learning rate 1e-3 with a scheduler, early stopping on validation WQL.

### 6.3 Cold-start handling
- Already defined in §4.4. Model architecture doesn't change — the point is the *evaluation methodology*, not a special cold-start model.

---

## 7. Backtesting Harness (build this yourself — do not import a library for this part)

- Rolling-origin (walk-forward) validation: multiple forecast origins within the train/val window, not a single static split.
- Suggested: 3-5 origins spaced 28 days apart.
- For each origin: fit/forecast, compute metrics, store results tagged by origin.
- Report metrics **aggregated across origins** (mean + std) to show stability, not just a single run's number.
- Code lives in `src/backtest/harness.py`. This is a genuine interview artifact — keep it clean and documented.

---

## 8. Evaluation Metrics

- **Primary:** Weighted Quantile Loss (WQL), pinball loss.
- **Secondary:** MASE (interpretable, scale-free).
- **Segmentation — report all metrics broken out by:**
  - Cold-start vs. warm-start
  - High-volume vs. long-tail series (define threshold: e.g., top 20% by average daily volume = high-volume)
  - (Cross-domain segmentation dropped in v1 since only one domain)

---

## 9. Business Translation Layer (Phase 4 — do not skip or shortcut this)

- Simulate an **order-up-to-P90** inventory policy using the quantile forecasts: order enough to cover P90 demand over the lead time.
- Compare against a **naive point-forecast policy** (order-up-to-mean or order-up-to-P50).
- Simulate over the test period per series (or a representative sample if full-scale is too slow — document the sample).
- Report: stockout rate (%) and holding cost (using a reasonable assumed unit holding cost — document the assumption) for both policies.
- This section directly answers "why does this matter to the business" — do not treat it as an afterthought bolt-on at the end.

---

## 10. Deliverables

### 10.1 GitHub repo structure
```
demand-forecasting/
├── README.md
├── PROGRESS.md              # phase tracker — update after every phase
├── run.sh                   # single command, end to end
├── requirements.txt
├── data/
│   ├── raw/                 # gitignored
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
    ├── test_no_leakage.py
    └── ...
```

### 10.2 README requirements
- Architecture diagram (reuse/adapt the one below).
- Results table: baselines vs. global model, warm vs. cold, WQL/MASE.
- How to reproduce (setup, data download, `./run.sh`).

### 10.3 Architecture diagram (for README)
```
Raw Data (M5 CSVs)
   -> Ingestion & Preprocessing (schema normalization, missing values, time split)
   -> Feature Engineering (calendar, lags, static embeddings, cold-start flag)
   -> Model Layer (baselines: naive/ETS/ARIMA/Prophet | global: DeepAR, quantile output)
   -> Backtesting Harness (rolling-origin, segmented metrics)
   -> Business Translation Layer (stockout/holding cost simulation)
   -> Reporting (Streamlit dashboard + technical write-up)
```

### 10.4 Streamlit dashboard — required views
1. Per-series forecast plot with P10/P50/P90 bands (series selector dropdown).
2. Cold-start vs. warm-start accuracy comparison (bar chart, by metric).
3. Business sim results: stockout rate and holding cost, policy comparison.

### 10.5 Technical write-up outline (3-5 pages)
1. Problem framing and why global + probabilistic + cold-start (not just "I built a forecaster")
2. Data and methodology (splits, features, leakage prevention)
3. Baseline results
4. Global model results, with honest discussion of whether/where it beat baselines
5. Cold-start results, discussed separately, with analysis of *why* it performs the way it does
6. Business translation results
7. Limitations and what you'd do with more time/compute
8. References (see §10.6)

### 10.6 References to cite
- Salinas et al., *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks* (2019)
- Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* (2021) — cite even if TFT isn't the primary model, since it's discussed as the alternative considered
- Makridakis et al., *The M5 Competition: Background, Organization, and Implementation* (2022)

### 10.7 "Defend this decision" note (short, ~1 paragraph, in README or write-up appendix)
Must explicitly answer:
- Why WQL over RMSE?
- Why DeepAR over TFT (or vice versa, if swapped)?
- How was cold-start holdout done without leakage?

---

## 11. Stretch Goals (only after §1 checklist is fully complete)

1. Cross-domain port to a second dataset (energy — PJM — or a financial time series domain, if the goal shifts toward finance-adjacent roles later).
2. TFT as a second global model, compared against DeepAR, with attention weight visualization.
3. Hierarchical reconciliation (store-level forecasts sum consistently to state-level).
4. Load-test batch inference pipeline, report throughput (series/sec).

---

## 12. Suggested Build Order (phases — update `PROGRESS.md` after each)

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| 0 | Repo scaffold, env setup, data downloaded | `run.sh` exists (can be a stub), repo structure matches §10.1, raw CSVs present in `data/raw/` |
| 1 | Ingestion + preprocessing + time split | Processed parquet files exist, train/val/test split matches §4.3, unit tests pass |
| 2 | Baselines + backtesting harness | Naive/ETS/ARIMA(or Prophet) produce forecasts and metrics on rolling origins; results saved to a results table |
| 3 | Global model (DeepAR) | Trains end to end, produces P10/P50/P90, beats or is honestly compared against baselines on warm-start WQL |
| 4 | Cold-start holdout + leakage test | `tests/test_no_leakage.py` passes; cold-start metrics reported separately |
| 5 | Business simulation | Stockout/holding cost comparison table produced for both policies |
| 6 | Dashboard | Streamlit app runs locally, all 3 required views work |
| 7 | Write-up + README polish | All of §1 checklist checked off |

---

## 13. Risks & Mitigations (carried over from PRD — still apply)

| Risk | Mitigation |
|---|---|
| Scope creep into serving infra | Explicitly out of scope; batch-only |
| Cold-start leakage | Automated test, not just design intent |
| Deep model doesn't beat baselines | Still a valid, documented finding — do not hide or fudge this |
| Timeline slips | Cross-domain and TFT are stretch goals only, cut first if behind |

---

## 14. Notes for the coding agent specifically

- Re-read §1 and §2 before starting any new phase — they define what's in and out of scope.
- Update `PROGRESS.md` after completing each phase in §12, noting what was done and any deviations from this spec (and why).
- If a design decision in this spec turns out to be impractical (e.g., DeepAR training is too slow on available compute), document the deviation and the reasoning in `PROGRESS.md` rather than silently changing course.
- Never skip the cold-start leakage test to save time — it is one of the two things (along with the business sim) that differentiates this project from a generic forecasting notebook.
