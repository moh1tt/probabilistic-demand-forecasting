# PROGRESS

Phase tracker for `global-demand-forecaster`, per §14 of
`demand_forecasting_project_spec.md`. Updated after every phase with what was
done and any deviations from the spec (and why).

---

## Phase 0 — Repo scaffold, env setup

**Status:** Done, pending your review.

**What was done:**
- Created repo structure matching spec §10.1: `src/{ingestion,features,models,backtest,business_sim}`,
  `tests/`, `dashboard/`, `reports/`, `notebooks/`, `data/{raw,processed}`.
- Added stub files for later phases (`src/models/baselines.py`, `src/models/global_model.py`,
  `src/backtest/harness.py`, `tests/test_no_leakage.py`, `dashboard/app.py`,
  `reports/technical_writeup.md`) — each a placeholder with a comment pointing to
  the phase that implements it, not empty files with no signal.
- `run.sh` created as a documented stub: checks for raw data, has commented-out
  stage calls for each future phase, currently just prints a status message.
- `requirements.txt` written pinning the spec §3 tech stack (torch+CUDA cu121,
  pytorch-forecasting, lightning, pandas, polars, duckdb, pyarrow, mlflow,
  streamlit, statsmodels, prophet, scikit-learn, matplotlib, plotly, pytest).
- `README.md` drafted with architecture diagram, setup instructions, and the
  manual Kaggle download step documented (not automated, per your instruction —
  you'll run the Kaggle CLI yourself).
- `.gitignore` added (venv/, data/raw, data/processed, mlruns, model checkpoints).
- Initialized git repo, committed the scaffold.
- Installed the full dependency stack into `venv/` and verified `torch.cuda.is_available()`.

**Deviation from spec — Python version:**
- Spec §3 pins Python 3.11. The pre-existing `venv/` in this directory was built
  against Python 3.14.6 (the only Python installed on this machine at the time).
  PyTorch Forecasting, GluonTS-adjacent tooling, and Prophet's cmdstanpy backend
  are all much more likely to have compatibility issues on a Python release this
  new. Flagged to you before proceeding; you chose to install Python 3.11.9
  (via `winget install Python.Python.3.11`) and rebuild `venv/` from scratch
  rather than risk the stack on 3.14. No spec requirement changed — this just
  brings the environment into compliance with §3 as originally written.

**Raw data:**
- Not downloaded by this agent, per your instruction — this is documented as a
  manual step in README.md (`kaggle competitions download -c
  m5-forecasting-accuracy -p data/raw/`, then unzip). `data/raw/` currently
  contains only a `.gitkeep`. **You still need to run this step before Phase 1
  can proceed** (Phase 1 needs `sales_train_evaluation.csv`, `calendar.csv`,
  `sell_prices.csv` in `data/raw/`).

**Acceptance criteria check (§12, Phase 0 row):**
- [x] `run.sh` exists (stub, documented)
- [x] Repo structure matches §10.1
- [ ] Raw CSVs present in `data/raw/` — **blocked on you running the Kaggle CLI**

---

## Phase 1 — Ingestion + preprocessing + time split

Not started. Blocked on raw data being present in `data/raw/`.

---

## Phase 2 — Baselines + backtesting harness

Not started.

---

## Phase 3 — Global model (DeepAR)

Not started.

---

## Phase 4 — Cold-start holdout + leakage test

Not started.

---

## Phase 5 — Business simulation

Not started.

---

## Phase 6 — Dashboard

Not started.

---

## Phase 7 — Write-up + README polish

Not started.
