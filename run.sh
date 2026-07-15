#!/usr/bin/env bash
# End-to-end pipeline entry point (spec §1.8, §10.1).
#
# Stages are filled in as each phase lands; see PROGRESS.md for what's
# implemented so far. Raw data download is a manual, one-time step (Kaggle
# auth required) and is NOT part of this script — see README.md.
set -euo pipefail

cd "$(dirname "$0")"

echo "== global-demand-forecaster: run.sh =="

if [ ! -f data/raw/sales_train_evaluation.csv ]; then
  echo "Raw M5 data not found in data/raw/. Download it first — see README.md." >&2
  exit 1
fi

# --- Phase 1: ingestion + preprocessing + time split -----------------------
python -m src.ingestion.build_dataset

# --- Phase 2: baselines + backtesting ---------------------------------------
python -m src.backtest.run_baselines

# --- Phase 3: global model (DeepAR) -----------------------------------------
python -m src.backtest.run_deepar

# --- Phase 4: cold-start evaluation -----------------------------------------
# pytest tests/test_no_leakage.py

# --- Phase 5: business simulation -------------------------------------------
# python -m src.business_sim.simulate

echo "== done (Phases 4-5 not wired up yet) =="
