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
- Installed the full dependency stack into `venv/`. Verified `torch.cuda.is_available() == True`
  (RTX 2070, torch 2.5.1+cu121) and did an actual `Prophet().fit()/.predict()` smoke test
  (not just import) — see the CmdStan deviation note below for what that took.

**Deviation from spec — Python version:**
- Spec §3 pins Python 3.11. The pre-existing `venv/` in this directory was built
  against Python 3.14.6 (the only Python installed on this machine at the time).
  PyTorch Forecasting, GluonTS-adjacent tooling, and Prophet's cmdstanpy backend
  are all much more likely to have compatibility issues on a Python release this
  new. Flagged to you before proceeding; you chose to install Python 3.11.9
  (via `winget install Python.Python.3.11`) and rebuild `venv/` from scratch
  rather than risk the stack on 3.14. No spec requirement changed — this just
  brings the environment into compliance with §3 as originally written.

**Deviation from spec — Git not pre-installed:**
- This machine had no `git` at all (not just an unversioned directory). Flagged
  to you; you approved installing Git via `winget install Git.Git` (2.55.0.3).
  Repo initialized and Phase 0 scaffold committed.

**Deviation from spec — Prophet/CmdStan required manual toolchain repair:**
- `pip install prophet` succeeded, but Prophet's Windows wheel ships a
  precompiled Stan binary that newer `cmdstanpy` (1.3.0, pulled in
  unconstrained by prophet's own dependency spec) refuses to load: `cmdstanpy`
  added a strict check requiring a `makefile` in the CmdStan directory it's
  pointed at, which Prophet's bundled minimal `cmdstan-2.33.1` folder (binaries
  only, no build files, since it's never rebuilt) doesn't have. Root cause
  confirmed by reproducing the exception directly (Prophet's own error handling
  swallows it into a misleading `AttributeError`). Fix: added an empty
  placeholder `makefile` to
  `venv/Lib/site-packages/prophet/stan_model/cmdstan-2.33.1/` — the documented
  community workaround for this exact prophet/cmdstanpy version mismatch.
  Verified with an actual `Prophet().fit()/.predict()` call, not just import.
- Along the way, also built a full standalone CmdStan 2.39.0 via
  `cmdstanpy.install_cmdstan` (not strictly required for the above fix, but
  needed for general `cmdstanpy` use and worth keeping working). This required
  installing the RTools40 MinGW toolchain by hand: `cmdstanpy`'s own Windows
  installers (`install_cxx_toolchain`, and the toolchain step inside
  `install_cmdstan`) invoke a GUI Inno Setup installer via `subprocess.Popen`
  with no working directory / desktop context in this automation shell, so it
  silently no-ops (exits 0, installs nothing) — not an antivirus block, verified
  by checking Defender logs and by direct invocation. Worked around by: (1)
  having you run the installer directly (`RTools40.exe`, no silent flags) in
  your own interactive session, (2) discovering the earlier silent attempt
  had actually installed to a wrong nested path due to a relative `/DIR` arg
  and moving it into place, (3) running RTools' own bundled `pacman` to install
  the `mingw-w64-x86_64-make` package (the actual missing piece —
  `mingw32-make.exe`), (4) re-running `cmdstanpy.install_cmdstan`, which then
  built cleanly. CmdStan lives at `C:\Users\mohit\.cmdstan\cmdstan-2.39.0`;
  RTools40/mingw32-make at `C:\Users\mohit\.cmdstan\RTools40`. Neither path
  needs to be on `PATH` for Prophet itself to work (confirmed), only if you
  invoke `cmdstanpy.install_cmdstan`/rebuild CmdStan again in a fresh shell.

**Incident — bad commit caught and reverted:** one of the RTools install
attempts (a relative `/DIR` arg resolving against whatever the shell's cwd
happened to be at the time) left a stray `Users\mohit\.cmdstan\RTools40\...`
copy (~270MB of MinGW/RTools binaries) sitting inside the repo root itself,
which `git add -A` picked up and one commit briefly included. Caught it
immediately after (git status showing files way outside the expected diff),
reverted with `git reset --mixed HEAD~1` (safe — repo is local-only, nothing
pushed), deleted the stray folder, added an `/Users/` rule to `.gitignore` to
guard against a repeat, expired the reflog and ran `git gc --prune=now` to
reclaim the space (`.git` went from 272MB back to ~48KB), then re-committed
cleanly. Current history is 2 commits, no oversized objects — verified via
`git count-objects -v`.

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
