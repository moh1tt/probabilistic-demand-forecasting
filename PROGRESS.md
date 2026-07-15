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
- Downloaded by you via the Kaggle CLI/website per the README's manual step.
  Verified present and correct in `data/raw/`: `sales_train_evaluation.csv`
  (30,491 rows = 30,490 series + header, matches M5's known series count),
  `calendar.csv` (1,970 rows = 1,969 days + header), `sell_prices.csv`.
  Headers spot-checked against the expected M5 schema. The download had
  extracted one directory level too deep (`data/raw/data/raw/...`, plus the
  `.zip` and two files the project doesn't use — `sales_train_validation.csv`,
  `sample_submission.csv`); moved the three needed files up to `data/raw/` and
  removed the rest. `data/raw/*` stays gitignored, confirmed not tracked.

**Acceptance criteria check (§12, Phase 0 row):**
- [x] `run.sh` exists (stub, documented)
- [x] Repo structure matches §10.1
- [x] Raw CSVs present in `data/raw/` — verified (see above)

**Phase 0 is now fully complete — all acceptance criteria met.**

---

## Phase 1 — Ingestion + preprocessing + time split

**Status:** Done, pending your review.

**What was done:**
- `src/ingestion/load.py` — raw CSV loaders (polars).
- `src/ingestion/build_dataset.py` — melts `sales_train_evaluation.csv` from
  wide (one column per day) to long (one row per series-day), joins in
  `calendar.csv` (date, events, SNAP flags) and `sell_prices.csv` (price),
  normalizes schema (id/category columns cast to `Categorical`, numeric
  columns downcast), handles missing values, assigns the train/val/test split,
  and writes one Parquet file per split to `data/processed/`.
- `tests/test_ingestion.py` — 6 unit tests against small synthetic frames
  (not the real CSVs, so they run without `data/raw/` populated): melt
  correctness, calendar join + null-event-fill, price join introducing nulls
  for pre-release rows, the drop policy, and split-boundary correctness
  (including a no-overlap check). All pass.
- `run.sh` Phase 1 stage uncommented (`python -m src.ingestion.build_dataset`).

**Missing-value handling (schema normalization, per §10.3):**
- M5's sales table is dense-zero-filled from day 1 for every series, even
  before an item was actually stocked at a given store — `sell_prices.csv`
  only has a row once the item is released, so a null price after the join
  marks a pre-release row. These rows were **dropped**, not treated as
  zero-demand history, since the item wasn't purchasable yet (standard M5
  preprocessing practice, not a novel modeling choice). Result: 59,181,090
  raw series-day rows -> 46,881,677 after dropping (20.78% dropped, all from
  the training window — the val/test windows, being very late in the
  5+-year history, had zero pre-release rows to drop, confirming virtually
  every item is released by then).
- Null `event_name_1/2`, `event_type_1/2` (no event that day) filled with
  the string `"none"`.
- No other nulls: verified `null_count()` is all-zero on every column across
  all three output Parquet files.

**Deviation from spec — split anchored to available data, not literal d_1942-1969:**
- Spec §4.3 defines the split in relative terms ("last 56 days", "final 28
  days") and separately notes the 28-day test length "matches M5's native
  evaluation horizon." The Kaggle `m5-forecasting-accuracy` competition
  download's `sales_train_evaluation.csv` only contains actuals through day
  1941 (1941 day-columns, confirmed by header inspection) — the true final
  28-day holdout (d_1942-d_1969) was never published as part of the standard
  Kaggle competition data; it's only available from the separate
  post-competition M5-methods GitHub release, which is outside the
  documented §4.2 download step. Rather than fetch data from a source the
  spec doesn't mention, `assign_split()` computes the split relative to
  whatever the max day in the loaded data actually is (currently 1941), so
  "final 28 days" = d_1914-d_1941, "val" = d_1886-d_1913, "train" = d_1-d_1885.
  This preserves the spec's actual methodology (rolling-origin backtesting in
  Phase 2, leakage prevention in Phase 4) exactly — only the absolute
  calendar anchor differs, and only because of a data-availability gap, not a
  methodology change. Convenient side note: d_1913/d_1941 are exactly M5's
  own historical public/private-leaderboard cutoffs, so this split reproduces
  a well-precedented boundary, not an arbitrary one.

**Split results (from the real run):**
| split | rows | days | series |
|---|---|---|---|
| train | 45,174,237 | d_1 – d_1885 | 30,490 |
| val | 853,720 | d_1886 – d_1913 (28 days) | 30,490 |
| test | 853,720 | d_1914 – d_1941 (28 days) | 30,490 |

**Output:** `data/processed/{train,val,test}.parquet` (train ~102MB, val/test
~0.7MB each — not committed, gitignored per `data/processed/*`).

**Acceptance criteria check (§12, Phase 1 row):**
- [x] Processed parquet files exist
- [x] Train/val/test split matches §4.3 (see deviation note above for the one
      anchor-point caveat)
- [x] Unit tests pass (6/6, plus the full existing suite re-run clean)

---

## Phase 2 — Baselines + backtesting harness

**Status:** Done, pending your review.

**What was built:**
- `src/backtest/metrics.py` — WQL (pinball loss, averaged across P10/P50/P90
  per spec §8) and MASE (scaled by in-sample lag-7 seasonal-naive error).
  `mase_polars` computes it vectorized (group_by, not a per-series Python
  loop) so it stays fast at full 30,490-series scale.
- `src/backtest/harness.py` — hand-rolled rolling-origin backtester (spec §7,
  no library): `generate_origins()` places N origins spaced 28 days apart;
  `run_backtest()` slices history/future per origin with an explicit
  assertion that history never contains a day past the origin (a real
  no-leakage check, not just a design intent) and reports metrics per
  origin **and** per segment; `aggregate_across_origins()` gives mean+std
  across origins per spec §7's "show stability, not just one number."
- `src/models/baselines.py` — seasonal naive (lag-7, fully vectorized in
  polars, runs at full scale), ETS (statsmodels Holt-Winters, quantiles via
  simulation), Prophet (80% interval via `interval_width=0.8`). All three
  return the same `[id, d_num, q0.1, q0.5, q0.9]` shape so the harness can
  run any of them interchangeably.
- `src/backtest/run_baselines.py` — orchestration: loads train+val, tags
  each series `high_volume`/`long_tail` (top 20% by avg daily volume over
  train, per spec §8), runs all three models across 5 origins, writes
  `reports/baseline_backtest_results.csv` (per origin/segment) and
  `reports/baseline_backtest_summary.csv` (aggregated). Wired into `run.sh`.
- 17 new unit tests (`test_metrics.py`, `test_harness.py`, `test_baselines.py`),
  all against small synthetic data — including a perfect-forecast check that
  WQL is exactly 0, and an explicit no-leakage assertion test on the harness
  itself. Full suite: 26/26 passing.

**Subset size and selection method (spec §6.1 — must be documented, not
silent):** ETS and Prophet fit one model per series; at full scale (30,490
series x 5 origins) that's ~150k+ fits each, infeasible here. Both run on
the same 100-series subset, **stratified by `cat_id` with proportional
allocation** (`select_baseline_subset()`, seed=42, reproducible). Seasonal
naive is cheap and vectorized, so it runs at true full scale (all 30,490
series) — no subsampling needed there.

**Origin placement:** 5 origins, 28 days apart, chosen so the *last*
origin's 28-day evaluation window lands exactly on the val split
(days 1886-1913) and all earlier origins' windows fall inside train — so
backtesting never touches the test split, per spec §7's "test set touched
only once, at the end."

**Volume segmentation (spec §8):** top 20% of series by mean daily sales
over the train period = `high_volume`; the rest = `long_tail`. Computed once
and reused across all origins (a series' segment doesn't change origin to
origin). Cold-start vs. warm-start segmentation (also in §8) is **deferred
to Phase 4**, per the spec's own build order — no cold-start holdout exists
yet.

**Results (full run, `reports/baseline_backtest_summary.csv`):**

| model | segment | WQL (mean ± std) | MASE (mean ± std) | avg n_series |
|---|---|---|---|---|
| seasonal_naive | overall | 0.606 ± 0.044 | 1.121 ± 0.052 | 30,479 |
| seasonal_naive | high_volume | 0.482 ± 0.044 | 1.017 ± 0.016 | 6,097 |
| seasonal_naive | long_tail | 0.833 ± 0.049 | 1.147 ± 0.069 | 24,382 |
| ets | overall | **0.506** ± 0.042 | **0.972** ± 0.032 | 100 |
| ets | high_volume | 0.429 ± 0.045 | 0.835 ± 0.099 | 22 |
| ets | long_tail | 0.683 ± 0.023 | 1.011 ± 0.031 | 78 |
| prophet | overall | 1.035 ± 0.091 | 1.342 ± 0.059 | 100 |
| prophet | high_volume | 1.105 ± 0.144 | 1.714 ± 0.170 | 22 |
| prophet | long_tail | 0.884 ± 0.025 | 1.237 ± 0.028 | 78 |

**Honest finding, not hidden (spec §13):** Prophet is the weakest baseline
here, clearly worse than both naive and ETS on this subset. Spot-checked one
`high_volume` series (`FOODS_1_052_WI_2_evaluation`, origin day 1885) to rule
out a plumbing bug before writing this up — dates align correctly and
forecast magnitudes are in the right range. The actual cause: this series
(like much of M5) is intermittent/low-count (mostly 0-6 units/day with
occasional spikes), and Prophet's continuous-trend, Gaussian-noise model is
a known poor fit for sparse count data — it collapses toward a near-zero
median with a narrow upper quantile that misses the spikes. ETS and naive,
which don't assume continuous Gaussian noise, handle this better. This is a
property of the data/model combination, not a bug, and is exactly the kind
of result the spec says to report honestly rather than tune away (§13,
"deep model doesn't beat baselines... still a valid, documented finding" —
same principle applied here to a baseline-vs-baseline comparison). Will
carry this framing into the DeepAR-vs-baselines comparison in Phase 3.

**Acceptance criteria check (§12, Phase 2 row):**
- [x] Naive/ETS/Prophet produce forecasts and metrics on rolling origins
- [x] Results saved to a results table (`reports/baseline_backtest_*.csv`)

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
