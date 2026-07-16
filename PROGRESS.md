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

---

## Phase 3 — Global model (DeepAR)

**Status:** Done, pending your review.

**What was built:**
- `src/features/lags.py`, updated `calendar.py`/`price.py` — added the
  remaining spec §5 feature groups needed for DeepAR (rolling mean/std 7/28,
  a collapsed per-state SNAP flag, price-change flag), each module now also
  declares which of its columns are "known" vs "unknown" future covariates,
  reused directly by the dataset builder.
- `src/models/global_model.py` — `load_and_engineer_features()` (loads
  train+val, restricts to a series subset, engineers features on each
  series' *full* history before truncating to a bounded training window, so
  early rows in the truncated window still get correct rolling/lag context),
  `build_datasets()` (wraps `TimeSeriesDataSet`), `build_model()` (wraps
  `DeepAR.from_dataset`).
- `src/backtest/run_deepar.py` — orchestration: selects a stratified series
  subset, trains with early stopping, generates P10/P50/P90 for the val
  period, evaluates with the *same* `src/backtest/metrics.py` functions used
  for the baselines, writes `reports/deepar_val_forecast.csv` and
  `reports/deepar_val_results.csv`. Wired into `run.sh`.
- `reports/phase3_comparison.csv` — combined table, all four models
  (naive/ets/prophet/deepar) on the val period.
- No new dedicated unit tests for `global_model.py`/`run_deepar.py` — the
  feature-engineering logic they depend on is already covered by
  `test_features.py`, and the two are thin wrappers around pytorch-forecasting
  library code; validated instead through incremental dry runs (10, then 30,
  then 300 series) before committing to the real run, plus the real run
  itself succeeding end to end. Full existing suite re-run clean: 30/30.

**Two technical findings that shaped the implementation (worth documenting,
not just silently coded around):**
1. **DeepAR's loss must be a `DistributionLoss`, not `QuantileLoss`.**
   Verified directly against the installed pytorch-forecasting 1.2.0
   (`DeepAR.__init__`'s `loss` parameter is typed `DistributionLoss`).
   DeepAR's decoder is autoregressive — it samples from a fitted
   distribution at each step and feeds that back in, which a plain
   quantile-regression loss can't support architecturally. Trained with
   `NegativeBinomialDistributionLoss` (the standard choice for
   over-dispersed count data like retail sales), and derived P10/P50/P90 at
   inference by sampling from the fitted distribution
   (`predict(mode="quantiles")`). Spec §8's WQL/pinball loss is still
   exactly what's used to *evaluate* the resulting forecasts — nothing about
   the reported metric changed, only the training objective's mechanics,
   which the spec's own wording ("Quantile output head... via pinball
   loss") doesn't actually pin down at the training-loss level.
2. **DeepAR requires encoder and decoder to see the same variable set**
   (verified via an `AssertionError` from the library when rolling
   mean/std were included as encoder-only "unknown" reals). Rolling
   mean/std of the target are genuinely future-unknown, so they can't be
   decoder inputs — unlike TFT, DeepAR's architecture doesn't support that
   asymmetry. Dropped them from DeepAR's feature set specifically (they're
   still implemented and tested in `src/features/lags.py`); the target's own
   lag-7/lag-28, fed via `TimeSeriesDataSet`'s native `lags={"sales": [7, 28]}`
   mechanism (which the library aligns correctly across the encoder/decoder
   boundary — hand-rolled lag columns wouldn't be), give the model
   equivalent recent-history signal through its recurrent state instead.

**Deviation from spec — training scope, checked with you first:** DeepAR
training turned out to be CPU-bound on data loading, not GPU-bound (the
spec's suggested model is tiny — hidden_size=32, 2 LSTM layers, ~20K
params — the RTX 2070 is barely exercised). Measured before committing to a
real run: 300 series/1yr history took 188s for 2 epochs with
`num_workers=0`, 114s with `num_workers=4`. Full 30,490-series training
would have taken multiple hours, which isn't warranted (§2 rules out
leaderboard-chasing, and §13 says cut scope before timeline, not the
reverse). Presented three concrete options with real timing estimates; you
picked **~2,000 series (stratified by cat_id, same method as the Phase 2
baseline subset, seed=43), last 365 days of train history**. Actual run:
2,001 series selected (target 2,000; proportional allocation rounds up
slightly), 726,785 training samples, early-stopped at epoch 4
(`EarlyStopping(monitor="val_loss", patience=3)`), total wall time well
within the estimated 30-50 minute window. One series
(`HOUSEHOLD_1_400_WI_2_evaluation`) was dropped by the library itself —
insufficient history for the configured encoder length — logged, not
hidden.

**Also spec-sanctioned adjustment:** early stopping monitors `val_loss`
(the NegBin NLL), not literal validation WQL. Computing full WQL every
epoch would require quantile-sampling every series each epoch, adding
significant overhead for a metric that's already well correlated with
NLL improving. Spec §6.2 explicitly frames its config as a starting point
("adjust based on compute"), so this is used as intended, not a silent
deviation.

**Evaluation methodology:** DeepAR is trained *once* (not re-fit per
rolling origin like the cheap classical baselines) and evaluated on the val
split (days 1886-1913). This is exactly the Phase 2 harness's last
origin's (1885) evaluation window by design — the origin spacing was chosen
in Phase 2 specifically so this alignment would hold — so the comparison
below uses identical ground truth and an identical WQL/MASE implementation
for every model, no re-derivation. Test set (1914-1941) untouched, per
spec §4.3.

**Results (`reports/phase3_comparison.csv`):**

| model | n_series | series population | WQL | MASE |
|---|---|---|---|---|
| seasonal_naive | 30,490 | ~full universe | 0.582 | 1.189 |
| ets | 100 | Phase 2 subset (seed=42) | **0.472** | **0.976** |
| prophet | 100 | same as ets | 1.035 | 1.387 |
| deepar | 2,001 | separate subset (seed=43) | 0.482 | 1.214 |

**Honest comparison (spec §1.3, §13 — report as-is, don't hide or tune
away):** DeepAR does **not** clearly beat the baselines. It essentially
ties ETS on WQL (0.482 vs 0.472, a ~2% relative gap that's plausibly within
noise given ETS's N=100) and is worse on MASE (1.214 vs ETS's 0.976, and
also worse than naive's 1.189). It clearly beats Prophet on both metrics.

One honest caveat on the comparison itself: ETS/Prophet's 100-series subset
and DeepAR's 2,001-series subset were deliberately drawn with **different**
seeds (documented reasoning at the time: DeepAR's training population
doesn't need to match the baselines' subset series-for-series). Checked the
actual overlap — only 4 series in common, too few to re-score DeepAR on
literally the same series ETS/Prophet used. All four numbers above are
therefore each model's real performance on its own actual evaluation
population, aligned on time window and metric formula, but not on series
identity. Naive's near-full-scale number and DeepAR's 2,001-series number
are both large enough samples to be reasonably stable; ETS/Prophet's
N=100 is the shakiest of the four.

**Why DeepAR likely doesn't win outright here, and what that means:**
Trained for only 4 epochs before early stopping triggered (patience=3) —
a short run with no hyperparameter tuning beyond spec §6.2's starting
config, so there's real headroom left on the table. More fundamentally,
per-series methods like ETS are, by construction, tailored to each
individual series' own history; a global model's theoretical edge is
expected to show most clearly on long-tail and cold-start series, where
per-series fitting has little data to work with — and cold-start evaluation
is exactly what Phase 4 adds next. Chasing DeepAR's WQL down further via
more epochs/tuning would drift toward the leaderboard-chasing the spec
rules out (§2); the honest finding as-is, plus the Phase 4 cold-start
breakdown, is the more informative story for the write-up than a marginally
better warm-start number.

**Acceptance criteria check (§12, Phase 3 row):**
- [x] Trains end to end
- [x] Produces P10/P50/P90 (`reports/deepar_val_forecast.csv`)
- [x] Honestly compared against baselines on warm-start WQL (see above —
      does not clearly beat them; documented, not hidden)

---

## Phase 4 — Cold-start holdout + leakage test

**Status:** Done, pending your review.

**What was built:**
- `src/features/coldstart.py` — `select_coldstart_series()` (stratified-by-cat_id
  sample, same method as Phase 2/3's `select_baseline_subset`), `coldstart_cutoff_day()`
  (derives a cutoff giving exactly 27 visible days of history at a given
  origin — see deviation note below), `truncate_coldstart_history()` (deletes
  every pre-cutoff row for the designated series, at the raw-row level, before
  any feature engineering or dataset windowing), and `add_coldstart_flag()`
  (spec §5's general "has <28 days history at time t" boolean feature,
  computed from each id's own first *visible* day).
- `src/models/global_model.py` — `load_and_engineer_features()` now accepts
  `coldstart_ids`/`coldstart_cutoff_day`; truncation and flag computation
  happen before rolling/price features so those are derived only from each
  cold-start series' deliberately shortened visible history, never its real
  (deleted) prior history. `coldstart_flag` added to `KNOWN_REALS` (it's a
  function of the time index and each series' own first-observed day, never
  the target's future values, so it's a legitimate known covariate).
  `build_datasets()` gained `min_encoder_length`/`lags` params (see findings
  below for why Phase 4 overrides both).
- `src/backtest/run_deepar_coldstart.py` — Phase 4 orchestration: retrains
  DeepAR on the *same* ~2,001-series population as Phase 3 (`select_training_series`,
  seed=43), with 100 of those series (5%, stratified by cat_id, seed=44)
  additionally designated cold-start and history-truncated. Reports WQL/MASE
  overall, cold-start vs. warm-start, and high-volume vs. long-tail.
- `tests/test_no_leakage.py` — 8 tests, two layers per spec §4.4's demand
  this be "verified programmatically, not just by design": unit tests on the
  truncation/flag functions, and library-level tests that build a real
  `TimeSeriesDataSet`, iterate *every* batch its training dataloader can
  produce, and assert the cold-start series' reconstructed encoder start day
  (`decoder_time_idx[0] - encoder_lengths`) never precedes its cutoff.
  Includes two regression tests for the findings below (one demonstrating
  the library dropping the series without the fix, one demonstrating the fix
  working). Full suite: 39/39 passing.

**Deviation from spec — cutoff day, and why "day 1400" doesn't transfer:**
Spec §4.4 suggests "day 1400" as the truncation point. Under our actual data
range (max day 1941, val origin at day 1885 — see Phase 1's own anchor
deviation), day 1400 would leave a cold-start series with 1885-1400=485
days of visible history at the evaluation origin — nowhere near "<28 days,"
which is the property §4.4 actually cares about (its own wording: "or an
appropriate cutoff giving them <28 days of history"). This isn't specific to
our data — even under the spec's *originally assumed* timeline (train
ending day 1913), day 1400 would still leave 513 days visible. Treated this
the same way as Phase 1's day-anchor issue: derive the cutoff from the
property that actually matters. `coldstart_cutoff_day(origin_day)` returns
`origin_day - 26`, giving exactly 27 visible days at the origin (comfortably
under 28). With origin_day=1885 (val's last encoder day, same as Phase 3),
that's cutoff_day=1859.

**Population — reused Phase 3's subset, not a fresh 5% of the full 30,490:**
The cold-start holdout (100 series) is drawn from *within* Phase 3's
existing ~2,001-series DeepAR population, not the full universe. This keeps
Phase 4 a direct, apples-to-apples extension of an already-validated
pipeline rather than a new unrelated run, and 100/2,001 ≈ 5% still matches
spec §4.4's fraction — just applied to our documented working population
(same style of deviation as Phase 2/3's own subsampling, and noted here for
the same reason: don't silently subsample without saying so).

**Two technical findings, both caught by testing before committing to the
real run (not assumed):**
1. **The library's own `min_encoder_length` filter silently drops any series
   without enough history.** With Phase 3's default (30, i.e.
   `max_encoder_length // 2`), a cold-start series with only 27 visible days
   has no valid window at all — `TimeSeriesDataSet` just removes it from the
   index with a `UserWarning` ("no predictions can be made for those
   series"), which would have made the whole cold-start report silently
   report on zero series rather than leak. Fixed by lowering
   `min_encoder_length` to 1 for the Phase 4 dataset — confirmed via a
   direct experiment (and now `tests/test_no_leakage.py`) that this gives
   the cold-start series a real, short (<28-day) encoder rather than
   dropping it, while warm-start series are unaffected (they still use as
   much of their available history as `max_encoder_length` allows).
2. **DeepAR's target-lag-28 feature (`TARGET_LAGS = {"sales": [7, 28]}`,
   Phase 3's config) is structurally incompatible with a <28-day cold-start
   holdout.** `TimeSeriesDataSet._preprocess_data` unconditionally drops
   each series' first `max(lags)` rows before building any window, to avoid
   NaN lag values. With lag-28 that deletes a cold-start series' *entire*
   visible pre-origin history (all 27 days), regardless of
   `min_encoder_length` — there's simply no data left. This isn't a tunable
   threshold; lag-28 and a <28-day cold-start definition can never coexist
   for the same series. Resolved by dropping target lags entirely from
   Phase 4's feature set (`lags={}`) rather than special-casing cold-start
   series with a different feature schema (DeepAR's recurrent state already
   sees the raw `sales` values across whatever encoder length it gets,
   lagged features or not). This is a genuine deviation from Phase 3's
   feature set, which is why Phase 4's *warm-start* WQL/MASE numbers below
   aren't perfectly apples-to-apples with Phase 3's DeepAR row — flagged
   explicitly rather than silently compared.

**Results (`reports/phase4_coldstart_results.csv`, all 2,001 series survived,
including all 100 cold-start — confirmed by `n_series` below summing
correctly; early-stopped at epoch 4, same as Phase 3):**

| segment_type | segment | n_series | WQL | MASE |
|---|---|---|---|---|
| overall | overall | 2,001 | 0.466 | 1.196 |
| start | cold_start | 100 | 0.460 | 0.867 |
| start | warm_start | 1,901 | 0.466 | 1.213 |
| volume | high_volume | 377 | 0.387 | 0.908 |
| volume | long_tail | 1,624 | 0.580 | 1.264 |

**Honest discussion (spec §10.5 point 5 — analysis of *why*, not just the
number):** Cold-start WQL (0.460) is essentially tied with warm-start
(0.466), and cold-start MASE (0.867) is notably *better* than warm-start
(1.213). Read this as evidence the global model's shared/static embeddings
carry real signal for brand-new items — exactly the property a global model
is supposed to have over per-series methods — but with one honest caveat on
MASE specifically: each series' MASE denominator (its own in-sample lag-7
naive error) is computed only from that series' *visible* history, which
for cold-start series is a ~20-diff sample from 27 days, versus ~830+ days
for warm-start series. A noisier, smaller-sample scale estimate can push
MASE in either direction, so the cold-start MASE advantage is more likely
partly a scale-estimation artifact than a clean win — WQL (denominator is
the actual evaluation-window sum, same size for every series) is the more
trustworthy of the two numbers here, and it shows near-parity, not a
warm-start advantage. Either way, the finding that cold-start does *not*
perform worse is itself the useful, non-obvious result (naive expectation
would be "less history → worse"), and is reported as-is per spec §13.

**Acceptance criteria check (§12, Phase 4 row):**
- [x] `tests/test_no_leakage.py` passes (8/8; full suite 39/39)
- [x] Cold-start metrics reported separately (`reports/phase4_coldstart_results.csv`)

---

## Phase 5 — Business simulation

**Status:** Done, pending your review.

**What was built:**
- `src/business_sim/simulate.py` — `simulate_policy()` (periodic-review
  order-up-to-S policy, per series: order every `REVIEW_PERIOD_DAYS`=7 days
  up to the summed forecast quantile over that cycle, deplete inventory
  against realized demand day by day, record stockouts and accrue holding
  cost) and `run_business_sim()` (runs both policies, returns a summary
  table). 5 unit tests (`tests/test_business_sim.py`) on synthetic data:
  stockout triggers correctly when the order level is below demand, holding
  cost matches the formula exactly, leftover inventory correctly carries
  into the next cycle instead of double-ordering, and — since q0.9 >= q0.5
  by construction — the P90 policy's stockout rate is never worse than
  P50's. Full suite: 44/44 passing.
- `src/backtest/run_business_sim.py` — Phase 5 orchestration: retrains
  DeepAR exactly as Phase 4 does (same population, cold-start holdout,
  early-stopping-on-val_loss — nothing about training/model-selection
  changes), then makes one additional inference pass extending through the
  **test** split (days 1914-1941) — the one-time final touch spec §4.3
  allows. Both policies (order-up-to-P90 and order-up-to-P50) run on the
  *same* DeepAR test-period forecast, differing only in which quantile they
  target — this isolates the value of quantile-aware ordering itself (the
  project's core claim, §0) rather than conflating it with a model-quality
  difference.
- `src/models/global_model.py` — `load_and_engineer_features()` gained
  `include_test`, and `train_deepar_coldstart()` (in `run_deepar_coldstart.py`)
  now also returns the fitted `training` `TimeSeriesDataSet` and a `config`
  dict (population/cutoff/origin), so Phase 5 can derive a test-period
  predict-mode dataset via `TimeSeriesDataSet.from_dataset(training, ...)`
  without refitting anything on test rows.

**Retraining instead of checkpointing:** Phase 3/4 both used
`enable_checkpointing=False`, so there's no saved model to reload. A third
~5-10 minute training run (same config, early-stopped at epoch 4 again) was
cheaper than adding checkpoint save/load plumbing retroactively — noted here
as a deliberate choice, not an oversight.

**Two more real problems found by testing before the real run (same spirit
as Phase 4's — verify against the actual library/data, don't assume):**
1. **A calendar event unseen during training breaks test-period inference.**
   The categorical encoder for `event_name_1` (etc.) is fit on whatever
   values appear in train+val; a rare event (e.g. a specific religious
   holiday) that only falls inside the test window raised a hard `KeyError`
   ("Unknown category ... encountered") the first time this was tried on
   real data. Fixed in `build_datasets()` by giving those four event columns
   an explicit `NaNLabelEncoder(add_nan=True)`, the library's documented
   mechanism for treating unseen categories as "unknown" instead of
   crashing — confirmed on the real run (`UserWarning: Found 4 unknown
   classes which were set to NaN`), not silently patched over.
2. **Minor:** the first draft filtered "actuals" for the business sim by
   `d_num > origin_day`, which actually spans both val *and* test days.
   Harmless in practice (the sim's own inner join on `[id, d_num]` against
   the test-only forecast already restricts to test days), but corrected to
   filter by the forecast's own `d_num` values directly so intent isn't
   ambiguous.

**Holding cost assumption (spec §9: "document the assumption"):** rather
than an arbitrary flat rate, holding cost is tied to each item's own
`sell_price` (already in the pipeline): `ANNUAL_HOLDING_RATE = 0.20`
(20%/year — a standard retail rule of thumb for capital + storage +
obsolescence cost as a fraction of unit value), so daily holding cost per
unit = `sell_price * 0.20 / 365`.

**Policy mechanics (documented simplification):** periodic review every 7
days; lead time assumed equal to the review period, with orders arriving
instantly at the start of each cycle — a standard simplification when lead
time <= review period, avoiding a separate in-transit order pipeline.
Stockouts are lost sales (inventory clipped to 0), not backordered.

**Population:** same ~2,001-series population as Phase 3/4 (documented
reuse, consistent with those phases' own subsampling).

**Results (`reports/business_sim_results.csv`, real test-period run, all
2,001 series, 28 days each = 56,028 series-days simulated per policy):**

| policy | order_up_to | n_series | mean stockout rate | mean holding cost/series | total holding cost |
|---|---|---|---|---|---|
| P90 | q0.9 | 2,001 | 1.90% | $0.77 | $1,542.29 |
| P50 | q0.5 | 2,001 | 23.86% | $0.13 | $258.65 |

**Honest discussion (spec §9 — "why this matters to the business"):**
Ordering up to P90 instead of P50 cuts the stockout rate by ~12.5x (23.9% ->
1.9%) at ~6x the holding cost (\$258.65 -> \$1,542.29 total across the
population). This is the textbook safety-stock trade-off, now quantified on
real forecasts rather than asserted: a point-forecast (P50) policy is
right about half the time by construction, so it stocks out on roughly half
of demand-review cycles; the P90 policy trades a modest, bounded increase in
holding cost for a large reduction in lost sales. Which trade-off is
"better" depends on the business's actual stockout-vs-holding cost ratio
(not estimated here — spec asks only for the rate/cost comparison, not a
combined objective), but the shape of the trade-off — and the fact that it's
*measurable at all* — is only possible because the model outputs quantiles
instead of a single point forecast. That is the concrete business case for
probabilistic forecasting this project set out to make (§0).

**Acceptance criteria check (§12, Phase 5 row):**
- [x] Stockout/holding cost comparison table produced for both policies
      (`reports/business_sim_results.csv`)

---

## Phase 6 — Dashboard

**Status:** Done, pending your review.

**What was built:**
- `dashboard/prepare_data.py` — consolidates already-computed reports into
  one CSV the dashboard's per-series view needs
  (`reports/dashboard_series_forecasts.csv`): joins Phase 5's
  `deepar_test_forecast.csv` against real sales history (120 days of
  context before the test window, for a readable plot) and per-series
  metadata (cold-start flag, volume segment, category), using the same
  deterministic, fixed-seed selection functions Phase 4/5 already use —
  reproduces the same population/cold-start ids without needing the
  trained model or a GPU. Doesn't retrain or re-predict anything. Output is
  committed (unlike `data/processed/*`), so the dashboard runs from a fresh
  clone without a Kaggle download or GPU, as long as the reports exist.
  Wired into `run.sh` as the last step.
- `dashboard/app.py` — Streamlit app, three tabs matching spec §10.4's
  required views exactly:
  1. **Per-series forecast** — series selector (with cold-start-only and
     category filters), Plotly line chart of real sales history with the
     P10/P50/P90 band overlaid on the test-period forecast window.
  2. **Cold-start vs. warm-start** — grouped bar chart of WQL/MASE by
     segment (from `reports/phase4_coldstart_results.csv`), plus an
     expander with the high-volume/long-tail secondary segmentation
     (spec §8).
  3. **Business simulation** — stockout rate and holding cost bar charts,
     P90 vs. P50 policy (from `reports/business_sim_results.csv`), with the
     honest takeaway numbers computed inline.
  All three read only already-committed CSVs — no live model inference.

**Verification (per repo convention — no dedicated pytest file for this
orchestration/UI layer, same as `run_deepar.py`/`run_business_sim.py`;
validated by actually running it, not just reading the code):**
- No headless-browser tool was available in this environment, so used
  Streamlit's own official headless testing API
  (`streamlit.testing.v1.AppTest`) instead — it executes the real script
  end to end (not just imports it) and inspects the resulting element
  tree. Confirmed: 0 exceptions, all 3 tabs render, each with its expected
  metrics/captions/charts (1 chart in tab 1, 2 each in tabs 2-3).
  Interacting with the cold-start-only checkbox and category filter (both
  trigger a script rerun with new widget state) also produced 0 exceptions
  and correctly updated the displayed segment/series.
- Also launched the real `streamlit run dashboard/app.py` process and
  confirmed it serves (HTTP 200) before shutting it down — the AppTest
  check above is what actually exercises the Python logic per session;
  this just confirms the process boots cleanly as a real server too.

**Acceptance criteria check (§12, Phase 6 row):**
- [x] Streamlit app runs locally via one command (`streamlit run dashboard/app.py`)
- [x] All 3 required views work (per-series forecast + bands, cold-start vs.
      warm-start, business sim results)

---

## Phase 7 — Write-up + README polish

**Status:** Done.

**What was built:**
- `reports/technical_writeup.md` — full write-up following spec §10.5's
  8-point outline (problem framing, data/methodology, baseline results,
  global model results with honest discussion, cold-start results with
  analysis of *why*, business translation results, limitations/future work,
  references), pulling every number directly from the reports already
  produced in Phases 2-5 rather than re-deriving anything.
- `README.md` — replaced the working-draft placeholders: architecture
  diagram (kept, already accurate), a real results table (baselines vs.
  global model warm-start, cold-start vs. warm-start, business sim policy
  comparison — spec §10.2), updated repo structure reflecting what actually
  got built (not the Phase 0 stub layout), reproduce instructions covering
  `run.sh` (now a real pipeline, not a stub) and the dashboard's separate
  `streamlit run` command, and the "Defend this decision" note (spec
  §10.7) answering all three required questions (WQL vs. RMSE, DeepAR vs.
  TFT, cold-start leakage prevention).
- Fixed a stale factual error carried over from the Phase 0 draft: the
  README's intro claimed "~42,840 item x store series" (the spec text's
  own figure, §4.1), but Phase 0's own verification against the raw CSV
  confirmed the real count is 30,490 — corrected throughout the README and
  write-up rather than left unreconciled.

**Definition of Done (spec §1) — final check:**
- [x] 1. Baselines (naive, ETS, Prophet) run with rolling-origin backtesting, reported.
- [x] 2. Global DeepAR model, jointly trained, outputs P10/P50/P90.
- [x] 3. Global model vs. baselines on warm-start WQL — honestly documented
      as an essential tie/non-win, not hidden (Phase 3).
- [x] 4. Cold-start holdout, zero leakage verified programmatically
      (`tests/test_no_leakage.py`, 8 tests), cold-start accuracy reported
      separately (Phase 4).
- [x] 5. Business simulation (order-up-to-P90 vs. naive point-forecast
      policy), stockout rate + holding cost reported for both (Phase 5).
- [x] 6. Streamlit dashboard, one command, all 3 required views (Phase 6).
- [x] 7. Technical write-up, 3-5 pages, spec §10 outline (this phase).
- [x] 8. Full pipeline reproducible via `./run.sh` from a clean clone
      (excluding the documented manual raw-data download step).
- [x] 9. README has an architecture diagram and a results table.
- [x] 10. "Defend this decision" note, all 3 required questions answered.

**All 10 Definition of Done items are now checked off. Project complete.**
