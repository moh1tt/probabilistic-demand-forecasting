# Technical Write-up: Global Probabilistic Demand Forecasting Engine

*Retail demand forecasting on the M5 / Walmart dataset. Full phase-by-phase
build log, deviations, and every experiment referenced below: `PROGRESS.md`.*

## 1. Problem framing

Retail demand forecasting is usually approached as a per-item time series
problem: fit one model per series (ARIMA, ETS, Prophet), roll forward,
repeat for tens of thousands of items. This scales poorly, ignores that
similar items and stores share demand patterns worth learning jointly, and
has no principled answer for a new product with days of history rather than
years. It also collapses the forecast to a single number, which silently
assumes the business is indifferent to which side of that number it errs
on — almost never true for an inventory decision, where too little safety
stock means stockouts and too much means capital tied up on the shelf.

This project's core claim: a single **global** (multi-series) deep learning
model, trained jointly across item x store series, can produce **calibrated
quantile forecasts** (P10/P50/P90) that are competitive with per-series
statistical baselines, generalize to genuinely new items (**cold start**)
using only category/store static attributes, and — critically — that this
quantile output translates into a **measurable inventory-policy
improvement**, not just a smaller error metric on a benchmark. Global +
probabilistic + cold-start-aware is the thesis; M5 (42,840-series dataset as
originally described, ~30,490 unique item x store series confirmed against
the actual raw file — see below) is large and heterogeneous enough to make
each of those three properties genuinely testable.

## 2. Data and methodology

**Source & split.** M5 Forecasting-Accuracy (Kaggle), 30,490 item x store
series (3,049 products x 10 stores; confirmed against `sales_train_evaluation.csv`'s
row count directly rather than assumed). Time-based split, not random: train
on days 1-1885, validate on 1886-1913 (28 days), test on 1914-1941 (28 days,
M5's native evaluation horizon). One deviation from the spec's literal day
numbers, driven by data availability rather than choice: the Kaggle download
only contains actuals through day 1941, not the longer history the spec
text assumed — the split is anchored to the real max day instead, which
preserves the split's actual purpose (a clean train/val/test boundary,
test touched only once) and happens to reproduce M5's own public/private
leaderboard cutoffs (1913/1941).

**Rolling-origin backtesting**, hand-built rather than a library: 5 origins
spaced 28 days apart within train/val, each with an explicit assertion that
a forecast's history slice never contains a day past its own origin — the
first of two leakage guards in this project.

**Features:** calendar/event (weekday, month, SNAP flags, event name/type),
price (sell price, price-change flag), lag/rolling (lag-7/28 via the
target's own history — for DeepAR, via the library's native lag mechanism,
not hand-rolled columns, since only the library can align lags correctly
across the encoder/decoder boundary), static covariates (category/dept/
store/state via a learned embedding table — `item_id` itself is deliberately
excluded: at 3,049 distinct values it would push the model toward
memorizing per-item identity instead of sharing statistical strength across
the hierarchy), and a cold-start flag (boolean, <28 days of *visible*
history as of day t, computed from each series' own first observed
row — which matters once some of those rows have been deliberately deleted,
next).

**Leakage prevention, cold-start specifically.** ~5% of the DeepAR training
population (100 of ~2,001 series, stratified by category) are designated
simulated new products: every row before a cutoff day is deleted for those
series, at the raw-row level, before any feature engineering or dataset
windowing happens — not filtered only at the final evaluation step. Because
the deletion happens upstream of everything else, no downstream window can
ever contain a pre-cutoff timestep for those series, by construction. It's
verified anyway: `tests/test_no_leakage.py` builds an actual
`TimeSeriesDataSet` (the real library class the model uses), iterates every
batch its training dataloader can produce, and asserts the reconstructed
encoder start day for the cold-start series never precedes its cutoff.

Two library behaviors would have silently defeated the "by design" part
without that test, both caught before the real training run:
1. The library's own `min_encoder_length` filter drops any series without
   enough history from the dataset index entirely (with a warning, not an
   error) — with the default config (min encoder length 30), a cold-start
   series with only 27 visible days would simply vanish from the report,
   not leak. Fixed by lowering it to 1.
2. DeepAR's target-lag-28 feature forces the library to discard each
   series' first 28 rows before building *any* window (to avoid NaN lag
   values) — which for a <28-day cold-start series deletes its *entire*
   visible history regardless of `min_encoder_length`. A hard structural
   conflict, not a tunable threshold; resolved by dropping target lags from
   the cold-start run's feature set entirely.

## 3. Baseline results

Three baselines, backtested across 5 rolling origins: seasonal-naive (lag-7,
vectorized, run at true full scale — all 30,490 series), ETS (Holt-Winters)
and Prophet, both on a 100-series subset stratified by category (documented,
not silently subsampled — full-scale ETS+Prophet across 5 origins would be
~150k+ per-series model fits, infeasible on this hardware in a reasonable
session).

| model | segment | WQL | MASE | n_series |
|---|---|---|---|---|
| seasonal_naive | overall | 0.606 ± 0.044 | 1.121 ± 0.052 | 30,479 |
| seasonal_naive | high_volume | 0.482 ± 0.044 | 1.017 ± 0.016 | 6,097 |
| seasonal_naive | long_tail | 0.833 ± 0.049 | 1.147 ± 0.069 | 24,382 |
| ets | overall | **0.506** ± 0.042 | **0.972** ± 0.032 | 100 |
| ets | high_volume | 0.429 ± 0.045 | 0.835 ± 0.099 | 22 |
| ets | long_tail | 0.683 ± 0.023 | 1.011 ± 0.031 | 78 |
| prophet | overall | 1.035 ± 0.091 | 1.342 ± 0.059 | 100 |

ETS is the strongest baseline on both metrics. Prophet is the weakest,
clearly worse than both naive and ETS — verified this wasn't a plumbing bug
by spot-checking a real series before writing it up; the actual cause is
that M5 series are largely intermittent/low-count (mostly 0-6 units/day
with occasional spikes), and Prophet's continuous-trend, Gaussian-noise
model is a known poor fit for sparse count data (it collapses toward a
near-zero median with a narrow upper quantile that misses the spikes). A
property of the model/data combination, reported as-is rather than tuned
away.

## 4. Global model results

DeepAR (PyTorch Forecasting), trained once (not re-fit per origin) on a
stratified ~2,001-series subset (documented scope reduction — full
30,490-series training was measured at multiple hours on this hardware for
a model this small, which this project's own non-goals rule out chasing),
365 days of history, early-stopped at epoch 4. Evaluated on the val split
(days 1886-1913), the same window as the baselines' final backtest origin,
so ground truth and metric code are identical across the comparison:

| model | n_series | population | WQL | MASE |
|---|---|---|---|---|
| seasonal_naive | 30,490 | full universe | 0.582 | 1.189 |
| ets | 100 | seed 42 | **0.472** | **0.976** |
| prophet | 100 | same | 1.035 | 1.387 |
| deepar | 2,001 | seed 43 | 0.482 | 1.214 |

**Honest comparison:** DeepAR does not clearly beat the baselines here. It
essentially ties ETS on WQL (0.482 vs. 0.472, a ~2% gap plausibly within
noise given ETS's much smaller N=100) and is worse on MASE (1.214 vs. ETS's
0.976, also worse than naive). It clearly beats Prophet on both. Per-series
methods like ETS are, by construction, tailored to each individual series'
own history — a global model's theoretical edge is expected to show most
clearly where per-series fitting has the least data to work with, i.e.
long-tail and cold-start series, which is exactly what the cold-start
evaluation (below) tests directly. Trained for only 4 epochs with no
hyperparameter tuning beyond the starting config, so there's real headroom
on the table; chasing DeepAR's warm-start WQL down further would drift
toward the leaderboard-chasing this project rules out as a non-goal, for a
marginal, less informative number than the cold-start breakdown.

## 5. Cold-start results

100 series (5% of the DeepAR population, stratified by category) had all
history before day 1859 removed — leaving exactly 27 visible days at the
evaluation origin (day 1885), under the <28-day threshold. Retrained the
same architecture (the point is the evaluation methodology, not a special
cold-start model) on this modified population:

| segment | n_series | WQL | MASE |
|---|---|---|---|
| overall | 2,001 | 0.466 | 1.196 |
| cold_start | 100 | 0.460 | 0.867 |
| warm_start | 1,901 | 0.466 | 1.213 |

**Why it performs the way it does:** cold-start WQL (0.460) is essentially
tied with warm-start (0.466) — not worse, which is the non-obvious, useful
result (naive intuition says "less history, worse forecast"). Read this as
evidence the model's shared static/category embeddings carry real
predictive signal for brand-new items — exactly the property a global model
is supposed to have over per-series methods, which have nothing to fall
back on for a new item. One honest caveat specifically on MASE: each
series' MASE denominator (its own in-sample lag-7 naive error) is computed
only from that series' visible history — for cold-start series that's a
~20-diff sample from 27 days, versus 800+ days for warm-start series, so a
noisier, smaller-sample scale estimate can push MASE in either direction.
WQL's denominator (the actual evaluation-window sum, same size for every
series) doesn't have this problem, so it's the more trustworthy of the two
numbers here — and it shows near-parity, not a warm-start advantage,
avoiding overclaiming a MASE artifact as "cold start wins."

## 6. Business translation results

Order-up-to-P90 vs. order-up-to-P50 inventory policy, both driven by the
*same* DeepAR test-period forecast — differing only in which quantile they
order up to, isolating the value of quantile-aware ordering itself rather
than conflating it with a model-quality difference. Periodic review every 7
days, lead time assumed equal to the review period (orders arrive instantly
at cycle start — a standard simplification when lead time <= review
period). Holding cost: 20%/year of each item's own sell price, charged
daily on end-of-day inventory (a standard retail rule of thumb for
capital + storage + obsolescence cost, tied to real per-item prices already
in the pipeline rather than an arbitrary flat rate). Simulated over the
full 2,001-series population, the test period (28 days) — touched here for
the first and only time in the whole project.

| policy | order_up_to | stockout rate | holding cost (total) |
|---|---|---|---|
| P90 | q0.9 | 1.90% | $1,542 |
| P50 | q0.5 | 23.86% | $259 |

Ordering to P90 instead of P50 cuts the stockout rate by ~12.5x at ~6x the
holding cost. This is the textbook safety-stock trade-off, now measured on
real forecasts rather than asserted: a point-forecast (P50) policy is right
about half the time by construction, so it stocks out on roughly half of
demand-review cycles; the P90 policy trades a bounded, modest increase in
holding cost for a large reduction in lost sales. Which trade-off is
"better" depends on the business's actual stockout-vs-holding-cost ratio
(not estimated here — this asks for the comparison itself, not a combined
objective) — but the fact that this trade-off is *measurable at all*, in
dollar and percentage terms, is only possible because the model outputs a
distribution rather than a point estimate. That is the concrete business
case for probabilistic forecasting this project set out to make.

## 7. Limitations and what I'd do with more time/compute

- **Scope reduction was compute-driven, not incidental.** DeepAR trained on
  ~2,001 of 30,490 series (~7%) due to CPU-bound data loading on this
  hardware, not a modeling limitation of DeepAR itself. Full-scale training
  (or a faster data pipeline — e.g. pre-tensorized shards instead of
  per-batch pandas indexing) would let the warm-start comparison run at a
  scale large enough that the DeepAR-vs-ETS gap (currently plausibly within
  noise) could be resolved with confidence either way.
- **Only 4 epochs, no hyperparameter search.** A deliberate choice to avoid
  leaderboard-chasing, but it means DeepAR's true ceiling on this data is
  unmeasured. A held-out validation sweep over hidden size / learning rate /
  encoder length, still stopped before it turned into metric-chasing, is
  the natural next step.
- **TFT as a second global model** would test whether attention-based
  interpretability and TFT's native support for asymmetric encoder/decoder
  features (the rolling mean/std features DeepAR couldn't use — its
  autoregressive decoder requires encoder and decoder to see the same
  variable set) close some of the current gap. DeepAR was chosen over TFT
  for its simpler, faster-to-train, more directly quantile-comparable
  design (§ "Defend this decision" in the README), not because TFT was
  ruled out on merit.
- **Cold-start holdout is simulated, not observed.** These are real, mature
  M5 items with their early history deliberately hidden — a genuinely
  brand-new product might differ in ways a simulated one can't capture
  (e.g. promotional support or unusual initial demand volatility a mature
  item's "recent 27 days" wouldn't reflect). The result (cold-start ~ ties
  warm-start) should be read as "the model can use static embeddings
  effectively when history is short," not as a guarantee about true
  novel-SKU launches.
- **Business sim uses a single, documented lead-time/review-period/holding-cost
  assumption.** A sensitivity sweep across holding cost rates and review
  periods would show whether the P90-vs-P50 trade-off's *shape* holds
  generally, and at what holding-cost rate the two policies' total cost
  (not just stockout rate) would cross over.
- **Hierarchical reconciliation** (store-level forecasts summing consistently
  to state-level) isn't implemented — a real deployment planning at
  multiple levels would likely need it.

## 8. References

- Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR:
  Probabilistic forecasting with autoregressive recurrent networks.
  *International Journal of Forecasting*, 36(3), 1181-1191.
- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion
  Transformers for interpretable multi-horizon time series forecasting.
  *International Journal of Forecasting*, 37(4), 1748-1764.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy
  competition: Results, findings, and conclusions. *International Journal
  of Forecasting*, 38(4), 1346-1364.
