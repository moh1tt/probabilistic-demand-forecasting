"""Bonus orchestration: global LightGBM baseline (not part of the graded
spec pipeline — see src/models/lightgbm_model.py's docstring for why this
exists and how it's scoped). Not wired into run.sh.

Evaluated via the same rolling-origin harness as Phase 2's naive/ETS/Prophet,
on the *same* ~2,001-series population as DeepAR (Phase 3, seed=43) — a
genuine apples-to-apples global-model-vs-global-model comparison, since
DeepAR itself was never run at full 30,490-series scale either.
"""

import sys
from pathlib import Path

import polars as pl

from src.backtest.harness import aggregate_across_origins, generate_origins, run_backtest
from src.backtest.run_deepar import select_training_series
from src.models.lightgbm_model import lightgbm_quantile_forecast, load_raw_frame

REPORTS_DIR = Path("reports")
HORIZON = 28
N_ORIGINS = 5
SPACING = 28


def main() -> pl.DataFrame:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    series_ids = select_training_series()  # same 2,001-series population as DeepAR
    data = load_raw_frame(series_ids)
    max_day = data.select(pl.col("d_num").max()).item()
    origins = generate_origins(max_day, horizon=HORIZON, n_origins=N_ORIGINS, spacing=SPACING)
    print(f"population: {len(series_ids)} series, origins: {origins}", flush=True)

    results = run_backtest(data, "lightgbm", lightgbm_quantile_forecast, series_ids, origins, horizon=HORIZON)
    results.write_csv(REPORTS_DIR / "lightgbm_backtest_results.csv")

    summary = aggregate_across_origins(results)
    summary.write_csv(REPORTS_DIR / "lightgbm_backtest_summary.csv")
    print(summary, flush=True)

    # Direct comparison against DeepAR/naive/ets/prophet on the exact same
    # window DeepAR was evaluated on (origin = last backtest origin, val split).
    last_origin = origins[-1]
    lgbm_overall = results.filter((pl.col("origin") == last_origin) & (pl.col("segment") == "overall"))
    lgbm_row = lgbm_overall.with_columns(pl.lit("lightgbm").alias("model")).select(
        ["model", "n_series", "wql", "mase"]
    )

    phase3 = pl.read_csv(REPORTS_DIR / "phase3_comparison.csv")
    combined = pl.concat([phase3, lgbm_row])
    combined.write_csv(REPORTS_DIR / "phase3_comparison_with_lightgbm.csv")
    print(combined, flush=True)
    return combined


if __name__ == "__main__":
    main()
