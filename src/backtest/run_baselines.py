"""Phase 2 orchestration (spec §6.1, §7, §8, §12): run naive/ETS/Prophet
baselines through the rolling-origin backtest harness, segment by volume,
aggregate across origins, and save results tables to reports/.

Cold-start vs. warm-start segmentation (also listed in §8) is deferred to
Phase 4, once the cold-start holdout exists (spec build order, §12) — this
script reports the overall and high-volume/long-tail breakdowns only.
"""

from pathlib import Path

import polars as pl

from src.backtest.harness import aggregate_across_origins, generate_origins, run_backtest
from src.models.baselines import (
    ets_quantile_forecast,
    prophet_quantile_forecast,
    seasonal_naive_quantile_forecast,
    select_baseline_subset,
)

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
SUBSET_SIZE = 100
SUBSET_SEED = 42
HORIZON = 28
N_ORIGINS = 5
SPACING = 28
HIGH_VOLUME_PCTL = 0.8


def load_trainval() -> pl.DataFrame:
    cols = ["id", "cat_id", "d_num", "sales", "date"]
    cast = [pl.col("id").cast(pl.Utf8), pl.col("cat_id").cast(pl.Utf8)]
    train = pl.read_parquet(PROCESSED_DIR / "train.parquet").select(cols).with_columns(cast)
    val = pl.read_parquet(PROCESSED_DIR / "val.parquet").select(cols).with_columns(cast)
    return pl.concat([train, val])


def add_volume_segment(df: pl.DataFrame, train_max_day: int) -> pl.DataFrame:
    """High-volume = top 20% of series by average daily volume over train
    history; long-tail = the rest (spec §8 segmentation)."""
    avg_vol = (
        df.filter(pl.col("d_num") <= train_max_day)
        .group_by("id")
        .agg(pl.col("sales").mean().alias("avg_daily_sales"))
    )
    threshold = avg_vol.select(pl.col("avg_daily_sales").quantile(HIGH_VOLUME_PCTL)).item()
    avg_vol = avg_vol.with_columns(
        pl.when(pl.col("avg_daily_sales") >= threshold)
        .then(pl.lit("high_volume"))
        .otherwise(pl.lit("long_tail"))
        .alias("volume_segment")
    )
    return df.join(avg_vol.select(["id", "volume_segment"]), on="id", how="left")


def main(
    subset_size: int = SUBSET_SIZE,
    n_origins: int = N_ORIGINS,
    all_series_override: list[str] | None = None,
) -> pl.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_trainval()
    train_max_day = (
        pl.read_parquet(PROCESSED_DIR / "train.parquet").select(pl.col("d_num").max()).item()
    )
    data = add_volume_segment(data, train_max_day)

    max_day = data.select(pl.col("d_num").max()).item()
    origins = generate_origins(max_day, horizon=HORIZON, n_origins=n_origins, spacing=SPACING)
    print("origins:", origins, flush=True)

    all_series = all_series_override or data["id"].unique().to_list()
    subset = select_baseline_subset(data, n=subset_size, seed=SUBSET_SEED)
    print(f"ETS/Prophet subset: {len(subset)} series (target {subset_size})", flush=True)

    results = []

    print("running seasonal_naive (full scale)...", flush=True)
    results.append(
        run_backtest(
            data,
            "seasonal_naive",
            seasonal_naive_quantile_forecast,
            all_series,
            origins,
            horizon=HORIZON,
            segment_col="volume_segment",
        )
    )

    print("running ets (subset)...", flush=True)
    results.append(
        run_backtest(
            data, "ets", ets_quantile_forecast, subset, origins, horizon=HORIZON, segment_col="volume_segment"
        )
    )

    print("running prophet (subset)...", flush=True)
    results.append(
        run_backtest(
            data,
            "prophet",
            prophet_quantile_forecast,
            subset,
            origins,
            horizon=HORIZON,
            segment_col="volume_segment",
        )
    )

    all_results = pl.concat(results)
    all_results.write_csv(REPORTS_DIR / "baseline_backtest_results.csv")

    summary = aggregate_across_origins(all_results)
    summary.write_csv(REPORTS_DIR / "baseline_backtest_summary.csv")
    print(summary, flush=True)
    return summary


if __name__ == "__main__":
    main()
