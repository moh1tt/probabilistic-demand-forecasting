"""Phase 6 data prep: consolidate already-computed reports into the single
CSV the dashboard's per-series view (spec §10.4 view 1) needs.

Doesn't retrain or re-predict anything — `reports/deepar_test_forecast.csv`
(Phase 5) already has the test-period P10/P50/P90 for the full ~2,001-series
population. This script just joins that against real sales history (for
plot context) and per-series metadata (cold-start flag, volume segment,
category) using the same deterministic selection functions Phase 4/5 already
use (fixed seeds — reproduces the same population/cold-start ids without
needing the trained model). Output is committed (unlike data/processed/*),
so the dashboard can run from a fresh clone without a GPU or a Kaggle
download, as long as `reports/deepar_test_forecast.csv` exists.
"""

from pathlib import Path

import polars as pl

from src.backtest.run_baselines import add_volume_segment
from src.backtest.run_deepar import select_training_series
from src.features.coldstart import COLDSTART_FRAC, select_coldstart_series

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

HISTORY_DAYS_FOR_PLOT = 120  # days of context shown before the test window starts
COLDSTART_SEED = 44
MAX_PREDICTION_LENGTH = 28


def main() -> pl.DataFrame:
    series_ids = select_training_series()

    train = pl.read_parquet(PROCESSED_DIR / "train.parquet").select(
        ["id", "cat_id", "d_num", "sales"]
    ).with_columns(pl.col("id").cast(pl.Utf8), pl.col("cat_id").cast(pl.Utf8))
    val = pl.read_parquet(PROCESSED_DIR / "val.parquet").select(
        ["id", "cat_id", "d_num", "sales"]
    ).with_columns(pl.col("id").cast(pl.Utf8), pl.col("cat_id").cast(pl.Utf8))
    test = pl.read_parquet(PROCESSED_DIR / "test.parquet").select(
        ["id", "cat_id", "d_num", "sales"]
    ).with_columns(pl.col("id").cast(pl.Utf8), pl.col("cat_id").cast(pl.Utf8))

    train_max_day = train.select(pl.col("d_num").max()).item()
    val_max_day = val.select(pl.col("d_num").max()).item()
    test_max_day = test.select(pl.col("d_num").max()).item()
    origin_day = val_max_day - MAX_PREDICTION_LENGTH

    ids_cats = train.filter(pl.col("id").is_in(series_ids)).select(["id", "cat_id"]).unique(maintain_order=True)
    coldstart_ids = select_coldstart_series(ids_cats, frac=COLDSTART_FRAC, seed=COLDSTART_SEED)

    plot_min_day = test_max_day - MAX_PREDICTION_LENGTH - HISTORY_DAYS_FOR_PLOT + 1
    actuals = (
        pl.concat([train, val, test])
        .filter(pl.col("id").is_in(series_ids) & (pl.col("d_num") >= plot_min_day))
        .with_columns(pl.col("d_num").cast(pl.Int64))
    )
    actuals = add_volume_segment(actuals, origin_day)

    forecast = pl.read_csv(REPORTS_DIR / "deepar_test_forecast.csv").with_columns(
        pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64)
    )

    out = actuals.join(forecast, on=["id", "d_num"], how="left").with_columns(
        pl.col("id").is_in(coldstart_ids).alias("is_coldstart")
    )
    out = out.sort(["id", "d_num"])
    out.write_csv(REPORTS_DIR / "dashboard_series_forecasts.csv")
    print(f"wrote {len(out)} rows for {out['id'].n_unique()} series", flush=True)
    return out


if __name__ == "__main__":
    main()
