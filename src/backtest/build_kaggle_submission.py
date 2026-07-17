"""Optional bonus, NOT part of the graded pipeline (spec §2 explicitly rules
out Kaggle leaderboard chasing as a non-goal — nothing here is tuned toward
a Kaggle score). This exists purely so the trained model's real quality can
be spot-checked against an actual score, out of curiosity, after the fact.
Not wired into `run.sh`.

Kaggle's m5-forecasting-accuracy submission format needs two 28-day blocks
per series: `_validation` (days 1914-1941, which Phase 5 already produced
real DeepAR forecasts for) and `_evaluation` (days 1942-1969) — the block
Kaggle actually grades. This project's own data only goes through day 1941
(see PROGRESS.md Phase 1: the Kaggle download never included the true
d1942-1969 ground truth), so the `_evaluation` block requires one more
inference pass, genuinely forecasting days this project has never touched.
That's feasible because `calendar.csv`/`sell_prices.csv` (confirmed) both
already extend through day 1969 even though the sales table doesn't — so
real known covariates (events, SNAP, price) exist for days this project has
no target values for.

Coverage: the ~2,001-series DeepAR population gets real model forecasts
(point estimate = q0.5, matching this project's own MASE convention) for
both blocks. The remaining ~28,489 series use the full-scale seasonal-naive
baseline (already cheap at full scale, Phase 2) for both blocks, so the
submission is complete and valid — Kaggle requires a forecast for every
series, and there's no trained-model output for series outside our subset.
"""

import sys
from pathlib import Path

import polars as pl
from pytorch_forecasting import TimeSeriesDataSet

from src.backtest.metrics import quantile_col
from src.backtest.run_deepar import predict_quantiles
from src.backtest.run_deepar_coldstart import train_deepar_coldstart
from src.ingestion.load import load_calendar, load_sell_prices
from src.models.baselines import seasonal_naive_quantile_forecast
from src.models.global_model import RAW_COLS, _STR_COLS, load_and_engineer_features

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

FUTURE_START_DAY = 1942
FUTURE_END_DAY = 1969
HORIZON = 28


def build_future_rows(series_ids: list[str]) -> pl.DataFrame:
    """Known-covariate-only rows for days 1942-1969 (never part of this
    project's own data — see module docstring). `sales` is a 0 placeholder:
    DeepAR's decoder is autoregressive, so during prediction it samples from
    its own fitted distribution at each step rather than consuming the
    target column, and the placeholder is never used as real signal.
    """
    cal = (
        load_calendar(RAW_DIR)
        .with_columns(pl.col("d").str.slice(2).cast(pl.Int32).alias("d_num"))
        .filter((pl.col("d_num") >= FUTURE_START_DAY) & (pl.col("d_num") <= FUTURE_END_DAY))
        .select(
            [
                "d_num", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                "snap_CA", "snap_TX", "snap_WI",
            ]
        )
        .with_columns(
            pl.col("event_name_1").fill_null("none"),
            pl.col("event_type_1").fill_null("none"),
            pl.col("event_name_2").fill_null("none"),
            pl.col("event_type_2").fill_null("none"),
        )
    )
    static_cols = ["id", "item_id", "cat_id", "dept_id", "store_id", "state_id"]
    static = (
        pl.read_parquet(PROCESSED_DIR / "train.parquet")
        .select(static_cols)
        .filter(pl.col("id").is_in(series_ids))
        .unique()
        .with_columns([pl.col(c).cast(pl.Utf8) for c in static_cols])
    )
    grid = static.join(cal, how="cross")

    prices = load_sell_prices(RAW_DIR).with_columns(
        pl.col("store_id").cast(pl.Utf8), pl.col("item_id").cast(pl.Utf8)
    )
    grid = grid.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    grid = grid.with_columns(pl.lit(0).cast(pl.Int32).alias("sales"))
    cast = [pl.col(c).cast(pl.Utf8) for c in _STR_COLS]
    return grid.select(RAW_COLS).with_columns(cast)


def deepar_evaluation_forecast(model, training: TimeSeriesDataSet, config: dict) -> pl.DataFrame:
    """One more inference pass extending the already-trained model through
    day 1969 — genuinely forecasting the true, never-released M5 holdout.
    """
    future_rows = build_future_rows(config["series_ids"])
    pdf_full = load_and_engineer_features(
        series_ids=config["series_ids"],
        min_day=config["min_day"],
        coldstart_ids=config["coldstart_ids"],
        coldstart_cutoff_day=config["cutoff_day"],
        include_test=True,
        extra_rows=future_rows,
    )
    eval_dataset = TimeSeriesDataSet.from_dataset(training, pdf_full, predict=True, stop_randomization=True)
    return predict_quantiles(model, eval_dataset)


def full_scale_naive_forecast(origin_day: int) -> pl.DataFrame:
    """Seasonal-naive point forecast (q0.5) for every series, `HORIZON` days
    past `origin_day`. Cheap and vectorized (Phase 2), used here purely to
    guarantee full coverage for series outside the DeepAR population.
    """
    id_cast = pl.col("id").cast(pl.Utf8)
    parts = [
        pl.read_parquet(PROCESSED_DIR / "train.parquet").select(["id", "d_num", "sales"]).with_columns(id_cast),
        pl.read_parquet(PROCESSED_DIR / "val.parquet").select(["id", "d_num", "sales"]).with_columns(id_cast),
    ]
    if origin_day > pl.read_parquet(PROCESSED_DIR / "val.parquet").select(pl.col("d_num").max()).item():
        parts.append(
            pl.read_parquet(PROCESSED_DIR / "test.parquet").select(["id", "d_num", "sales"]).with_columns(id_cast)
        )
    history = pl.concat(parts).filter(pl.col("d_num") <= origin_day)
    all_ids = history["id"].unique().to_list()
    return seasonal_naive_quantile_forecast(history, all_ids, horizon=HORIZON)


def to_wide(forecast: pl.DataFrame, period_start_day: int, id_suffix: str) -> pl.DataFrame:
    """Long [id, d_num, q0.5] -> Kaggle wide [id, F1..F28], id suffixed
    per Kaggle's `_validation`/`_evaluation` convention. Point forecast =
    q0.5, matching this project's own MASE convention elsewhere.
    """
    f = forecast.with_columns(
        (pl.col("d_num") - period_start_day + 1).alias("F"),
        pl.col("id").str.replace(r"_evaluation$", f"_{id_suffix}").alias("id"),
    )
    return f.pivot(index="id", on="F", values=quantile_col(0.5)).select(
        ["id"] + [str(i) for i in range(1, HORIZON + 1)]
    ).rename({str(i): f"F{i}" for i in range(1, HORIZON + 1)})


def main() -> pl.DataFrame:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model, training, _validation, _pdf, config = train_deepar_coldstart()

    deepar_val = pl.read_csv(REPORTS_DIR / "deepar_test_forecast.csv").with_columns(
        pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64)
    )
    deepar_eval = deepar_evaluation_forecast(model, training, config)

    # NOTE: config["origin_day"] (1885) is the *val*-split origin DeepAR
    # trains against internally — not the boundary for the "_validation"
    # (d1914-1941) / "_evaluation" (d1942-1969) submission blocks. Those are
    # anchored to the real val/test max days instead.
    val_max_day = pl.read_parquet(PROCESSED_DIR / "val.parquet").select(pl.col("d_num").max()).item()
    test_max_day = pl.read_parquet(PROCESSED_DIR / "test.parquet").select(pl.col("d_num").max()).item()

    naive_val = full_scale_naive_forecast(origin_day=val_max_day)
    naive_eval = full_scale_naive_forecast(origin_day=test_max_day)

    def blend(naive: pl.DataFrame, deepar: pl.DataFrame) -> pl.DataFrame:
        deepar_ids = deepar["id"].unique().to_list()
        return pl.concat([naive.filter(~pl.col("id").is_in(deepar_ids)), deepar])

    val_forecast = blend(naive_val, deepar_val)
    eval_forecast = blend(naive_eval, deepar_eval)

    val_wide = to_wide(val_forecast, period_start_day=val_max_day + 1, id_suffix="validation")
    eval_wide = to_wide(eval_forecast, period_start_day=test_max_day + 1, id_suffix="evaluation")

    submission = pl.concat([val_wide, eval_wide])
    submission.write_csv(REPORTS_DIR / "kaggle_submission.csv")
    print(
        f"submission rows: {len(submission)} (expected {2 * naive_val['id'].n_unique()}), "
        f"deepar-covered series: {deepar_val['id'].n_unique()} of {naive_val['id'].n_unique()}",
        flush=True,
    )
    return submission


if __name__ == "__main__":
    main()
