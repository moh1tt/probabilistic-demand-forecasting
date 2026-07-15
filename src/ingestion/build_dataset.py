"""Ingestion + preprocessing + time split (spec §4, Phase 1).

Melts the wide per-day sales table to long (series x day) format, joins
calendar and price data, normalizes schema, handles missing values, and
assigns each row to train/val/test per §4.3. Writes one Parquet file per
split to data/processed/.
"""

from pathlib import Path
from typing import Optional

import polars as pl

from src.ingestion.load import load_calendar, load_sales_wide, load_sell_prices

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CATEGORICAL_COLS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]


def melt_sales(sales_wide: pl.DataFrame) -> pl.DataFrame:
    """Wide (one column per day) -> long (one row per series-day)."""
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    long = sales_wide.unpivot(
        index=ID_COLS, on=day_cols, variable_name="d", value_name="sales"
    )
    return long.with_columns(
        pl.col("d").str.slice(2).cast(pl.Int32).alias("d_num"),
        pl.col("sales").cast(pl.Int32),
    )


def join_calendar(long: pl.DataFrame, calendar: pl.DataFrame) -> pl.DataFrame:
    cal = calendar.select(
        [
            "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ]
    ).with_columns(
        pl.col("event_name_1").fill_null("none"),
        pl.col("event_type_1").fill_null("none"),
        pl.col("event_name_2").fill_null("none"),
        pl.col("event_type_2").fill_null("none"),
    )
    return long.join(cal, on="d", how="left")


def join_prices(df: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    return df.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def drop_pre_release_rows(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Drop rows with no sell_price.

    M5's sales table is dense-zero-filled from day 1 for every series, even
    before an item was actually stocked at a store. sell_prices.csv only has
    a row once the item is released, so a null price after the join marks a
    pre-release row. These aren't real "zero demand" observations (the item
    wasn't purchasable), so they're dropped rather than treated as sales
    history. This is standard M5 preprocessing practice, not a modeling choice.
    """
    n_before = df.height
    out = df.filter(pl.col("sell_price").is_not_null())
    n_after = out.height
    stats = {
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_before - n_after,
        "pct_dropped": round(100 * (n_before - n_after) / n_before, 2) if n_before else 0.0,
    }
    return out, stats


def cast_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    return df.with_columns([pl.col(c).cast(pl.Categorical) for c in cols])


def assign_split(df: pl.DataFrame, test_days: int = 28, val_days: int = 28) -> pl.DataFrame:
    """Time-based split per spec §4.3.

    Train = all days except the last `val_days` + `test_days`. Val = the
    `val_days` immediately before the final `test_days`. Test = the final
    `test_days`. Anchored to the max `d_num` actually present in the loaded
    data (see PROGRESS.md Phase 1 note: the public M5 Kaggle download's
    sales_train_evaluation.csv ends at day 1941, not the competition's true
    d_1969 horizon, since the final 28-day ground truth was never published
    on Kaggle — "final 28 days" is therefore relative to our data, not a
    literal calendar anchor).
    """
    max_day = df.select(pl.col("d_num").max()).item()
    test_start = max_day - test_days + 1
    val_start = test_start - val_days
    return df.with_columns(
        pl.when(pl.col("d_num") >= test_start)
        .then(pl.lit("test"))
        .when(pl.col("d_num") >= val_start)
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )


def build(
    raw_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/processed"),
    test_days: int = 28,
    val_days: int = 28,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    with pl.StringCache():
        sales_wide = load_sales_wide(raw_dir)
        calendar = load_calendar(raw_dir)
        prices = load_sell_prices(raw_dir)

        long = melt_sales(sales_wide)
        long = join_calendar(long, calendar)
        long = join_prices(long, prices)
        long, drop_stats = drop_pre_release_rows(long)
        long = cast_categoricals(long)
        long = assign_split(long, test_days=test_days, val_days=val_days)

        split_stats = {}
        for split_name in ("train", "val", "test"):
            split_df = long.filter(pl.col("split") == split_name)
            split_df.write_parquet(out_dir / f"{split_name}.parquet")
            split_stats[split_name] = {
                "rows": split_df.height,
                "min_day": split_df.select(pl.col("d_num").min()).item(),
                "max_day": split_df.select(pl.col("d_num").max()).item(),
                "n_series": split_df.select(pl.col("id").n_unique()).item(),
            }

    return {"drop_stats": drop_stats, "split_stats": split_stats}


if __name__ == "__main__":
    result = build()
    print(result)
