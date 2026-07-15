"""Unit tests for src/ingestion/build_dataset.py (spec §4, Phase 1).

Uses small synthetic frames, not the real M5 CSVs, so these run without
data/raw/ being populated.
"""

import polars as pl
import pytest

from src.ingestion.build_dataset import (
    assign_split,
    drop_pre_release_rows,
    join_calendar,
    join_prices,
    melt_sales,
)


def make_sales_wide() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["A_CA_1", "B_CA_1"],
            "item_id": ["A", "B"],
            "dept_id": ["DEPT_1", "DEPT_1"],
            "cat_id": ["CAT_1", "CAT_1"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            "d_1": [0, 5],
            "d_2": [1, 6],
            "d_3": [2, 7],
        }
    )


def make_calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "d": ["d_1", "d_2", "d_3"],
            "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
            "wm_yr_wk": [11101, 11101, 11102],
            "weekday": ["Saturday", "Sunday", "Monday"],
            "wday": [1, 2, 3],
            "month": [1, 1, 1],
            "year": [2011, 2011, 2011],
            "event_name_1": [None, "SuperBowl", None],
            "event_type_1": [None, "Sporting", None],
            "event_name_2": [None, None, None],
            "event_type_2": [None, None, None],
            "snap_CA": [0, 1, 0],
            "snap_TX": [0, 0, 0],
            "snap_WI": [0, 0, 0],
        }
    )


def make_prices() -> pl.DataFrame:
    # Item "B" has no price row for wm_yr_wk 11101 -> pre-release rows.
    return pl.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["A", "A"],
            "wm_yr_wk": [11101, 11102],
            "sell_price": [9.99, 9.99],
        }
    ).vstack(
        pl.DataFrame(
            {
                "store_id": ["CA_1"],
                "item_id": ["B"],
                "wm_yr_wk": [11102],
                "sell_price": [4.50],
            }
        )
    )


def test_melt_sales_shape_and_values():
    long = melt_sales(make_sales_wide())
    assert long.height == 6  # 2 series x 3 days
    assert set(long.columns) >= {"id", "d", "d_num", "sales"}
    row = long.filter((pl.col("id") == "A_CA_1") & (pl.col("d_num") == 2))
    assert row["sales"].item() == 1


def test_join_calendar_fills_null_events():
    long = melt_sales(make_sales_wide())
    joined = join_calendar(long, make_calendar())
    assert joined.filter(pl.col("event_name_1").is_null()).height == 0
    row = joined.filter((pl.col("d") == "d_2") & (pl.col("id") == "A_CA_1"))
    assert row["event_name_1"].item() == "SuperBowl"


def test_join_prices_introduces_nulls_for_pre_release():
    long = melt_sales(make_sales_wide())
    joined = join_calendar(long, make_calendar())
    priced = join_prices(joined, make_prices())
    # Item B, week 11101 (days 1-2) has no price row -> null.
    pre_release = priced.filter((pl.col("id") == "B_CA_1") & (pl.col("d_num") <= 2))
    assert pre_release["sell_price"].null_count() == pre_release.height
    released = priced.filter((pl.col("id") == "B_CA_1") & (pl.col("d_num") == 3))
    assert released["sell_price"].item() == 4.50


def test_drop_pre_release_rows():
    long = melt_sales(make_sales_wide())
    joined = join_calendar(long, make_calendar())
    priced = join_prices(joined, make_prices())
    cleaned, stats = drop_pre_release_rows(priced)
    assert cleaned.filter(pl.col("sell_price").is_null()).height == 0
    assert stats["rows_before"] == 6
    assert stats["rows_dropped"] == 2  # B_CA_1 days 1-2
    assert stats["rows_after"] == 4


def test_assign_split_boundaries():
    df = pl.DataFrame({"d_num": list(range(1, 101))})  # days 1..100
    split = assign_split(df, test_days=28, val_days=28)
    counts = split.group_by("split").len().sort("split")
    counts_dict = dict(zip(counts["split"].to_list(), counts["len"].to_list()))
    assert counts_dict["test"] == 28
    assert counts_dict["val"] == 28
    assert counts_dict["train"] == 100 - 28 - 28

    # Boundaries: train ends at day 44, val is 45-72, test is 73-100.
    assert split.filter(pl.col("d_num") == 44)["split"].item() == "train"
    assert split.filter(pl.col("d_num") == 45)["split"].item() == "val"
    assert split.filter(pl.col("d_num") == 72)["split"].item() == "val"
    assert split.filter(pl.col("d_num") == 73)["split"].item() == "test"
    assert split.filter(pl.col("d_num") == 100)["split"].item() == "test"


def test_assign_split_no_overlap():
    df = pl.DataFrame({"d_num": list(range(1, 201))})
    split = assign_split(df, test_days=28, val_days=28)
    assert split["split"].n_unique() == 3
    assert set(split["split"].unique().to_list()) == {"train", "val", "test"}
