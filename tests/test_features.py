"""Unit tests for src/features/* (spec §5)."""

import polars as pl

from src.features.calendar import snap_flag_for_state
from src.features.lags import add_rolling_features
from src.features.price import add_price_change_flag


def test_snap_flag_for_state_picks_correct_column():
    df = pl.DataFrame(
        {
            "state_id": ["CA", "TX", "WI"],
            "snap_CA": [1, 0, 0],
            "snap_TX": [0, 1, 0],
            "snap_WI": [0, 0, 1],
        }
    )
    out = snap_flag_for_state(df)
    assert out["snap"].to_list() == [1, 1, 1]


def test_add_price_change_flag_detects_changes_not_first_row():
    df = pl.DataFrame(
        {
            "id": ["A", "A", "A", "A"],
            "d_num": [1, 2, 3, 4],
            "sell_price": [9.99, 9.99, 8.99, 8.99],
        }
    )
    out = add_price_change_flag(df).sort("d_num")
    assert out["price_change_flag"].to_list() == [0, 0, 1, 0]


def test_add_rolling_features_excludes_current_day():
    # sales = [10, 20, 30]; rolling_mean_7 at day 3 should be mean(10, 20) = 15,
    # not mean(10, 20, 30) — today's value must not leak into today's feature.
    df = pl.DataFrame({"id": ["A", "A", "A"], "d_num": [1, 2, 3], "sales": [10, 20, 30]})
    out = add_rolling_features(df, windows=(7,)).sort("d_num")
    assert out["rolling_mean_7"].to_list()[0] is None or out["rolling_mean_7"][0] is None
    assert out["rolling_mean_7"][2] == 15.0


def test_add_rolling_features_per_series_independence():
    df = pl.DataFrame(
        {
            "id": ["A", "A", "B", "B"],
            "d_num": [1, 2, 1, 2],
            "sales": [100, 200, 1, 2],
        }
    )
    out = add_rolling_features(df, windows=(7,)).sort(["id", "d_num"])
    b_rows = out.filter(pl.col("id") == "B")
    # Series B's rolling mean must not be contaminated by series A's values.
    assert b_rows["rolling_mean_7"][1] == 1.0
