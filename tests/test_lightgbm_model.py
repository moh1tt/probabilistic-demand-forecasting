"""Unit tests for src/models/lightgbm_model.py (bonus global LightGBM baseline)."""

import polars as pl

from src.models.lightgbm_model import _build_training_panel, _engineer, lightgbm_quantile_forecast


def _synthetic_history(n_days=120, series=("A", "B")) -> pl.DataFrame:
    rows = []
    for sid in series:
        base = 10 if sid == "A" else 3
        for d in range(1, n_days + 1):
            rows.append(
                {
                    "id": sid,
                    "cat_id": "FOODS",
                    "dept_id": "FOODS_1",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "d_num": d,
                    "sales": float(base + (d % 7)),
                    "wday": (d % 7) + 1,
                    "month": 1,
                    "year": 2020,
                    "weekday": str(d % 7),
                    "event_name_1": "none",
                    "event_type_1": "none",
                    "event_name_2": "none",
                    "event_type_2": "none",
                    "snap_CA": 0,
                    "snap_TX": 0,
                    "snap_WI": 0,
                    "sell_price": 5.0,
                }
            )
    return pl.DataFrame(rows)


def test_training_panel_target_matches_actual_future_sales():
    hist = _engineer(_synthetic_history())
    horizon = 5
    origin_day = 100
    panel = _build_training_panel(hist, horizon=horizon, origin_day=origin_day, stride=7, lookback_days=60)

    # Spot-check: for series A, at candidate origin t with h=2, target should
    # equal the actual sales value at day t+2.
    sub = panel.filter((pl.col("id") == "A") & (pl.col("h") == 2)).sort("d_num")
    actual = hist.filter(pl.col("id") == "A").select(["d_num", "sales"])
    for row in sub.iter_rows(named=True):
        t = row["d_num"]
        expected = actual.filter(pl.col("d_num") == t + 2)["sales"].item()
        assert row["target"] == expected


def test_training_panel_never_uses_days_past_origin():
    hist = _engineer(_synthetic_history())
    horizon = 5
    origin_day = 100
    panel = _build_training_panel(hist, horizon=horizon, origin_day=origin_day, stride=7, lookback_days=60)
    # every training example's target day (d_num + h) must be <= origin_day —
    # this is the panel's own no-leakage guarantee, independent of harness.py's.
    assert (panel["d_num"] + panel["h"] <= origin_day).all()
    assert panel["d_num"].max() <= origin_day - horizon


def test_lightgbm_quantile_forecast_output_shape_and_monotonicity():
    hist = _synthetic_history(n_days=150)
    series_ids = ["A", "B"]
    horizon = 7
    out = lightgbm_quantile_forecast(hist, series_ids, horizon=horizon, stride=7, lookback_days=90)

    assert set(out.columns) == {"id", "d_num", "q0.1", "q0.5", "q0.9"}
    assert out["id"].n_unique() == 2
    assert out.height == len(series_ids) * horizon

    origin_day = hist["d_num"].max()
    assert out["d_num"].min() == origin_day + 1
    assert out["d_num"].max() == origin_day + horizon

    assert (out["q0.1"] <= out["q0.5"]).all()
    assert (out["q0.5"] <= out["q0.9"]).all()
    assert (out["q0.1"] >= 0).all()


def test_lightgbm_quantile_forecast_never_receives_future_rows():
    # the function signature only accepts `history` (already the harness's
    # own contract) — assert here that restricting input to day <= 100
    # produces forecasts only for days > 100, i.e. it can't have peeked ahead.
    hist = _synthetic_history(n_days=150).filter(pl.col("d_num") <= 100)
    out = lightgbm_quantile_forecast(hist, ["A", "B"], horizon=7, stride=7, lookback_days=90)
    assert out["d_num"].min() > 100
