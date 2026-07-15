"""Unit tests for src/models/baselines.py (spec §6.1)."""

from datetime import date, timedelta

import polars as pl

from src.backtest.metrics import quantile_col
from src.models.baselines import (
    ets_quantile_forecast,
    prophet_quantile_forecast,
    seasonal_naive_quantile_forecast,
    select_baseline_subset,
)


def test_select_baseline_subset_is_stratified_and_reproducible():
    ids_cats = pl.DataFrame(
        {
            "id": [f"s{i}" for i in range(60)],
            "cat_id": ["A"] * 30 + ["B"] * 20 + ["C"] * 10,
        }
    )
    sample1 = select_baseline_subset(ids_cats, n=12, seed=42)
    sample2 = select_baseline_subset(ids_cats, n=12, seed=42)
    assert sample1 == sample2  # reproducible
    cats_sampled = ids_cats.filter(pl.col("id").is_in(sample1))["cat_id"].to_list()
    # Roughly proportional: A(30/60) B(20/60) C(10/60) of a 12-sample -> ~6/4/2
    assert cats_sampled.count("A") >= cats_sampled.count("B") >= cats_sampled.count("C")


def make_periodic_series(n_days: int = 35, pattern=(1, 2, 3, 4, 5, 6, 7)) -> pl.DataFrame:
    sales = [float(pattern[i % 7]) for i in range(n_days)]
    return pl.DataFrame({"id": ["X"] * n_days, "d_num": list(range(1, n_days + 1)), "sales": sales})


def test_seasonal_naive_perfect_periodic_series_has_zero_spread():
    hist = make_periodic_series(35)
    fc = seasonal_naive_quantile_forecast(hist, ["X"], horizon=7)
    assert fc.height == 7
    # Perfectly periodic -> lag-7 residuals are all 0 -> q10 == q50 == q90.
    assert (fc[quantile_col(0.1)] == fc[quantile_col(0.5)]).all()
    assert (fc[quantile_col(0.5)] == fc[quantile_col(0.9)]).all()
    # And the tiled pattern should repeat exactly.
    assert fc.sort("d_num")[quantile_col(0.5)].to_list() == [1, 2, 3, 4, 5, 6, 7]


def test_seasonal_naive_skips_series_with_insufficient_history():
    hist = pl.DataFrame({"id": ["Y"] * 3, "d_num": [1, 2, 3], "sales": [1.0, 2.0, 3.0]})
    fc = seasonal_naive_quantile_forecast(hist, ["Y"], horizon=7)
    assert fc.height == 0


def test_seasonal_naive_quantiles_are_monotonic_with_noise():
    import random

    random.seed(0)
    n_days = 70
    sales = [max(0.0, 10 + (i % 7) + random.gauss(0, 3)) for i in range(n_days)]
    hist = pl.DataFrame({"id": ["Z"] * n_days, "d_num": list(range(1, n_days + 1)), "sales": sales})
    fc = seasonal_naive_quantile_forecast(hist, ["Z"], horizon=14)
    assert (fc[quantile_col(0.1)] <= fc[quantile_col(0.5)]).all()
    assert (fc[quantile_col(0.5)] <= fc[quantile_col(0.9)]).all()


def make_smooth_series(n_days: int = 30) -> pl.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    sales = [float(10 + (i % 7)) for i in range(n_days)]
    return pl.DataFrame(
        {"id": ["S"] * n_days, "d_num": list(range(1, n_days + 1)), "sales": sales, "date": dates}
    )


def test_ets_quantile_forecast_smoke():
    hist = make_smooth_series(30)
    fc = ets_quantile_forecast(hist, ["S"], horizon=7, n_sims=50)
    assert fc.height == 7
    assert set(fc.columns) == {"id", "d_num", quantile_col(0.1), quantile_col(0.5), quantile_col(0.9)}
    assert (fc[quantile_col(0.1)] <= fc[quantile_col(0.5)]).all()
    assert (fc[quantile_col(0.5)] <= fc[quantile_col(0.9)]).all()
    assert (fc[quantile_col(0.1)] >= 0).all()


def test_prophet_quantile_forecast_smoke():
    hist = make_smooth_series(30)
    fc = prophet_quantile_forecast(hist, ["S"], horizon=7)
    assert fc.height == 7
    assert set(fc.columns) == {"id", "d_num", quantile_col(0.1), quantile_col(0.5), quantile_col(0.9)}
    assert (fc[quantile_col(0.1)] <= fc[quantile_col(0.5)]).all()
    assert (fc[quantile_col(0.5)] <= fc[quantile_col(0.9)]).all()
    assert (fc[quantile_col(0.1)] >= 0).all()
