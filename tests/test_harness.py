"""Unit tests for src/backtest/harness.py (spec §7)."""

import polars as pl
import pytest

from src.backtest.harness import aggregate_across_origins, generate_origins, run_backtest
from src.backtest.metrics import QUANTILES, quantile_col


def test_generate_origins_spacing_and_alignment():
    origins = generate_origins(max_day=1913, horizon=28, n_origins=5, spacing=28)
    assert len(origins) == 5
    assert origins == sorted(origins)
    # Last origin's forecast window (origin, origin+28] ends exactly at max_day.
    assert origins[-1] + 28 == 1913
    # Spaced 28 days apart.
    diffs = [origins[i + 1] - origins[i] for i in range(len(origins) - 1)]
    assert diffs == [28, 28, 28, 28]


def test_generate_origins_raises_if_insufficient_history():
    with pytest.raises(ValueError):
        generate_origins(max_day=100, horizon=28, n_origins=5, spacing=28)


def make_synthetic_data() -> pl.DataFrame:
    # 2 series, 60 days, deterministic values so a "perfect" forecast_fn is easy to write.
    rows = []
    for sid, base in [("A", 10), ("B", 100)]:
        for d in range(1, 61):
            rows.append({"id": sid, "d_num": d, "sales": float(base + d), "segment": "hi" if base == 100 else "lo"})
    return pl.DataFrame(rows)


def perfect_forecast_fn(history: pl.DataFrame, series_ids: list[str], horizon: int) -> pl.DataFrame:
    """Cheats by reading the true future from closure-captured full data —
    only valid in this test, to check the harness's own plumbing in isolation."""
    max_hist_day = history.select(pl.col("d_num").max()).item()
    full = make_synthetic_data()
    future_actuals = full.filter(
        (pl.col("d_num") > max_hist_day) & (pl.col("d_num") <= max_hist_day + horizon) & pl.col("id").is_in(series_ids)
    )
    out = future_actuals.select(["id", "d_num"])
    for q in QUANTILES:
        out = out.with_columns(future_actuals["sales"].alias(quantile_col(q)))
    return out


def test_run_backtest_perfect_forecast_gives_zero_wql():
    data = make_synthetic_data()
    origins = [30]
    results = run_backtest(
        data, "perfect", perfect_forecast_fn, ["A", "B"], origins, horizon=10, segment_col="segment"
    )
    assert (results["wql"] == 0).all()
    assert set(results["segment"].to_list()) == {"overall", "hi", "lo"}


def test_run_backtest_no_leakage_assertion_holds():
    # history must never exceed the origin day; run_backtest asserts this internally.
    data = make_synthetic_data()
    results = run_backtest(data, "perfect", perfect_forecast_fn, ["A", "B"], [20, 40], horizon=10)
    assert results.height == 2  # 2 origins x 1 segment (overall only)


def test_aggregate_across_origins_mean_std():
    results = pl.DataFrame(
        {
            "model": ["m", "m", "m"],
            "origin": [1, 2, 3],
            "segment": ["overall", "overall", "overall"],
            "n_series": [2, 2, 2],
            "wql": [0.1, 0.3, 0.2],
            "mase": [1.0, 1.0, 1.0],
        }
    )
    agg = aggregate_across_origins(results)
    assert agg.height == 1
    row = agg.row(0, named=True)
    assert row["n_origins"] == 3
    assert abs(row["wql_mean"] - 0.2) < 1e-9
    assert row["mase_mean"] == 1.0
