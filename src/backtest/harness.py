"""Rolling-origin (walk-forward) backtesting harness (spec §7).

Hand-rolled, no backtesting library. For each forecast origin, a model sees
only history up to and including that origin day and is scored against the
following `horizon` actual days — never against anything later. Metrics are
reported per origin and aggregated (mean + std) across origins.
"""

from typing import Callable, Optional

import numpy as np
import polars as pl

from src.backtest.metrics import QUANTILES, mase_polars, quantile_col, weighted_quantile_loss

# forecast_fn(history_df, series_ids, horizon) -> pl.DataFrame with columns
# [id, d_num] + one column per QUANTILES (via quantile_col), for d_num in
# (origin, origin + horizon].
ForecastFn = Callable[[pl.DataFrame, list[str], int], pl.DataFrame]


def generate_origins(
    max_day: int, horizon: int = 28, n_origins: int = 5, spacing: int = 28
) -> list[int]:
    """Origins spaced `spacing` days apart. The most recent origin's forecast
    window ends exactly at `max_day`, so with max_day = end of val, the last
    origin's evaluation window lines up with the val split (see PROGRESS.md
    Phase 2 note). All earlier origins fall further back in train.
    """
    last_origin = max_day - horizon
    origins = sorted(last_origin - i * spacing for i in range(n_origins))
    if origins[0] < 1:
        raise ValueError(
            f"Not enough history for {n_origins} origins spaced {spacing} days "
            f"apart with horizon {horizon}: earliest origin would be day {origins[0]}."
        )
    return origins


def run_backtest(
    data: pl.DataFrame,
    model_name: str,
    forecast_fn: ForecastFn,
    series_ids: list[str],
    origins: list[int],
    horizon: int = 28,
    segment_col: Optional[str] = None,
) -> pl.DataFrame:
    """Run one model across all origins, return a long results table with
    one row per (origin, segment): model, origin, segment, n_series, wql, mase.

    `data` must contain columns [id, d_num, sales] (+ `segment_col` if given)
    for every day needed: history up to the earliest origin, and actuals
    through the latest origin + horizon.
    """
    q_cols = [quantile_col(q) for q in QUANTILES]
    results = []

    for origin in origins:
        history = data.filter((pl.col("d_num") <= origin) & pl.col("id").is_in(series_ids))
        future = data.filter(
            (pl.col("d_num") > origin)
            & (pl.col("d_num") <= origin + horizon)
            & pl.col("id").is_in(series_ids)
        )
        if history.height:
            assert history.select(pl.col("d_num").max()).item() <= origin, "leakage: history past origin"
        if future.height:
            assert future.select(pl.col("d_num").min()).item() > origin, "future window starts at/before origin"

        forecast = forecast_fn(history, series_ids, horizon).with_columns(
            pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64)
        )

        keep_cols = ["id", "d_num", "sales"] + ([segment_col] if segment_col else [])
        merged = (
            future.select(keep_cols)
            .with_columns(pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64))
            .join(forecast, on=["id", "d_num"], how="inner")
        )

        segments = ["overall"]
        if segment_col:
            segments += sorted(merged[segment_col].unique().to_list())

        for seg in segments:
            seg_df = merged if seg == "overall" else merged.filter(pl.col(segment_col) == seg)
            if seg_df.height == 0:
                continue
            y_true = seg_df["sales"].to_numpy().astype(float)
            q_preds = {q: seg_df[quantile_col(q)].to_numpy().astype(float) for q in QUANTILES}
            wql = weighted_quantile_loss(y_true, q_preds)

            seg_ids = seg_df["id"].unique().to_list()
            seg_history = history.filter(pl.col("id").is_in(seg_ids))
            m = mase_polars(seg_history, seg_df, m=7)

            results.append(
                {
                    "model": model_name,
                    "origin": origin,
                    "segment": seg,
                    "n_series": seg_df["id"].n_unique(),
                    "wql": wql,
                    "mase": m,
                }
            )

    return pl.DataFrame(results)


def aggregate_across_origins(results: pl.DataFrame) -> pl.DataFrame:
    """Mean + std of each metric across origins, per model/segment (spec §7:
    report aggregated across origins to show stability, not a single number).
    """
    return (
        results.group_by(["model", "segment"])
        .agg(
            pl.col("wql").mean().alias("wql_mean"),
            pl.col("wql").std().alias("wql_std"),
            pl.col("mase").mean().alias("mase_mean"),
            pl.col("mase").std().alias("mase_std"),
            pl.col("n_series").mean().alias("avg_n_series"),
            pl.col("origin").len().alias("n_origins"),
        )
        .sort(["model", "segment"])
    )
