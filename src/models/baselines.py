"""Baseline forecasters (spec §6.1): seasonal naive, ETS, Prophet.

Seasonal naive runs at full scale (all series) since it's cheap and
vectorized. ETS and Prophet fit one model per series, which is too slow to
run at full 30,490-series scale on this hardware across multiple backtest
origins — per spec §6.1, they run on a documented stratified subset instead
(see select_baseline_subset).
"""

import logging

import numpy as np
import pandas as pd
import polars as pl

from src.backtest.metrics import QUANTILES, quantile_col

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True  # setLevel alone doesn't
# suppress cmdstanpy's per-fit "Chain [1] start/done processing" INFO lines
# in this version — it resets its own level per call.

EMPTY_FORECAST_SCHEMA = {"id": pl.Utf8, "d_num": pl.Int64} | {
    quantile_col(q): pl.Float64 for q in QUANTILES
}


def select_baseline_subset(df: pl.DataFrame, n: int = 100, seed: int = 42) -> list[str]:
    """Stratified sample of series IDs, proportionally allocated by cat_id,
    for the ETS/Prophet baselines (spec §6.1: document subset size and
    selection method when full-scale fitting is too slow — full-scale ETS+
    Prophet across 5 backtest origins would mean ~150k+ per-series model
    fits, infeasible on this hardware within a reasonable session).
    """
    ids_cats = df.select(["id", "cat_id"]).unique(maintain_order=True).to_pandas()
    frac = min(1.0, n / len(ids_cats))
    sampled = ids_cats.groupby("cat_id", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed), include_groups=False
    )
    return sorted(sampled["id"].tolist())


def seasonal_naive_quantile_forecast(
    history: pl.DataFrame, series_ids: list[str], horizon: int, lag: int = 7
) -> pl.DataFrame:
    """Vectorized seasonal-naive forecast (lag-7) for every series in
    `series_ids`, run at full scale.

    Point forecast (q0.5): the last `lag` observed values, tiled across the
    horizon (standard snaive convention). Quantile spread (q0.1/q0.9): the
    empirical 10th/90th percentile of each series' own in-sample lag-7
    residuals, added to the point forecast and clipped at 0 (sales can't be
    negative) and to preserve q0.1 <= q0.5 <= q0.9.
    """
    hist = history.filter(pl.col("id").is_in(series_ids)).sort(["id", "d_num"])
    counts = hist.group_by("id").agg(pl.len().alias("n"))
    eligible_ids = counts.filter(pl.col("n") >= lag)["id"].to_list()
    if not eligible_ids:
        return pl.DataFrame(schema=EMPTY_FORECAST_SCHEMA)
    hist = hist.filter(pl.col("id").is_in(eligible_ids))
    origin = hist.select(pl.col("d_num").max()).item()
    # Guard: series may have gaps or fewer rows near the origin if some
    # upstream filtering left holes; take literally the last `lag` rows.
    pattern = (
        hist.group_by("id", maintain_order=True)
        .tail(lag)
        .with_columns(pl.int_range(0, pl.len()).over("id").alias("pattern_idx"))
        .select(["id", "pattern_idx", "sales"])
    )
    # Only keep series whose last `lag` rows are a full, gapless pattern.
    full_pattern_ids = (
        pattern.group_by("id").agg(pl.len().alias("n")).filter(pl.col("n") == lag)["id"].to_list()
    )
    pattern = pattern.filter(pl.col("id").is_in(full_pattern_ids))

    future_days = pl.DataFrame({"h": list(range(1, horizon + 1))}).with_columns(
        ((pl.col("h") - 1) % lag).alias("pattern_idx"),
        (pl.col("h") + origin).alias("d_num"),
    )
    future_grid = pl.DataFrame({"id": full_pattern_ids}).join(future_days, how="cross")
    point = future_grid.join(pattern, on=["id", "pattern_idx"], how="left").rename(
        {"sales": "point"}
    )

    resid = (
        hist.filter(pl.col("id").is_in(full_pattern_ids))
        .with_columns((pl.col("sales") - pl.col("sales").shift(lag).over("id")).alias("resid"))
        .drop_nulls("resid")
    )
    resid_q = resid.group_by("id").agg(
        pl.col("resid").quantile(0.1).alias("resid_q10"),
        pl.col("resid").quantile(0.9).alias("resid_q90"),
    )

    out = point.join(resid_q, on="id", how="left").with_columns(
        pl.col("resid_q10").fill_null(0.0),
        pl.col("resid_q90").fill_null(0.0),
    )
    out = out.with_columns(
        pl.max_horizontal(pl.col("point"), pl.lit(0.0)).alias(quantile_col(0.5)),
        pl.max_horizontal(pl.col("point") + pl.col("resid_q10"), pl.lit(0.0)).alias(
            quantile_col(0.1)
        ),
        pl.max_horizontal(pl.col("point") + pl.col("resid_q90"), pl.lit(0.0)).alias(
            quantile_col(0.9)
        ),
    )
    # Enforce monotonicity in case a skewed residual distribution pushed a
    # bound past the point forecast.
    out = out.with_columns(
        pl.min_horizontal(quantile_col(0.1), quantile_col(0.5)).alias(quantile_col(0.1)),
        pl.max_horizontal(quantile_col(0.9), quantile_col(0.5)).alias(quantile_col(0.9)),
    )
    return out.select(["id", "d_num"] + [quantile_col(q) for q in QUANTILES])


def ets_quantile_forecast(
    history: pl.DataFrame,
    series_ids: list[str],
    horizon: int,
    n_sims: int = 200,
    seed: int = 42,
    min_history_cycles: int = 2,
    seasonal_periods: int = 7,
) -> pl.DataFrame:
    """Per-series ETS (Holt-Winters, statsmodels), quantiles via simulation.

    Series with fewer than `min_history_cycles` full seasonal cycles of
    history, or for which the fit raises, are skipped (not silently
    substituted) — the caller sees a smaller n_series for that origin, which
    is reported, not hidden.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    min_len = min_history_cycles * seasonal_periods
    rows = []
    for i, sid in enumerate(series_ids):
        s = history.filter(pl.col("id") == sid).sort("d_num")
        if s.height < min_len:
            continue
        origin_day = s.select(pl.col("d_num").max()).item()
        y = s["sales"].to_numpy().astype(float)
        try:
            model = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            sims = np.asarray(
                fit.simulate(nsimulations=horizon, repetitions=n_sims, random_state=seed + i)
            )
        except Exception:
            continue
        q10 = np.clip(np.quantile(sims, 0.1, axis=1), 0, None)
        q50 = np.clip(np.quantile(sims, 0.5, axis=1), 0, None)
        q90 = np.clip(np.quantile(sims, 0.9, axis=1), 0, None)
        for h in range(horizon):
            rows.append(
                {
                    "id": sid,
                    "d_num": origin_day + h + 1,
                    quantile_col(0.1): min(q10[h], q50[h]),
                    quantile_col(0.5): q50[h],
                    quantile_col(0.9): max(q90[h], q50[h]),
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema=EMPTY_FORECAST_SCHEMA)


def prophet_quantile_forecast(
    history: pl.DataFrame, series_ids: list[str], horizon: int, min_history_days: int = 14
) -> pl.DataFrame:
    """Per-series Prophet (spec §6.1), 80% interval (P10/P90) via
    interval_width=0.8, MAP optimization (no MCMC, for speed).
    """
    from prophet import Prophet

    rows = []
    for sid in series_ids:
        s = history.filter(pl.col("id") == sid).sort("d_num")
        if s.height < min_history_days:
            continue
        origin_day = s.select(pl.col("d_num").max()).item()
        df = s.select(["date", "sales"]).rename({"date": "ds", "sales": "y"}).to_pandas()
        try:
            m = Prophet(interval_width=0.8, weekly_seasonality=True, daily_seasonality=False)
            m.fit(df)
            future = m.make_future_dataframe(periods=horizon, include_history=False)
            fc = m.predict(future)
        except Exception:
            continue
        d_nums = list(range(origin_day + 1, origin_day + horizon + 1))
        for i in range(len(fc)):
            q50 = max(fc["yhat"].iloc[i], 0.0)
            q10 = min(max(fc["yhat_lower"].iloc[i], 0.0), q50)
            q90 = max(max(fc["yhat_upper"].iloc[i], 0.0), q50)
            rows.append(
                {
                    "id": sid,
                    "d_num": d_nums[i],
                    quantile_col(0.1): q10,
                    quantile_col(0.5): q50,
                    quantile_col(0.9): q90,
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema=EMPTY_FORECAST_SCHEMA)
