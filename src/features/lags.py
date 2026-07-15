"""Lag/rolling features (spec §5): rolling mean/std over 7/28-day windows.

Lag-7 and lag-28 of the target itself are handled separately, via
TimeSeriesDataSet's native `lags={"sales": [7, 28]}` in global_model.py
rather than as manually engineered columns here — pytorch-forecasting's
autoregressive models (DeepAR) expect target lags in that form so the
library can align them correctly across the encoder/decoder boundary
(a future decoder step's lag-7 may fall inside the encoder window, or may
need the model's own prior prediction; the library handles this, hand-
rolled columns would not). This module covers the other half of §5's
"Lag/rolling" row: rolling mean/std, which aren't targets-lags and so don't
have that alignment subtlety.
"""

import polars as pl

ROLLING_WINDOWS = (7, 28)


def add_rolling_features(df: pl.DataFrame, windows=ROLLING_WINDOWS) -> pl.DataFrame:
    """Rolling mean/std of `sales` over each window, computed on the
    `window` days *before* the current row (shift(1) first) so the feature
    at time t never includes day t's own value — using today's sales to
    predict today's sales would be leakage.
    """
    df = df.sort(["id", "d_num"])
    exprs = []
    for w in windows:
        exprs.append(
            pl.col("sales")
            .shift(1)
            .rolling_mean(window_size=w, min_periods=1)
            .over("id")
            .alias(f"rolling_mean_{w}")
        )
        exprs.append(
            pl.col("sales")
            .shift(1)
            .rolling_std(window_size=w, min_periods=2)
            .over("id")
            .fill_null(0.0)
            .alias(f"rolling_std_{w}")
        )
    return df.with_columns(exprs)


def rolling_feature_names(windows=ROLLING_WINDOWS) -> list[str]:
    names = []
    for w in windows:
        names += [f"rolling_mean_{w}", f"rolling_std_{w}"]
    return names
