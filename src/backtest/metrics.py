"""Evaluation metrics (spec §8): Weighted Quantile Loss (primary) and MASE (secondary)."""

import numpy as np
import polars as pl

QUANTILES = (0.1, 0.5, 0.9)


def quantile_col(q: float) -> str:
    """Column-naming convention for a quantile forecast, used consistently
    across baselines.py and harness.py so forecast frames join cleanly."""
    return f"q{q}"


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> np.ndarray:
    """Elementwise pinball (quantile) loss."""
    diff = y_true - y_pred
    return np.maximum(quantile * diff, (quantile - 1) * diff)


def weighted_quantile_loss(
    y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]
) -> float:
    """WQL, averaged across quantile levels.

    For each quantile q: WQL_q = 2 * sum(pinball_loss_q) / sum(|y_true|).
    The 2x scales pinball loss to be comparable to absolute error at q=0.5.
    Overall WQL is the mean of WQL_q across the requested quantile levels
    (0.1, 0.5, 0.9 per spec §8), a standard aggregation (cf. GluonTS).
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    per_quantile = []
    for q, y_pred in quantile_preds.items():
        loss = pinball_loss(y_true, y_pred, q)
        per_quantile.append(2 * np.sum(loss) / denom)
    return float(np.mean(per_quantile))


def mase(
    y_true: np.ndarray,
    y_pred_point: np.ndarray,
    in_sample_actuals: np.ndarray,
    m: int = 7,
) -> float:
    """Mean Absolute Scaled Error, scaled by the in-sample seasonal-naive
    (lag-m) error, per spec §8 (m=7, matching M5's weekly seasonality).
    """
    if len(in_sample_actuals) <= m:
        return float("nan")
    naive_errors = np.abs(in_sample_actuals[m:] - in_sample_actuals[:-m])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float("nan")
    mae = np.mean(np.abs(y_true - y_pred_point))
    return float(mae / scale)


def mase_polars(history: pl.DataFrame, merged: pl.DataFrame, m: int = 7) -> float:
    """Vectorized MASE across all series present in `merged` (mean of each
    series' own MASE — the standard convention, since MASE is inherently a
    per-series scale-normalized metric).

    `history`: columns [id, d_num, sales] — in-sample data available at the
    forecast origin, used to compute each series' seasonal-naive (lag-m) scale.
    `merged`: columns [id, d_num, sales, q0.5] — actuals + median forecast
    for the evaluation window.
    """
    scale = (
        history.sort(["id", "d_num"])
        .with_columns(
            (pl.col("sales") - pl.col("sales").shift(m).over("id")).abs().alias("lag_abs_diff")
        )
        .group_by("id")
        .agg(pl.col("lag_abs_diff").mean().alias("scale"))
    )
    mae = (
        merged.with_columns((pl.col("sales") - pl.col(quantile_col(0.5))).abs().alias("abs_err"))
        .group_by("id")
        .agg(pl.col("abs_err").mean().alias("mae"))
    )
    per_series = mae.join(scale, on="id", how="inner").with_columns(
        pl.when(pl.col("scale") > 0)
        .then(pl.col("mae") / pl.col("scale"))
        .otherwise(None)
        .alias("mase")
    )
    vals = per_series["mase"].drop_nulls()
    return float(vals.mean()) if vals.len() > 0 else float("nan")
