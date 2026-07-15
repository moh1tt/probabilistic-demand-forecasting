"""Unit tests for src/backtest/metrics.py (spec §8)."""

import numpy as np

from src.backtest.metrics import mase, pinball_loss, weighted_quantile_loss


def test_pinball_loss_perfect_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert np.allclose(pinball_loss(y, y, 0.1), 0.0)
    assert np.allclose(pinball_loss(y, y, 0.9), 0.0)


def test_pinball_loss_asymmetry():
    y_true = np.array([10.0])
    y_pred = np.array([5.0])  # under-forecast
    loss_low_q = pinball_loss(y_true, y_pred, 0.1)
    loss_high_q = pinball_loss(y_true, y_pred, 0.9)
    # Under-forecasting is penalized more heavily at high quantiles.
    assert loss_high_q > loss_low_q


def test_weighted_quantile_loss_perfect_forecast_is_zero():
    y_true = np.array([5.0, 10.0, 0.0, 3.0])
    preds = {0.1: y_true, 0.5: y_true, 0.9: y_true}
    assert weighted_quantile_loss(y_true, preds) == 0.0


def test_weighted_quantile_loss_positive_for_imperfect_forecast():
    y_true = np.array([5.0, 10.0, 0.0, 3.0])
    preds = {0.1: y_true - 1, 0.5: y_true - 1, 0.9: y_true - 1}
    wql = weighted_quantile_loss(y_true, preds)
    assert wql > 0


def test_weighted_quantile_loss_all_zero_actuals_is_nan():
    y_true = np.zeros(5)
    preds = {0.5: np.zeros(5)}
    assert np.isnan(weighted_quantile_loss(y_true, preds))


def test_mase_perfect_forecast_is_zero():
    in_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_true = np.array([11.0, 12.0])
    assert mase(y_true, y_true, in_sample, m=7) == 0.0


def test_mase_scales_by_seasonal_naive_error():
    # Each week is the previous week + 1 -> lag-7 diffs are all 1.0 -> scale = 1.0.
    in_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 3.0])  # off by 1 -> MAE = 1.0, scale = 1.0
    assert mase(y_true, y_pred, in_sample, m=7) == 1.0


def test_mase_zero_scale_is_nan():
    # Perfectly repeating seasonal pattern -> lag-7 error is 0 -> scale undefined.
    in_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 3.0])
    assert np.isnan(mase(y_true, y_pred, in_sample, m=7))


def test_mase_short_history_is_nan():
    in_sample = np.array([1.0, 2.0, 3.0])  # shorter than m
    y_true = np.array([1.0])
    y_pred = np.array([1.0])
    assert np.isnan(mase(y_true, y_pred, in_sample, m=7))
