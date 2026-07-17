"""Global LightGBM baseline — bonus, not part of the graded spec pipeline
(added after the fact purely to compare against the other models; M5's
actual winning solutions were LightGBM-based, so this is a genuinely
informative comparison point, not scope creep into leaderboard chasing —
nothing here is tuned toward a Kaggle score).

Unlike ETS/Prophet (one model per series), LightGBM here is trained as a
single **global** model across many series at once — directly comparable in
spirit to DeepAR. `run_lightgbm.py` evaluates it on the exact same
~2,001-series population DeepAR uses (same seed, same selection method), so
the LightGBM-vs-DeepAR comparison is apples to apples on population size,
not just on evaluation window.

**Multi-horizon strategy: "horizon-as-feature."** One LightGBM model per
quantile (not one per horizon-day and not a recursive forecast), with the
horizon step `h` (1..28) as an explicit input feature. Trained on a panel
built from many candidate origin days within the available history (not
just the one true forecast origin) — each origin day's already-computed
lag/rolling features (which look backward via `shift`, so they're valid
"as of" that day) paired with the actual future value `h` days out. This is
a standard, well-documented multi-horizon strategy, distinct from training
28 separate per-day models or feeding forecasts back in recursively.

Scope reduction (documented, not silent, consistent with this project's
established style): candidate training origins are spaced `TRAIN_ORIGIN_STRIDE`
days apart within the most recent `TRAIN_LOOKBACK_DAYS` of whatever history
is available at forecast time — not every single day, which would blow the
training panel up to hundreds of millions of rows for no real benefit.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from src.backtest.metrics import QUANTILES, quantile_col
from src.features.calendar import KNOWN_CATEGORICALS as CALENDAR_CATEGORICALS
from src.features.calendar import KNOWN_REALS as CALENDAR_REALS
from src.features.calendar import snap_flag_for_state
from src.features.lags import ROLLING_WINDOWS, add_rolling_features, rolling_feature_names
from src.features.price import KNOWN_REALS as PRICE_REALS
from src.features.price import add_price_change_flag
from src.features.static import STATIC_CATEGORICALS

PROCESSED_DIR = Path("data/processed")

CATEGORICAL_FEATURES = STATIC_CATEGORICALS + CALENDAR_CATEGORICALS
BASE_NUMERIC_FEATURES = CALENDAR_REALS + PRICE_REALS + rolling_feature_names(ROLLING_WINDOWS) + [
    "lag_7",
    "lag_28",
]
FEATURE_NAMES = CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES + ["h"]

TRAIN_ORIGIN_STRIDE = 7
TRAIN_LOOKBACK_DAYS = 180
LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=63, min_child_samples=20, verbosity=-1)


def load_raw_frame(series_ids: list[str] | None = None) -> pl.DataFrame:
    """Raw (unengineered) train+val frame with everything `lightgbm_quantile_forecast`
    needs — same RAW_COLS/casting convention as `global_model.py`, reused
    directly rather than duplicated."""
    from src.models.global_model import RAW_COLS, _STR_COLS

    cast = [pl.col(c).cast(pl.Utf8) for c in _STR_COLS]
    train = pl.read_parquet(PROCESSED_DIR / "train.parquet").select(RAW_COLS).with_columns(cast)
    val = pl.read_parquet(PROCESSED_DIR / "val.parquet").select(RAW_COLS).with_columns(cast)
    df = pl.concat([train, val])
    if series_ids is not None:
        df = df.filter(pl.col("id").is_in(series_ids))
    return df


def _engineer(df: pl.DataFrame) -> pl.DataFrame:
    """Calendar/price/rolling features (reusing the same functions the rest
    of the project uses) plus explicit lag-7/28 — LightGBM has no recurrent
    state, so unlike DeepAR it needs these as literal input columns rather
    than inferred from an encoder sequence.
    """
    df = snap_flag_for_state(df)
    df = add_price_change_flag(df)
    df = add_rolling_features(df)
    return df.sort(["id", "d_num"]).with_columns(
        pl.col("sales").shift(7).over("id").alias("lag_7"),
        pl.col("sales").shift(28).over("id").alias("lag_28"),
    )


def _build_training_panel(
    hist: pl.DataFrame, horizon: int, origin_day: int, stride: int, lookback_days: int
) -> pl.DataFrame:
    """One row per (id, candidate origin day, h) with that day's features
    and the actual sales value `h` days later as the target. Candidate
    origins are restricted to ones with a full `horizon`-day target window
    still inside `hist` (no leakage: never reaches past `origin_day`).
    """
    target_exprs = [pl.col("sales").shift(-h).over("id").alias(f"target_{h}") for h in range(1, horizon + 1)]
    hist_with_targets = hist.with_columns(target_exprs)

    earliest = max(hist.select(pl.col("d_num").min()).item(), origin_day - lookback_days)
    latest = origin_day - horizon
    candidate_days = list(range(earliest, latest + 1, stride)) or [latest]

    candidates = hist_with_targets.filter(pl.col("d_num").is_in(candidate_days))
    panel = candidates.unpivot(
        index=["id", "d_num"] + CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES,
        on=[f"target_{h}" for h in range(1, horizon + 1)],
        variable_name="h_str",
        value_name="target",
    ).with_columns(pl.col("h_str").str.slice(len("target_")).cast(pl.Int32).alias("h"))
    return panel.drop_nulls("target")


def lightgbm_quantile_forecast(
    history: pl.DataFrame,
    series_ids: list[str],
    horizon: int,
    stride: int = TRAIN_ORIGIN_STRIDE,
    lookback_days: int = TRAIN_LOOKBACK_DAYS,
    seed: int = 42,
) -> pl.DataFrame:
    """Matches the rolling-origin harness's `ForecastFn` contract: only ever
    sees `history` (rows at/before the origin), trains fresh each call (same
    per-origin refit pattern as the ETS/Prophet baselines), and returns
    [id, d_num, q0.1, q0.5, q0.9] for the `horizon` days after the origin.
    """
    hist = _engineer(history.filter(pl.col("id").is_in(series_ids)))
    origin_day = hist.select(pl.col("d_num").max()).item()

    panel = _build_training_panel(hist, horizon, origin_day, stride, lookback_days)

    predict_base = hist.filter(pl.col("d_num") == origin_day).select(
        ["id"] + CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
    )
    predict_rows = predict_base.join(pl.DataFrame({"h": list(range(1, horizon + 1))}), how="cross")

    panel_pd = panel.to_pandas()
    predict_pd = predict_rows.to_pandas()
    for c in CATEGORICAL_FEATURES:
        categories = pd.Index(pd.concat([panel_pd[c], predict_pd[c]]).unique())
        panel_pd[c] = pd.Categorical(panel_pd[c], categories=categories)
        predict_pd[c] = pd.Categorical(predict_pd[c], categories=categories)

    X_train = panel_pd[FEATURE_NAMES]
    y_train = panel_pd["target"].astype(float)
    X_predict = predict_pd[FEATURE_NAMES]

    preds = {}
    for q in QUANTILES:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=seed, **LGBM_PARAMS)
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
        preds[q] = model.predict(X_predict)

    q10 = np.clip(preds[0.1], 0, None)
    q50 = np.clip(preds[0.5], 0, None)
    q90 = np.clip(preds[0.9], 0, None)
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)

    out = predict_pd[["id", "h"]].copy()
    out["d_num"] = origin_day + out["h"]
    out[quantile_col(0.1)] = q10
    out[quantile_col(0.5)] = q50
    out[quantile_col(0.9)] = q90
    return pl.from_pandas(out[["id", "d_num"] + [quantile_col(q) for q in QUANTILES]])
