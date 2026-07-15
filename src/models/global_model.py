"""Global DeepAR model (spec §6.2), via PyTorch Forecasting.

DeepAR's decoder is autoregressive and samples from a predictive
distribution at each step, so its `loss` must be a DistributionLoss, not a
QuantileLoss (verified against the installed pytorch-forecasting 1.2.0
signature — QuantileLoss isn't accepted). We train with
NegativeBinomialDistributionLoss, the standard choice for over-dispersed,
non-negative count data like retail sales, and derive P10/P50/P90 at
inference by sampling from the fitted distribution (`predict(mode="quantiles")`).
Spec §8's WQL/pinball loss is then used purely as the *evaluation* metric,
consistent with how it's already used for the Phase 2 baselines — nothing
about the reported metric changes, only the training objective's mechanics.
"""

from pathlib import Path

import pandas as pd
import polars as pl

from src.features.calendar import KNOWN_CATEGORICALS as CALENDAR_KNOWN_CATEGORICALS
from src.features.calendar import KNOWN_REALS as CALENDAR_KNOWN_REALS
from src.features.calendar import snap_flag_for_state
from src.features.lags import add_rolling_features, rolling_feature_names
from src.features.price import KNOWN_REALS as PRICE_KNOWN_REALS
from src.features.price import add_price_change_flag
from src.features.static import STATIC_CATEGORICALS

PROCESSED_DIR = Path("data/processed")

KNOWN_REALS = CALENDAR_KNOWN_REALS + PRICE_KNOWN_REALS
KNOWN_CATEGORICALS = CALENDAR_KNOWN_CATEGORICALS
# DeepAR's autoregressive decoder requires encoder_variables == decoder_variables
# (apart from the target) — verified via an AssertionError from the library when
# rolling_mean_*/rolling_std_* (genuinely future-unknown, encoder-only signals)
# were included. Dropped for DeepAR specifically; the target's own lag-7/lag-28
# (below, via TimeSeriesDataSet's native `lags`) give the model equivalent
# recent-history signal through its recurrent state instead. The rolling
# features themselves are still implemented and tested in src/features/lags.py
# — just not usable as DeepAR inputs.
UNKNOWN_REALS = ["sales"]
TARGET_LAGS = {"sales": [7, 28]}

RAW_COLS = [
    "id", "cat_id", "dept_id", "store_id", "state_id", "d_num", "sales", "date",
    "wday", "month", "year", "weekday", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI", "sell_price",
]
_STR_COLS = [
    "id", "cat_id", "dept_id", "store_id", "state_id", "weekday",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]


def load_and_engineer_features(
    series_ids: list[str] | None = None, min_day: int | None = None
) -> pd.DataFrame:
    """Load train+val, restrict to `series_ids` if given, engineer features
    (on full per-series history, so rolling/price features near a later
    `min_day` truncation are still computed from real prior context), then
    truncate to `min_day` if given (to bound dataset/window count).
    """
    cast = [pl.col(c).cast(pl.Utf8) for c in _STR_COLS]
    train = pl.read_parquet(PROCESSED_DIR / "train.parquet").select(RAW_COLS).with_columns(cast)
    val = pl.read_parquet(PROCESSED_DIR / "val.parquet").select(RAW_COLS).with_columns(cast)
    df = pl.concat([train, val])

    if series_ids is not None:
        df = df.filter(pl.col("id").is_in(series_ids))

    df = snap_flag_for_state(df)
    df = add_price_change_flag(df)
    df = add_rolling_features(df)

    if min_day is not None:
        df = df.filter(pl.col("d_num") >= min_day)

    df = df.with_columns(
        pl.col("d_num").cast(pl.Int64),
        pl.col("wday").cast(pl.Int64),
        pl.col("month").cast(pl.Int64),
        pl.col("year").cast(pl.Int64),
        pl.col("snap").cast(pl.Int64),
        pl.col("price_change_flag").cast(pl.Int64),
        pl.col("sell_price").cast(pl.Float64),
        pl.col("sales").cast(pl.Float64),
    )
    pdf = df.to_pandas()
    for c in _STR_COLS:
        pdf[c] = pdf[c].astype(str).astype("category")
    return pdf


def build_datasets(pdf: pd.DataFrame, max_encoder_length: int, max_prediction_length: int = 28):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    training_cutoff = pdf["d_num"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        pdf[pdf.d_num <= training_cutoff],
        time_idx="d_num",
        target="sales",
        group_ids=["id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_known_categoricals=KNOWN_CATEGORICALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
        lags=TARGET_LAGS,
        target_normalizer=GroupNormalizer(groups=["id"], center=False),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, pdf, predict=True, stop_randomization=True)
    return training, validation


def build_model(
    training_dataset,
    hidden_size: int = 32,
    rnn_layers: int = 2,
    learning_rate: float = 1e-3,
    reduce_on_plateau_patience: int = 2,
):
    from pytorch_forecasting import DeepAR
    from pytorch_forecasting.metrics import NegativeBinomialDistributionLoss

    return DeepAR.from_dataset(
        training_dataset,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        learning_rate=learning_rate,
        loss=NegativeBinomialDistributionLoss(),
        # spec §6.2: "learning rate 1e-3 with a scheduler" — ReduceLROnPlateau
        # on val_loss, pytorch-forecasting's built-in scheduler hook.
        reduce_on_plateau_patience=reduce_on_plateau_patience,
    )
