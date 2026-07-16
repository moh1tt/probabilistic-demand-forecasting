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
from src.features.coldstart import add_coldstart_flag, truncate_coldstart_history
from src.features.lags import add_rolling_features, rolling_feature_names
from src.features.price import KNOWN_REALS as PRICE_KNOWN_REALS
from src.features.price import add_price_change_flag
from src.features.static import STATIC_CATEGORICALS

PROCESSED_DIR = Path("data/processed")

# coldstart_flag (spec §5) is deterministically known ahead of time (it's a
# function of the time index and each series' own first-observed day, never
# the target's future values), so it's safe as a known covariate for both
# encoder and decoder, same as the calendar/price known reals.
KNOWN_REALS = CALENDAR_KNOWN_REALS + PRICE_KNOWN_REALS + ["coldstart_flag"]
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
    series_ids: list[str] | None = None,
    min_day: int | None = None,
    coldstart_ids: list[str] | None = None,
    coldstart_cutoff_day: int | None = None,
    include_test: bool = False,
) -> pd.DataFrame:
    """Load train+val (plus test if `include_test`), restrict to
    `series_ids` if given, engineer features (on full per-series history, so
    rolling/price features near a later `min_day` truncation are still
    computed from real prior context), then truncate to `min_day` if given
    (to bound dataset/window count).

    `coldstart_ids`/`coldstart_cutoff_day` (spec §4.4): if given, every row
    before `coldstart_cutoff_day` is removed for those series *before*
    anything else is computed — so rolling features and the coldstart_flag
    itself are derived only from each cold-start series' deliberately
    shortened visible history, never its real (deleted) prior history.

    `include_test`: only Phase 5's business simulation sets this — it needs
    the model's forecast target (encoder ends in val, decoder covers test)
    plus real test-period sales/prices to run the inventory sim against.
    Nothing upstream of this flag (training, model selection, early
    stopping) ever sees test rows; this is purely for the one-time final
    read spec §4.3 allows.
    """
    cast = [pl.col(c).cast(pl.Utf8) for c in _STR_COLS]
    train = pl.read_parquet(PROCESSED_DIR / "train.parquet").select(RAW_COLS).with_columns(cast)
    val = pl.read_parquet(PROCESSED_DIR / "val.parquet").select(RAW_COLS).with_columns(cast)
    parts = [train, val]
    if include_test:
        test = pl.read_parquet(PROCESSED_DIR / "test.parquet").select(RAW_COLS).with_columns(cast)
        parts.append(test)
    df = pl.concat(parts)

    if series_ids is not None:
        df = df.filter(pl.col("id").is_in(series_ids))

    if coldstart_ids:
        df = truncate_coldstart_history(df, coldstart_ids, coldstart_cutoff_day)
    df = add_coldstart_flag(df)

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
        pl.col("coldstart_flag").cast(pl.Int64),
        pl.col("sell_price").cast(pl.Float64),
        pl.col("sales").cast(pl.Float64),
    )
    pdf = df.to_pandas()
    for c in _STR_COLS:
        pdf[c] = pdf[c].astype(str).astype("category")
    return pdf


def build_datasets(
    pdf: pd.DataFrame,
    max_encoder_length: int,
    max_prediction_length: int = 28,
    min_encoder_length: int | None = None,
    lags: dict | None = None,
):
    """`min_encoder_length` defaults to `max_encoder_length // 2` (Phase 3's
    original config). Phase 4 passes a much smaller value (1) so cold-start
    series — which only have a handful of real days before the evaluation
    origin by design (spec §4.4) — still produce a valid (short-encoder)
    sample instead of being silently dropped by the library's own
    min-encoder-length filter; verified this is what actually happens (not
    an assumption) in tests/test_no_leakage.py.

    `lags` defaults to `TARGET_LAGS` (Phase 3's `{"sales": [7, 28]}`), but
    Phase 4 passes `{}` instead. Reason (found by testing, not assumed):
    `TimeSeriesDataSet._preprocess_data` unconditionally drops each series'
    first `max(lags)` rows before building any window, to avoid NaN lag
    values — with lag-28 that drops a cold-start series' entire ~27-day
    visible pre-origin history (plus part of its decoder window), so it can
    never form a valid sample no matter how low `min_encoder_length` is set.
    This is a hard structural conflict between target-lag features and a
    <28-day cold-start holdout, not a tunable threshold. Phase 4 drops target
    lags entirely rather than special-casing cold-start series with a
    different feature set (DeepAR's recurrent state already sees the raw
    `sales` values in whatever encoder it gets, lags or not).

    `event_name_1/2`/`event_type_1/2` get an explicit `NaNLabelEncoder(add_nan=True)`
    (found by testing Phase 5's test-period inference, not assumed): these
    categorical encoders are fit on whatever event names appear in the rows
    passed here, and rare calendar events (e.g. a specific religious holiday)
    can easily occur in the test split but never in the train+val window they
    were fit on — without `add_nan=True` the library raises a hard
    `KeyError` on any such unseen category instead of treating it as unknown.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    if min_encoder_length is None:
        min_encoder_length = max_encoder_length // 2
    if lags is None:
        lags = TARGET_LAGS
    training_cutoff = pdf["d_num"].max() - max_prediction_length
    event_categoricals = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

    training = TimeSeriesDataSet(
        pdf[pdf.d_num <= training_cutoff],
        time_idx="d_num",
        target="sales",
        group_ids=["id"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_known_categoricals=KNOWN_CATEGORICALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
        categorical_encoders={c: NaNLabelEncoder(add_nan=True) for c in event_categoricals},
        lags=lags,
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
