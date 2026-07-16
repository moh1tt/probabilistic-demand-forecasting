"""Phase 4 orchestration (spec §4.4, §6.3, §8, §12): retrain the global
DeepAR model with a documented cold-start holdout baked into its training
population, then report warm-start vs. cold-start WQL/MASE separately.

Population: the same ~2,000-series subset used for Phase 3 (`select_training_series`,
same stratified-by-cat_id method, seed=43) — reusing it keeps this a direct,
apples-to-apples extension of Phase 3 rather than a new unrelated run. Within
that population, ~5% (spec §4.4) are additionally chosen (stratified by
cat_id, seed=44) to be simulated new products: every row before
`coldstart_cutoff_day` is deleted for those series (src/features/coldstart.py)
before any feature engineering or dataset windowing happens, so no training
fold — not just the final one — ever sees their pre-cutoff history (verified
programmatically in tests/test_no_leakage.py, not just by design, per spec
§4.4's own instruction).

Model architecture is unchanged from Phase 3 (spec §6.3: "the point is the
evaluation methodology, not a special cold-start model") — the dataset
config changes twofold, both found by testing (not assumed) before
committing to the real run, both documented in PROGRESS.md Phase 4 notes:
1. `min_encoder_length` (60/2=30 in Phase 3) lowered to 1 — otherwise the
   library's own min-encoder-length filter silently drops every cold-start
   series from training and evaluation.
2. Target lags (`{"sales": [7, 28]}` in Phase 3) dropped entirely — the
   library unconditionally removes each series' first `max(lags)` rows
   before building any window (to avoid NaN lag values), which for lag-28
   deletes a cold-start series' entire ~27-day visible history. This is a
   structural conflict between target-lag features and a <28-day cold-start
   holdout, not a tunable threshold.
Both are TimeSeriesDataSet windowing/feature-set parameters, not model
architecture changes.
"""

import sys
from pathlib import Path

import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
import polars as pl

from src.backtest.metrics import QUANTILES, mase_polars, quantile_col, weighted_quantile_loss
from src.backtest.run_baselines import add_volume_segment
from src.backtest.run_deepar import predict_quantiles, select_training_series
from src.features.coldstart import COLDSTART_FRAC, coldstart_cutoff_day, select_coldstart_series
from src.models.global_model import build_datasets, build_model, load_and_engineer_features

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

HISTORY_DAYS = 365
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 28
MIN_ENCODER_LENGTH = 1  # see module docstring — required for cold-start series to survive
BATCH_SIZE = 128
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 3
NUM_WORKERS = 4
COLDSTART_SEED = 44


def train_deepar_coldstart():
    series_ids = select_training_series()
    train_max_day = (
        pl.read_parquet(PROCESSED_DIR / "train.parquet").select(pl.col("d_num").max()).item()
    )
    val_max_day = pl.read_parquet(PROCESSED_DIR / "val.parquet").select(pl.col("d_num").max()).item()
    min_day = train_max_day - HISTORY_DAYS
    origin_day = val_max_day - MAX_PREDICTION_LENGTH
    cutoff_day = coldstart_cutoff_day(origin_day)

    ids_cats = (
        pl.read_parquet(PROCESSED_DIR / "train.parquet")
        .select(["id", "cat_id"])
        .filter(pl.col("id").is_in(series_ids))
        .with_columns(pl.col("id").cast(pl.Utf8), pl.col("cat_id").cast(pl.Utf8))
        .unique(maintain_order=True)
    )
    coldstart_ids = select_coldstart_series(ids_cats, frac=COLDSTART_FRAC, seed=COLDSTART_SEED)
    print(
        f"cold-start holdout: {len(coldstart_ids)}/{len(series_ids)} series "
        f"({COLDSTART_FRAC:.0%}), cutoff_day={cutoff_day} (origin_day={origin_day})",
        flush=True,
    )

    pdf = load_and_engineer_features(
        series_ids=series_ids,
        min_day=min_day,
        coldstart_ids=coldstart_ids,
        coldstart_cutoff_day=cutoff_day,
    )
    print(f"engineered features: {len(pdf)} rows", flush=True)

    training, validation = build_datasets(
        pdf,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        min_encoder_length=MIN_ENCODER_LENGTH,
        lags={},  # see module docstring — target lags are structurally incompatible with the holdout
    )
    print(f"train samples: {len(training)}, val samples: {len(validation)}", flush=True)

    model = build_model(training)
    train_loader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, mode="min")
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(
        f"stopped at epoch {trainer.current_epoch} (best val_loss checkpointed via early stopping)",
        flush=True,
    )
    config = {
        "series_ids": series_ids,
        "min_day": min_day,
        "coldstart_ids": coldstart_ids,
        "cutoff_day": cutoff_day,
        "origin_day": origin_day,
    }
    return model, training, validation, pdf, config


def evaluate_segmented(
    forecast: pl.DataFrame, pdf_actuals, coldstart_ids: list[str], origin_day: int
) -> pl.DataFrame:
    import pandas as pd

    actuals = pl.from_pandas(pdf_actuals[["id", "d_num", "sales"]]).with_columns(
        pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64)
    )
    forecast = forecast.with_columns(pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64))
    merged = actuals.join(forecast, on=["id", "d_num"], how="inner")

    is_coldstart = pl.col("id").is_in(coldstart_ids)
    merged = merged.with_columns(
        pl.when(is_coldstart).then(pl.lit("cold_start")).otherwise(pl.lit("warm_start")).alias("start_segment")
    )

    train = pl.read_parquet(PROCESSED_DIR / "train.parquet").select(
        ["id", "cat_id", "d_num", "sales"]
    ).with_columns(pl.col("id").cast(pl.Utf8), pl.col("cat_id").cast(pl.Utf8))
    train = add_volume_segment(train, origin_day)
    merged = merged.join(
        train.select(["id", "volume_segment"]).unique(), on="id", how="left"
    )

    history = actuals.filter(pl.col("d_num") <= origin_day)

    def score(sub: pl.DataFrame, segment_type: str, segment: str) -> dict:
        y_true = sub["sales"].to_numpy().astype(float)
        q_preds = {q: sub[quantile_col(q)].to_numpy().astype(float) for q in QUANTILES}
        wql = weighted_quantile_loss(y_true, q_preds)
        sub_history = history.filter(pl.col("id").is_in(sub["id"].unique().to_list()))
        m = mase_polars(sub_history, sub, m=7)
        return {
            "model": "deepar_coldstart",
            "origin": origin_day,
            "segment_type": segment_type,
            "segment": segment,
            "n_series": sub["id"].n_unique(),
            "wql": wql,
            "mase": m,
        }

    rows = [score(merged, "overall", "overall")]
    for seg in ["cold_start", "warm_start"]:
        sub = merged.filter(pl.col("start_segment") == seg)
        if sub.height > 0:
            rows.append(score(sub, "start", seg))
    for seg in ["high_volume", "long_tail"]:
        sub = merged.filter(pl.col("volume_segment") == seg)
        if sub.height > 0:
            rows.append(score(sub, "volume", seg))

    return pl.DataFrame(rows)


def main() -> pl.DataFrame:
    # polars' wide-table rendering can emit box-drawing characters that
    # Windows' default console codepage (cp1252) can't encode — found while
    # dry-running this script; doesn't affect the CSV output, only the
    # terminal echo of `results` below.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model, training, validation, pdf, config = train_deepar_coldstart()
    forecast = predict_quantiles(model, validation)
    forecast.write_csv(REPORTS_DIR / "deepar_coldstart_val_forecast.csv")

    results = evaluate_segmented(forecast, pdf, config["coldstart_ids"], config["origin_day"])
    results.write_csv(REPORTS_DIR / "phase4_coldstart_results.csv")
    print(results, flush=True)
    return results


if __name__ == "__main__":
    main()
