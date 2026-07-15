"""Phase 3 orchestration (spec §6.2, §12): train the global DeepAR model,
generate P10/P50/P90 forecasts for the val period, and compare against the
Phase 2 baselines using the same WQL/MASE metrics on the same data.

Scope (checked with the user given hardware constraints — see PROGRESS.md
Phase 3 notes): ~2,000 series (stratified by cat_id, same method as the
Phase 2 baseline subset), last 365 days of train history. DeepAR is trained
*once* with early stopping (not re-fit per rolling origin like the cheap
classical baselines) and evaluated on the val split (days 1886-1913) — which
is exactly the Phase 2 harness's last origin's evaluation window, so the
comparison in reports/phase3_comparison.csv is apples-to-apples against
`reports/baseline_backtest_results.csv` (origin=1885, segment=overall) without
needing to retrain DeepAR per origin.

Test set (days 1914-1941) is untouched here, per spec §4.3 "touched only
once, at the end."
"""

from pathlib import Path

import lightning.pytorch as L
import polars as pl
from lightning.pytorch.callbacks import EarlyStopping

from src.backtest.metrics import QUANTILES, mase_polars, quantile_col, weighted_quantile_loss
from src.models.baselines import select_baseline_subset
from src.models.global_model import build_datasets, build_model, load_and_engineer_features

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

N_SERIES = 2000
HISTORY_DAYS = 365
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 28
BATCH_SIZE = 128
MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 3
NUM_WORKERS = 4
SUBSET_SEED = 43  # different from Phase 2's baseline subset seed (42) on purpose:
# DeepAR's value proposition is training jointly across many series, so its
# training population doesn't need to be identical to the baselines' subset,
# it just needs to be a comparably-selected, documented sample.


def select_training_series(n: int = N_SERIES, seed: int = SUBSET_SEED) -> list[str]:
    cols = ["id", "cat_id"]
    ids_cats = (
        pl.read_parquet(PROCESSED_DIR / "train.parquet")
        .select(cols)
        .with_columns(pl.col("id").cast(pl.Utf8), pl.col("cat_id").cast(pl.Utf8))
        .unique(maintain_order=True)
    )
    return select_baseline_subset(ids_cats, n=n, seed=seed)


def train_deepar():
    series_ids = select_training_series()
    train_max_day = (
        pl.read_parquet(PROCESSED_DIR / "train.parquet").select(pl.col("d_num").max()).item()
    )
    min_day = train_max_day - HISTORY_DAYS

    print(f"selecting {len(series_ids)} series, history from day {min_day}", flush=True)
    pdf = load_and_engineer_features(series_ids=series_ids, min_day=min_day)
    print(f"engineered features: {len(pdf)} rows", flush=True)

    training, validation = build_datasets(
        pdf, max_encoder_length=MAX_ENCODER_LENGTH, max_prediction_length=MAX_PREDICTION_LENGTH
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
    print(f"stopped at epoch {trainer.current_epoch} (best val_loss checkpointed via early stopping)", flush=True)
    return model, training, validation, pdf, series_ids


def predict_quantiles(model, validation) -> pl.DataFrame:
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    pred = model.predict(
        val_loader,
        mode="quantiles",
        mode_kwargs={"quantiles": list(QUANTILES)},
        return_index=True,
        trainer_kwargs={"accelerator": "gpu", "devices": 1},
    )
    output = pred.output.cpu().numpy()  # [n_series, horizon, n_quantiles]
    index_df = pred.index  # columns: id, d_num (first prediction day per series)

    rows = []
    for i, row in index_df.iterrows():
        sid = row["id"]
        start_day = int(row["d_num"])
        for h in range(output.shape[1]):
            rows.append(
                {
                    "id": sid,
                    "d_num": start_day + h,
                    quantile_col(0.1): float(output[i, h, 0]),
                    quantile_col(0.5): float(output[i, h, 1]),
                    quantile_col(0.9): float(output[i, h, 2]),
                }
            )
    return pl.DataFrame(rows)


def evaluate(forecast: pl.DataFrame, pdf_actuals: pl.DataFrame, series_ids: list[str]) -> pl.DataFrame:
    actuals = pl.from_pandas(pdf_actuals[["id", "d_num", "sales", "cat_id"]])
    actuals = actuals.with_columns(pl.col("id").cast(pl.Utf8), pl.col("cat_id").cast(pl.Utf8))
    forecast = forecast.with_columns(pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64))
    actuals = actuals.with_columns(pl.col("d_num").cast(pl.Int64))

    merged = actuals.join(forecast, on=["id", "d_num"], how="inner")
    y_true = merged["sales"].to_numpy().astype(float)
    q_preds = {q: merged[quantile_col(q)].to_numpy().astype(float) for q in QUANTILES}
    wql = weighted_quantile_loss(y_true, q_preds)

    train_max_day = (
        pl.read_parquet(PROCESSED_DIR / "train.parquet").select(pl.col("d_num").max()).item()
    )
    history = pl.from_pandas(pdf_actuals[["id", "d_num", "sales"]]).filter(
        pl.col("d_num") <= train_max_day
    )
    history = history.with_columns(pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64))
    m = mase_polars(history, merged, m=7)

    return pl.DataFrame(
        [{"model": "deepar", "origin": train_max_day, "segment": "overall", "n_series": merged["id"].n_unique(), "wql": wql, "mase": m}]
    )


def main() -> pl.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model, training, validation, pdf, series_ids = train_deepar()
    forecast = predict_quantiles(model, validation)
    forecast.write_csv(REPORTS_DIR / "deepar_val_forecast.csv")

    results = evaluate(forecast, pdf, series_ids)
    results.write_csv(REPORTS_DIR / "deepar_val_results.csv")
    print(results, flush=True)
    return results


if __name__ == "__main__":
    main()
