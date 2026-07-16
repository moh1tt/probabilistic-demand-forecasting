"""Phase 5 orchestration (spec §9, §12): the Business Translation Layer.

Retrains DeepAR exactly as Phase 4 does (same population, same cold-start
holdout, same early stopping on val_loss — nothing about training or model
selection changes), then makes one additional inference pass: a predict-mode
dataset extended through the **test** split (days 1914-1941), which the
project has not touched until now (spec §4.3: "the test set is touched only
once, at the end"). This *is* that one time — Phase 5 is the last phase that
needs real forecasts; the dashboard/write-up phases after it only consume
already-computed reports.

Retraining rather than checkpointing: Phase 3/4 both used
`enable_checkpointing=False` (no saved weights to reload), and training is
cheap here (~a few minutes, early-stopped by epoch 4 both prior times) — a
third run costs less than adding checkpoint plumbing would, and keeps this
script self-contained.

Business comparison (spec §9): both policies use the *same* DeepAR test-period
forecast, differing only in which quantile they order up to (P90 vs. P50) —
see src/business_sim/simulate.py for why (isolates the value of quantile-aware
ordering itself, the project's core claim, rather than conflating it with a
model-quality difference).
"""

import sys
from pathlib import Path

import polars as pl
from pytorch_forecasting import TimeSeriesDataSet

from src.backtest.run_deepar import predict_quantiles
from src.backtest.run_deepar_coldstart import train_deepar_coldstart
from src.business_sim.simulate import run_business_sim
from src.models.global_model import load_and_engineer_features

REPORTS_DIR = Path("reports")


def build_test_forecast(model, training: TimeSeriesDataSet, config: dict) -> pl.DataFrame:
    """One-time test-period inference pass (spec §4.3's final touch): extend
    the same population/cold-start config through the test split and derive
    a predict-mode dataset from the already-fitted `training` dataset (reuses
    its normalizers/encoders, doesn't refit anything on test rows).
    """
    pdf_with_test = load_and_engineer_features(
        series_ids=config["series_ids"],
        min_day=config["min_day"],
        coldstart_ids=config["coldstart_ids"],
        coldstart_cutoff_day=config["cutoff_day"],
        include_test=True,
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        training, pdf_with_test, predict=True, stop_randomization=True
    )
    forecast = predict_quantiles(model, test_dataset)
    return forecast, pdf_with_test


def main() -> pl.DataFrame:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model, training, _validation, _pdf, config = train_deepar_coldstart()
    forecast, pdf_with_test = build_test_forecast(model, training, config)
    forecast.write_csv(REPORTS_DIR / "deepar_test_forecast.csv")

    test_days = forecast["d_num"].unique().to_list()
    actuals = pl.from_pandas(pdf_with_test[["id", "d_num", "sales", "sell_price"]]).filter(
        pl.col("d_num").is_in(test_days)
    )
    print(
        f"test forecast rows: {len(forecast)}, actuals rows: {len(actuals)}, "
        f"series: {forecast['id'].n_unique()}",
        flush=True,
    )

    results = run_business_sim(forecast, actuals)
    results.write_csv(REPORTS_DIR / "business_sim_results.csv")
    print(results, flush=True)
    return results


if __name__ == "__main__":
    main()
