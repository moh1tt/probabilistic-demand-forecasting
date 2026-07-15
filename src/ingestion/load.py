"""Raw M5 CSV loaders (spec §4.1)."""

from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw")


def load_sales_wide(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / "sales_train_evaluation.csv")


def load_calendar(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / "calendar.csv", try_parse_dates=True)


def load_sell_prices(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / "sell_prices.csv")
