"""Calendar/event features (spec §5). Already joined onto the processed
table in Phase 1 (src/ingestion/build_dataset.py); this module just names
the standard feature groups so downstream code (global_model.py) has one
place to look them up, per spec's "one function per feature group,
independently testable" directive.
"""

import polars as pl

KNOWN_REALS = ["wday", "month", "year", "snap"]
KNOWN_CATEGORICALS = ["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
SNAP_FLAGS = ["snap_CA", "snap_TX", "snap_WI"]


def snap_flag_for_state(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse the three state-specific SNAP columns into the single flag
    relevant to each row's own state_id, so the model sees one clean
    "is SNAP day" signal per series instead of three columns where only one
    is ever relevant to a given store.
    """
    return df.with_columns(
        pl.when(pl.col("state_id") == "CA")
        .then(pl.col("snap_CA"))
        .when(pl.col("state_id") == "TX")
        .then(pl.col("snap_TX"))
        .when(pl.col("state_id") == "WI")
        .then(pl.col("snap_WI"))
        .otherwise(0)
        .alias("snap")
    )
