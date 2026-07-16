"""Cold-start holdout (spec §4.4) and cold-start flag feature (spec §5).

Design note on the cutoff, since spec §4.4's literal "day 1400" doesn't
transfer to our actual data range (see PROGRESS.md Phase 1 deviation: our
train/val/test split is anchored to the real max day, 1941, not the higher
day count the spec text assumed): what matters per §4.4 is that cold-start
series have **fewer than 28 days of history at the evaluation origin**, not
the literal day number. `coldstart_cutoff_day()` derives a cutoff from
whatever origin day is actually in use, giving exactly 27 visible days of
history at that origin (comfortably under the 28-day threshold).
"""

import polars as pl

COLDSTART_FRAC = 0.05  # spec §4.4: "~5% of series"
COLDSTART_HORIZON = 28  # spec §4.4/§5: "<28 days of history"
COLDSTART_VISIBLE_DAYS = COLDSTART_HORIZON - 1  # 27 — comfortably under the threshold


def coldstart_cutoff_day(origin_day: int, visible_days: int = COLDSTART_VISIBLE_DAYS) -> int:
    """Cutoff day such that a cold-start series has exactly `visible_days`
    (<28) days of history as of `origin_day` (the last encoder day)."""
    return origin_day - visible_days + 1


def select_coldstart_series(
    ids_cats: pl.DataFrame, frac: float = COLDSTART_FRAC, seed: int = 44
) -> list[str]:
    """Stratified sample of series IDs (by cat_id, proportional allocation) to
    simulate as brand-new products — same method as Phase 2/3's
    `select_baseline_subset`, reused here for the cold-start population.
    `ids_cats` should have columns [id, cat_id] and defines the population
    to sample from (e.g. the DeepAR training subset, not necessarily the
    full 30,490-series universe — documented in run_deepar_coldstart.py).
    """
    from src.models.baselines import select_baseline_subset

    n = round(frac * ids_cats.select(pl.col("id").n_unique()).item())
    return select_baseline_subset(ids_cats, n=n, seed=seed)


def truncate_coldstart_history(
    df: pl.DataFrame, coldstart_ids: list[str], cutoff_day: int
) -> pl.DataFrame:
    """Remove every row before `cutoff_day` for series in `coldstart_ids`
    (spec §4.4) — simulates these series as new products first observed at
    `cutoff_day`. Applied at the raw-row level, not just at prediction time,
    so the truncation holds for every window any downstream modeling code
    constructs from this frame, not just the final evaluation origin —
    there is simply no pre-cutoff data left to leak into any fold.
    """
    is_coldstart = pl.col("id").is_in(coldstart_ids)
    return df.filter(~is_coldstart | (pl.col("d_num") >= cutoff_day))


def add_coldstart_flag(df: pl.DataFrame, horizon: int = COLDSTART_HORIZON) -> pl.DataFrame:
    """Boolean feature (spec §5): whether a series has fewer than `horizon`
    days of history as of day t, computed from each id's own first
    *observed* row in `df`.

    Call this after `truncate_coldstart_history` (and before restricting the
    frame to a bounded training window) — the deliberately-shortened
    visible history of a simulated cold-start series is what should drive
    the flag, while a normal series' true (much earlier) release day keeps
    its flag at 0 throughout, regardless of any later window truncation.
    """
    first_day = df.group_by("id").agg(pl.col("d_num").min().alias("first_day"))
    df = df.join(first_day, on="id", how="left")
    # Days of history as of day t, inclusive of t's own row: t - first_day + 1.
    days_of_history = pl.col("d_num") - pl.col("first_day") + 1
    return df.with_columns(
        (days_of_history < horizon).cast(pl.Int64).alias("coldstart_flag")
    ).drop("first_day")
