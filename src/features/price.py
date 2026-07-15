"""Price features (spec §5): sell price + a price-change flag.

Both are treated as *known* future covariates (time_varying_known_reals),
not unknown — M5 convention: the retailer sets prices in advance, so future
sell_price (and therefore whether it differs from the prior day) is known
at forecast time, unlike the target itself.
"""

import polars as pl

KNOWN_REALS = ["sell_price", "price_change_flag"]


def add_price_change_flag(df: pl.DataFrame) -> pl.DataFrame:
    """1 on the first day of a new sell_price for a series (a promo or price
    reset), 0 otherwise. First observed price for a series is not flagged
    (nothing to compare against)."""
    return df.sort(["id", "d_num"]).with_columns(
        (pl.col("sell_price") != pl.col("sell_price").shift(1).over("id"))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("price_change_flag")
    )
