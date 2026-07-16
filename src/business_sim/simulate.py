"""Business Translation Layer (spec §9): order-up-to-P90 inventory policy
(using the quantile forecasts) vs. a naive order-up-to-P50/mean point-forecast
policy, simulated over the test period, reporting stockout rate and holding
cost for both.

Both policies use the *same* DeepAR forecasts (spec's own model), differing
only in which quantile they order up to — this isolates the value of
quantile-aware ordering itself (the project's core claim, §0) rather than
conflating it with a model-quality difference.

**Policy mechanics (periodic review, order-up-to-S):** the 28-day test
horizon is split into `REVIEW_PERIOD_DAYS`-day cycles. At the start of each
cycle, an order brings inventory up to S = the forecast quantile's summed
demand over that cycle. Demand is then realized day by day; unmet demand is
a stockout (inventory clipped to 0, not backordered); end-of-day inventory
accrues holding cost. Simplifying assumption, documented per spec's own
"document the assumption" instruction: **lead time equals the review
period, and orders arrive instantly at the start of each cycle** — a
standard simplification when lead time <= review period, avoiding the need
to separately model an order pipeline.

**Holding cost assumption (spec §9: "using a reasonable assumed unit
holding cost — document the assumption"):** rather than an arbitrary flat
rate, holding cost is tied to each item's own `sell_price` (already in the
pipeline from `sell_prices.csv`): `ANNUAL_HOLDING_RATE` (20%/year) is a
standard retail rule of thumb for capital + storage + obsolescence cost as
a fraction of unit value; daily cost per unit = sell_price * rate / 365.
"""

from pathlib import Path

import pandas as pd
import polars as pl

from src.backtest.metrics import quantile_col

REPORTS_DIR = Path("reports")

REVIEW_PERIOD_DAYS = 7
ANNUAL_HOLDING_RATE = 0.20
POLICIES = {"p90": quantile_col(0.9), "p50": quantile_col(0.5)}


def simulate_policy(merged: pd.DataFrame, policy_quantile_col: str, review_period: int = REVIEW_PERIOD_DAYS) -> pd.DataFrame:
    """Run the order-up-to-S policy for one quantile column, per series.

    `merged`: one row per (id, d_num) over the test horizon, sorted by
    d_num within each id, with columns [id, d_num, sales, sell_price] plus
    the quantile columns. Returns one row per id: n_days, stockout_days,
    stockout_rate, holding_cost.
    """
    rows = []
    for sid, g in merged.groupby("id", sort=False, observed=True):
        g = g.sort_values("d_num")
        demand = g["sales"].to_numpy()
        forecast_q = g[policy_quantile_col].to_numpy()
        price = g["sell_price"].to_numpy()
        n_days = len(g)

        inventory = 0.0
        stockout_days = 0
        holding_cost = 0.0
        for cycle_start in range(0, n_days, review_period):
            cycle_end = min(cycle_start + review_period, n_days)
            order_up_to = forecast_q[cycle_start:cycle_end].sum()
            order_qty = max(0.0, order_up_to - inventory)
            inventory += order_qty
            for day in range(cycle_start, cycle_end):
                if demand[day] > inventory:
                    stockout_days += 1
                    inventory = 0.0
                else:
                    inventory -= demand[day]
                daily_cost_per_unit = price[day] * ANNUAL_HOLDING_RATE / 365
                holding_cost += inventory * daily_cost_per_unit

        rows.append(
            {
                "id": sid,
                "n_days": n_days,
                "stockout_days": stockout_days,
                "stockout_rate": stockout_days / n_days,
                "holding_cost": holding_cost,
            }
        )
    return pd.DataFrame(rows)


def run_business_sim(forecast: pl.DataFrame, actuals: pl.DataFrame) -> pl.DataFrame:
    """`forecast`: [id, d_num, q0.1, q0.5, q0.9]. `actuals`: [id, d_num,
    sales, sell_price]. Runs both policies, returns one summary row each
    (mean stockout rate, mean + total holding cost across series).
    """
    forecast = forecast.with_columns(pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64))
    actuals = actuals.with_columns(pl.col("id").cast(pl.Utf8), pl.col("d_num").cast(pl.Int64))
    merged = actuals.join(forecast, on=["id", "d_num"], how="inner").to_pandas()
    merged["id"] = merged["id"].astype("category")

    rows = []
    for policy_name, qcol in POLICIES.items():
        per_series = simulate_policy(merged, qcol)
        rows.append(
            {
                "policy": policy_name,
                "order_up_to": qcol,
                "n_series": len(per_series),
                "mean_stockout_rate": per_series["stockout_rate"].mean(),
                "mean_holding_cost_per_series": per_series["holding_cost"].mean(),
                "total_holding_cost": per_series["holding_cost"].sum(),
            }
        )
    return pl.DataFrame(rows)
