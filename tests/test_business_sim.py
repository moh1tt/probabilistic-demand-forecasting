"""Unit tests for src/business_sim/simulate.py (spec §9)."""

import pandas as pd
import polars as pl

from src.business_sim.simulate import ANNUAL_HOLDING_RATE, run_business_sim, simulate_policy


def _rows(sid, d_nums, sales, q10, q50, q90, price):
    return pd.DataFrame(
        {
            "id": [sid] * len(d_nums),
            "d_num": d_nums,
            "sales": sales,
            "sell_price": price,
            "q0.1": q10,
            "q0.5": q50,
            "q0.9": q90,
        }
    )


def test_stockout_when_order_up_to_below_realized_demand():
    # single 7-day cycle: order_up_to = sum(q0.5) = 2*7 = 14. Demand is 10/day
    # (70 total), so day 1 is covered (14-10=4 left) but every day after
    # stocks out once the 4 leftover units run out.
    merged = _rows(
        "A", list(range(1, 8)), sales=[10] * 7, q10=[1] * 7, q50=[2] * 7, q90=[3] * 7, price=[5.0] * 7
    )
    out = simulate_policy(merged, "q0.5", review_period=7)
    row = out.iloc[0]
    assert row["stockout_days"] == 6
    assert row["stockout_rate"] == 6 / 7


def test_no_stockout_when_order_up_to_covers_realized_demand():
    merged = _rows(
        "A", list(range(1, 8)), sales=[2] * 7, q10=[1] * 7, q50=[2] * 7, q90=[5] * 7, price=[5.0] * 7
    )
    out = simulate_policy(merged, "q0.9", review_period=7)
    row = out.iloc[0]
    assert row["stockout_days"] == 0
    assert row["holding_cost"] > 0


def test_holding_cost_matches_manual_formula_for_single_day():
    # order_up_to=10, demand=4 -> leftover=6, price=8 -> daily cost = 6 * 8 * rate/365.
    merged = _rows("A", [1], sales=[4], q10=[0], q50=[0], q90=[10], price=[8.0])
    out = simulate_policy(merged, "q0.9", review_period=7)
    expected = 6 * 8.0 * ANNUAL_HOLDING_RATE / 365
    assert abs(out.iloc[0]["holding_cost"] - expected) < 1e-9


def test_leftover_inventory_carries_into_next_cycle_reducing_next_order():
    # cycle 1 (days 1-7): order_up_to=70 (10/day), demand=0/day -> leftover 70 at cycle end.
    # cycle 2 (days 8-14): order_up_to=70 again, but leftover already covers it -> order_qty=0,
    # so inventory stays at 70 throughout (never doubles to 140), and no stockout.
    d_nums = list(range(1, 15))
    sales = [0.0] * 14
    q90 = [10.0] * 14
    merged = _rows("A", d_nums, sales=sales, q10=[0] * 14, q50=[0] * 14, q90=q90, price=[1.0] * 14)
    out = simulate_policy(merged, "q0.9", review_period=7)
    assert out.iloc[0]["stockout_days"] == 0
    expected_holding_cost = 14 * 70 * 1.0 * ANNUAL_HOLDING_RATE / 365
    assert abs(out.iloc[0]["holding_cost"] - expected_holding_cost) < 1e-9


def test_run_business_sim_p90_never_worse_stockout_rate_than_p50():
    # p90 >= p50 by construction, so ordering to p90 should never stock out
    # more than ordering to p50, for the same realized demand.
    ids = ["A", "B"]
    rows = []
    for sid in ids:
        for d in range(1, 29):
            rows.append(
                {"id": sid, "d_num": d, "sales": 6.0, "sell_price": 4.0, "q0.1": 2.0, "q0.5": 5.0, "q0.9": 9.0}
            )
    merged = pd.DataFrame(rows)
    forecast = pl.from_pandas(merged[["id", "d_num", "q0.1", "q0.5", "q0.9"]])
    actuals = pl.from_pandas(merged[["id", "d_num", "sales", "sell_price"]])

    results = run_business_sim(forecast, actuals)
    p90 = results.filter(pl.col("policy") == "p90")["mean_stockout_rate"].item()
    p50 = results.filter(pl.col("policy") == "p50")["mean_stockout_rate"].item()
    assert p90 <= p50
