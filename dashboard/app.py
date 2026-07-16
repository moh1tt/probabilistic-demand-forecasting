"""Streamlit dashboard (spec §10.4). Run: `streamlit run dashboard/app.py`.

Three required views, each reading a report already produced by earlier
phases (nothing here retrains or re-predicts anything):
1. Per-series forecast + P10/P50/P90 bands, series selector.
   -> reports/dashboard_series_forecasts.csv (dashboard/prepare_data.py)
2. Cold-start vs. warm-start accuracy comparison.
   -> reports/phase4_coldstart_results.csv (Phase 4)
3. Business sim: stockout rate + holding cost, policy comparison.
   -> reports/business_sim_results.csv (Phase 5)
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPORTS_DIR = Path("reports")

st.set_page_config(page_title="Global Demand Forecaster", layout="wide")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    if not path.exists():
        st.error(
            f"Missing `{path}`. Run the pipeline first (`./run.sh`), then "
            f"`python -m dashboard.prepare_data` before starting the dashboard."
        )
        st.stop()
    return pd.read_csv(path)


st.title("Global Probabilistic Demand Forecaster")
st.caption("M5 / Walmart retail demand — DeepAR global model, cold-start holdout, business simulation.")

series_view, coldstart_view, business_view = st.tabs(
    ["Per-series forecast", "Cold-start vs. warm-start", "Business simulation"]
)

# --- View 1: per-series forecast + P10/P50/P90 bands -----------------------
with series_view:
    df = load_csv("dashboard_series_forecasts.csv")

    col1, col2 = st.columns([1, 3])
    with col1:
        coldstart_only = st.checkbox("Cold-start series only", value=False)
        cat_options = ["All"] + sorted(df["cat_id"].unique().tolist())
        cat_filter = st.selectbox("Category", cat_options)

    filtered_ids = df
    if coldstart_only:
        filtered_ids = filtered_ids[filtered_ids["is_coldstart"]]
    if cat_filter != "All":
        filtered_ids = filtered_ids[filtered_ids["cat_id"] == cat_filter]
    id_options = sorted(filtered_ids["id"].unique().tolist())

    if not id_options:
        st.warning("No series match the current filters.")
    else:
        with col1:
            selected_id = st.selectbox("Series", id_options)

        series = df[df["id"] == selected_id].sort_values("d_num")
        is_cold = bool(series["is_coldstart"].iloc[0])
        volume_segment = series["volume_segment"].iloc[0]

        with col1:
            st.metric("Segment", "Cold-start" if is_cold else "Warm-start")
            st.metric("Volume", volume_segment.replace("_", " ").title())

        with col2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=series["d_num"], y=series["sales"], mode="lines",
                    name="Actual sales", line=dict(color="#3b6ea5"),
                )
            )
            forecast_rows = series.dropna(subset=["q0.5"])
            if not forecast_rows.empty:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_rows["d_num"], y=forecast_rows["q0.9"], mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_rows["d_num"], y=forecast_rows["q0.1"], mode="lines",
                        line=dict(width=0), fill="tonexty", fillcolor="rgba(217,95,2,0.2)",
                        name="P10-P90 band", hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_rows["d_num"], y=forecast_rows["q0.5"], mode="lines",
                        name="Forecast (P50)", line=dict(color="#d95f02", dash="dash"),
                    )
                )
            fig.update_layout(
                xaxis_title="day", yaxis_title="units sold",
                margin=dict(l=10, r=10, t=30, b=10), height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Forecast band covers the test period (days 1914-1941, the model's "
            "one-time final evaluation). Earlier days show real sales history only."
        )

# --- View 2: cold-start vs. warm-start accuracy -----------------------------
with coldstart_view:
    results = load_csv("phase4_coldstart_results.csv")

    start_seg = results[results["segment_type"] == "start"]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=start_seg["segment"], y=start_seg["wql"], name="WQL"))
    fig2.add_trace(go.Bar(x=start_seg["segment"], y=start_seg["mase"], name="MASE"))
    fig2.update_layout(
        barmode="group", yaxis_title="metric value",
        title="Cold-start vs. warm-start: WQL and MASE",
        margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)

    n_cold = int(start_seg.loc[start_seg["segment"] == "cold_start", "n_series"].iloc[0])
    n_warm = int(start_seg.loc[start_seg["segment"] == "warm_start", "n_series"].iloc[0])
    st.caption(
        f"Cold-start: {n_cold} series (all history before day 1859 removed, simulating brand-new "
        f"products with <28 days of visible history). Warm-start: {n_warm} series with full history. "
        "Cold-start WQL is essentially tied with warm-start — the global model's static/category "
        "embeddings carry real signal for brand-new items (see PROGRESS.md Phase 4 for the honest "
        "caveat on the MASE comparison specifically)."
    )

    volume_seg = results[results["segment_type"] == "volume"]
    with st.expander("High-volume vs. long-tail (secondary segmentation, spec §8)"):
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=volume_seg["segment"], y=volume_seg["wql"], name="WQL"))
        fig3.add_trace(go.Bar(x=volume_seg["segment"], y=volume_seg["mase"], name="MASE"))
        fig3.update_layout(barmode="group", yaxis_title="metric value", height=380)
        st.plotly_chart(fig3, use_container_width=True)

# --- View 3: business simulation --------------------------------------------
with business_view:
    sim = load_csv("business_sim_results.csv")
    sim = sim.set_index("policy")

    col1, col2 = st.columns(2)
    with col1:
        fig4 = go.Figure(
            go.Bar(x=sim.index, y=sim["mean_stockout_rate"] * 100, marker_color=["#d95f02", "#7570b3"])
        )
        fig4.update_layout(
            title="Mean stockout rate by policy", yaxis_title="stockout rate (%)",
            margin=dict(l=10, r=10, t=40, b=10), height=380,
        )
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        fig5 = go.Figure(
            go.Bar(x=sim.index, y=sim["total_holding_cost"], marker_color=["#d95f02", "#7570b3"])
        )
        fig5.update_layout(
            title="Total holding cost by policy ($)", yaxis_title="holding cost ($)",
            margin=dict(l=10, r=10, t=40, b=10), height=380,
        )
        st.plotly_chart(fig5, use_container_width=True)

    p90 = sim.loc["p90"]
    p50 = sim.loc["p50"]
    st.caption(
        f"Order-up-to-P90 vs. order-up-to-P50, same DeepAR test-period forecast for both (spec §9): "
        f"P90 cuts the stockout rate from {p50['mean_stockout_rate']:.1%} to {p90['mean_stockout_rate']:.1%} "
        f"(~{p50['mean_stockout_rate'] / p90['mean_stockout_rate']:.1f}x fewer stockouts) at "
        f"~{p90['total_holding_cost'] / p50['total_holding_cost']:.1f}x the holding cost "
        f"(${p50['total_holding_cost']:,.0f} -> ${p90['total_holding_cost']:,.0f} across "
        f"{int(p90['n_series'])} series). Holding cost assumes 20%/year of each item's own sell price, "
        "charged daily on end-of-day inventory (see PROGRESS.md Phase 5)."
    )
