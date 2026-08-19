import itertools
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MONTHLY_PATH = "data/lastmile_monthly.parquet"
SHIPMENT_PATH = "/Workspace/Users/sgt4gul@mckessoncorp.onmicrosoft.com/Last Mile/data/shipment_report.parquet"

st.set_page_config(page_title="Last Mile Analytics", layout="wide")
st.markdown("<h3 style='margin-top:-1rem;margin-bottom:0.5rem'>Last Mile Analytics</h3>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading monthly data...")
def load_monthly(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["DATE"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MONTH_NUM"].astype(str) + "-01"
    )
    return df


@st.cache_data(show_spinner="Loading shipment data...")
def load_shipment(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Load monthly data
# ---------------------------------------------------------------------------
if not os.path.exists(MONTHLY_PATH):
    st.error(f"Monthly data not found at `{MONTHLY_PATH}`. Run the notebook first.")
    st.stop()

monthly = load_monthly(MONTHLY_PATH)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_trend, tab_corr, tab_sim, tab_geo = st.tabs(["📈 Trend", "🔗 Correlation", "🎯 Similarity", "🗺️ Geo Map"])

# ========================== TAB 1: TREND ==========================
with tab_trend:
    carriers = sorted(monthly["CARRIER_SCAC_NAME"].dropna().unique())
    sel_carriers = st.multiselect(
        "Filter by Carrier (leave empty for all)", carriers, key="trend_carrier"
    )
    filtered = monthly if not sel_carriers else monthly[monthly["CARRIER_SCAC_NAME"].isin(sel_carriers)]

    trend = (
        filtered.groupby("DATE", as_index=False)
        .agg(
            LASTMILE_BASE_COST=("LASTMILE_BASE_COST", "sum"),
            LASTMILE_FUEL_COST=("LASTMILE_FUEL_COST", "sum"),
            LASTMILE_MISC_COST=("LASTMILE_MISC_COST", "sum"),
            LASTMILE_TOTAL_COST=("LASTMILE_TOTAL_COST", "sum"),
            TOTAL_ROUTES=("ROUTE_COUNT_VAL_ROUTE_LVL", "sum"),
            TOTAL_STOPS=("STOP_COUNT_VAL_ROUTE_LVL", "sum"),
            TOTAL_TOTES=("TOTE_COUNT_VAL_ROUTE_LVL", "sum"),
        )
        .sort_values("DATE")
    )
    trend["MOM_PCT"] = trend["LASTMILE_TOTAL_COST"].pct_change() * 100

    # Row 1: Cost components + MoM change
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for col, name in [
            ("LASTMILE_BASE_COST", "Base"),
            ("LASTMILE_FUEL_COST", "Fuel"),
            ("LASTMILE_MISC_COST", "Misc"),
        ]:
            fig.add_trace(
                go.Scatter(x=trend["DATE"], y=trend[col], mode="lines", stackgroup="cost", name=name)
            )
        fig.update_layout(title="Cost Components", yaxis_title="$", height=300, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(
            trend.dropna(subset=["MOM_PCT"]),
            x="DATE", y="MOM_PCT",
            title="MoM % Change",
            labels={"MOM_PCT": "%", "DATE": ""},
            color="MOM_PCT", color_continuous_scale="RdYlGn_r", height=300,
        )
        fig2.update_layout(margin=dict(t=30, b=30), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Volume trend (full width)
    fig3 = px.line(
        trend, x="DATE", y=["TOTAL_ROUTES", "TOTAL_STOPS", "TOTAL_TOTES"],
        title="Monthly Volume (Routes / Stops / Totes)",
        labels={"value": "Count", "DATE": "", "variable": "Metric"},
        height=280,
    )
    fig3.update_layout(margin=dict(t=30, b=30))
    st.plotly_chart(fig3, use_container_width=True)

# ========================== TAB 2: CORRELATION ==========================
with tab_corr:
    st.caption("Pearson correlation between cost and volume metrics across all route-months.")

    numeric_cols = [
        "ROUTE_COUNT_VAL_ROUTE_LVL", "STOP_COUNT_VAL_ROUTE_LVL", "TOTE_COUNT_VAL_ROUTE_LVL",
        "LASTMILE_BASE_COST", "LASTMILE_FUEL_COST", "LASTMILE_MISC_COST",
        "LASTMILE_TOTAL_COST", "AVG_COST_PER_ROUTE", "AVG_COST_PER_STOP", "AVG_COST_PER_TOTE",
    ]
    short_labels = ["Routes", "Stops", "Totes", "Base$", "Fuel$", "Misc$", "Total$", "$/Route", "$/Stop", "$/Tote"]

    corr = monthly[numeric_cols].corr()
    corr_vals = corr.values
    n = len(short_labels)

    # Lower triangle only (cleaner, no redundancy)
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    masked_corr = np.where(mask_upper, np.nan, corr_vals)
    text_matrix = [
        ["" if np.isnan(masked_corr[i][j]) else f"{masked_corr[i][j]:.2f}" for j in range(n)]
        for i in range(n)
    ]

    scatter_df = monthly.query("LASTMILE_TOTAL_COST > 0 and TOTE_COUNT_VAL_ROUTE_LVL > 0").copy()
    sel_corr_carriers = st.multiselect(
        "Filter carriers (scatter)",
        sorted(scatter_df["CARRIER_SCAC_NAME"].dropna().unique()),
        key="corr_carrier",
    )
    if sel_corr_carriers:
        scatter_df = scatter_df[scatter_df["CARRIER_SCAC_NAME"].isin(sel_corr_carriers)]

    fig_scatter = px.scatter(
        scatter_df,
        x="TOTE_COUNT_VAL_ROUTE_LVL",
        y="LASTMILE_TOTAL_COST",
        color="CARRIER_SCAC_NAME",
        opacity=0.35,
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="#111111",
        labels={
            "TOTE_COUNT_VAL_ROUTE_LVL": "Tote Count",
            "LASTMILE_TOTAL_COST": "Total Cost ($)",
            "CARRIER_SCAC_NAME": "Carrier",
        },
        title="Total Cost vs Tote Volume (black line = overall OLS)",
        height=360,
    )
    fig_scatter.update_layout(margin=dict(t=45, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ========================== TAB 3: SIMILARITY ==========================
with tab_sim:
    st.caption("Cosine similarity of carriers based on standardized cost & volume profiles.")

    carrier_profile = (
        monthly.groupby("CARRIER_SCAC_NAME", as_index=False)
        .agg(
            AVG_BASE_COST=("LASTMILE_BASE_COST", "mean"),
            AVG_FUEL_COST=("LASTMILE_FUEL_COST", "mean"),
            AVG_MISC_COST=("LASTMILE_MISC_COST", "mean"),
            AVG_TOTAL_COST=("LASTMILE_TOTAL_COST", "mean"),
            AVG_ROUTES=("ROUTE_COUNT_VAL_ROUTE_LVL", "mean"),
            AVG_STOPS=("STOP_COUNT_VAL_ROUTE_LVL", "mean"),
            AVG_TOTES=("TOTE_COUNT_VAL_ROUTE_LVL", "mean"),
            AVG_CPR=("AVG_COST_PER_ROUTE", "mean"),
            AVG_CPS=("AVG_COST_PER_STOP", "mean"),
            AVG_CPT=("AVG_COST_PER_TOTE", "mean"),
        )
    )

    feature_cols = [c for c in carrier_profile.columns if c != "CARRIER_SCAC_NAME"]
    X = carrier_profile[feature_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    sim_matrix = cosine_similarity(X_scaled)
    carrier_names = carrier_profile["CARRIER_SCAC_NAME"].tolist()

    fig = px.imshow(
        sim_matrix,
        x=carrier_names, y=carrier_names,
        color_continuous_scale="Blues", zmin=0, zmax=1,
        title="Carrier Similarity Matrix",
        height=700, width=850,
    )
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Top similar pairs table
    pairs = []
    for i, j in itertools.combinations(range(len(carrier_names)), 2):
        pairs.append((carrier_names[i], carrier_names[j], round(sim_matrix[i, j], 4)))
    pairs_df = (
        pd.DataFrame(pairs, columns=["Carrier A", "Carrier B", "Similarity"])
        .sort_values("Similarity", ascending=False)
    )
    st.subheader("Top Similar Carrier Pairs")
    st.dataframe(pairs_df.head(15), use_container_width=True, hide_index=True)

# ========================== TAB 4: GEO MAP ==========================
with tab_geo:
    try:
        import pgeocode
        ship_df = load_shipment(SHIPMENT_PATH)
        view = st.radio("View", ["By Zip Code", "By State"], horizontal=True)

        if view == "By State":
            daily = ship_df.groupby(["State", "Del Date"], as_index=False)["Total Rate"].sum()
            state_rate = (
                daily.groupby("State", as_index=False)["Total Rate"]
                .mean()
                .rename(columns={"Total Rate": "Avg Daily Total Rate"})
                .dropna(subset=["State"])
                .query("State.str.strip() != ''", engine="python")
            )
            fig = px.choropleth(
                state_rate,
                locations="State", locationmode="USA-states",
                color="Avg Daily Total Rate", scope="usa",
                color_continuous_scale="YlOrRd",
                title="Average Daily Total Rate by State",
            )
        else:
            z = ship_df.assign(
                Zip5=ship_df["Zip"].astype(str).str.extract(r"(\d{5})", expand=False)
            ).dropna(subset=["Zip5"])
            daily = z.groupby(["Zip5", "Del Date"], as_index=False)["Total Rate"].sum()
            zip_rate = (
                daily.groupby("Zip5", as_index=False)["Total Rate"]
                .mean()
                .rename(columns={"Total Rate": "Avg Daily Total Rate"})
            )
            nomi = pgeocode.Nominatim("us")
            coords = nomi.query_postal_code(zip_rate["Zip5"].tolist())[
                ["postal_code", "latitude", "longitude", "place_name", "state_code"]
            ]
            zip_geo = zip_rate.merge(
                coords, left_on="Zip5", right_on="postal_code", how="left"
            ).dropna(subset=["latitude", "longitude"])
            zip_geo = zip_geo[zip_geo["Avg Daily Total Rate"] > 0]
            color_max = float(zip_geo["Avg Daily Total Rate"].quantile(0.99))
            st.caption(
                f"Showing {len(zip_geo):,} zips with non-zero rate "
                f"(color capped at ${color_max:,.0f}/day)."
            )
            fig = px.scatter_geo(
                zip_geo,
                lat="latitude", lon="longitude",
                color="Avg Daily Total Rate",
                size="Avg Daily Total Rate", size_max=22,
                range_color=[0, color_max], scope="usa",
                color_continuous_scale="YlOrRd",
                hover_name="Zip5",
                hover_data={
                    "place_name": True, "state_code": True,
                    "Avg Daily Total Rate": ":$,.0f",
                    "latitude": False, "longitude": False,
                },
                title="Average Daily Total Rate by Zip Code",
            )

        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.info(f"Shipment data not found at `{SHIPMENT_PATH}`.")
    except Exception as _geo_err:
        st.error(f"Geo Map error: {_geo_err}")
