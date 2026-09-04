"""
Last Mile Cost Outlier Detection
============================

Interactive Streamlit tool for last-mile / xdock market cost review.
It compares actual CPS to model-expected CPS, normalized cost, and cleansheet benchmarks.

Run locally:
    streamlit run lm_cost_app.py

Databricks Apps / Databricks terminal:
    export NORM_DATA_TABLE="catalog.schema.table_name"
    # or
    export NORM_DATA_PATH="/Volumes/catalog/schema/vol/LM_CS_slim.csv"
    streamlit run lm_cost_app.py
"""

from __future__ import annotations

import os
import re
import gc
import html
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Last Mile Cost Intelligence (LMCI)", layout="wide")

# --------------------------------------------------------------------------- #
# Paths / env
# --------------------------------------------------------------------------- #
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOCAL_CSV_PATH = os.path.join(_DATA_DIR, "LM_CS_slim.csv.gz")
LOCAL_PREPARED_PICKLE_PATH = os.path.join(_DATA_DIR, "LM_CS_slim_prepared.pkl")
LOCAL_PREPARED_PARQUET_PATH = os.path.join(_DATA_DIR, "LM_CS_slim_prepared.parquet")
LOCAL_SCORECARD_ARTIFACT_PATH = os.path.join(_DATA_DIR, "LM_CS_scorecard_default.pkl")
DATABRICKS_TABLE = os.environ.get("NORM_DATA_TABLE", "")
DATABRICKS_PATH = os.environ.get("NORM_DATA_PATH", "")

# --------------------------------------------------------------------------- #
# Column candidates
# --------------------------------------------------------------------------- #
XDOCK_CANDIDATES = ["XDOCK", "XDOCK_x", "XDOCK_y", "DC_CD"]
YEAR_CANDIDATES = ["YEAR", "ROUTE_YEAR", "year"]
MONTH_CANDIDATES = ["MONTH", "ROUTE_MONTH", "month"]
TARGET = "cost per stop"

# --------------------------------------------------------------------------- #
# Brand palette
# --------------------------------------------------------------------------- #
MCK_BLUE = "#00447C"
MCK_BRIGHT = "#0091DA"
MCK_GREEN = "#78BE20"
MCK_RED = "#C8102E"
MCK_GRAY = "#53565A"
MCK_OLIVE = "#4A7C1F"
MCK_ORANGE = "#FFB81C"
MCK_LIGHT = "#F6F9FC"
MCK_NAVY = "#002855"
MCK_LOGO = "https://www.mckesson.com/siteassets/images/z-mckesson-logosicons/mck_logo_blue.svg"

CLASS_COLORS = {
    "Overpay candidate": MCK_RED,
    "Normal / inside expected band": MCK_BLUE,
    "Possible underpay": MCK_GREEN,
    "Strong underpay candidate": MCK_OLIVE,
    "Not enough evidence": MCK_GRAY,
}

RECOMMENDER_MODEL_PACK: dict | None = None
RECOMMENDATION_ACTIONS = [
    "Renegotiate rate card / sourcing",
    "Route consolidation and stop-density lift",
    "Optimize fill DC allocation",
    "Fuel program and surcharge validation",
    "Invoice and accessorial audit",
]

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def existing_cols(df: pd.DataFrame, wanted: Iterable[str]) -> list[str]:
    return [c for c in wanted if c in df.columns]


def safe_div(a, b):
    return np.where((pd.notna(b)) & (b != 0), a / b, np.nan)


def clean_col_name(c: str) -> str:
    return (
        str(c)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def filter_table_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = str(query or "").strip()
    if not q:
        return df
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        mask = mask | df[col].astype(str).str.contains(q, case=False, na=False, regex=False)
    return df[mask]


def money(x, digits=2):
    if pd.isna(x):
        return "n/a"
    return f"${x:,.{digits}f}"


def pct(x, digits=1):
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:+,.{digits}f}%"


def plain_pct(x, digits=1):
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:,.{digits}f}%"


def market_name(xdock: str) -> str:
    s = re.sub(r"^XD_\d+_", "", str(xdock))
    s = re.sub(r"\.[A-Za-z0-9]+$", "", s)
    return s.replace("_", " ").title()


def make_label(df: pd.DataFrame) -> pd.Series:
    if "Market / Xdock" in df.columns:
        label = df["Market / Xdock"].astype(str)
    elif "XDOCK" in df.columns:
        label = df["XDOCK"].astype(str)
    else:
        label = df.index.astype(str)
    return label


def numeric_coerce(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in existing_cols(df, cols):
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def selected_rows_from_event(event) -> list[int]:
    if event is None:
        return []
    if isinstance(event, dict):
        return list(event.get("selection", {}).get("rows", []) or [])
    selection = getattr(event, "selection", None)
    if selection is None:
        return []
    rows = getattr(selection, "rows", None)
    return list(rows) if rows is not None else []


def build_xdock_geo_view(raw_df: pd.DataFrame, business_view: pd.DataFrame) -> pd.DataFrame:
    lat_col = _first_present(
        raw_df,
        [
            "ORIGIN_LATITUDE",
            "XDOCK_LATITUDE",
            "DESTINATION_LATITUDE",
            "latitude",
            "Latitude",
            "LATITUDE",
        ],
    )
    lon_col = _first_present(
        raw_df,
        [
            "ORIGIN_LONGITUDE",
            "XDOCK_LONGITUDE",
            "DESTINATION_LONGITUDE",
            "longitude",
            "Longitude",
            "LONGITUDE",
        ],
    )
    xdock_col = _first_present(raw_df, ["XDOCK", "DC_CD"])

    if lat_col is None or lon_col is None or xdock_col is None:
        return pd.DataFrame()

    geo = raw_df[[xdock_col, lat_col, lon_col]].copy()
    geo = geo.rename(columns={xdock_col: "XDOCK", lat_col: "latitude", lon_col: "longitude"})
    geo["XDOCK"] = geo["XDOCK"].astype(str).str.upper().str.strip()
    geo["latitude"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["longitude"] = pd.to_numeric(geo["longitude"], errors="coerce")
    geo = geo.dropna(subset=["latitude", "longitude"])
    geo = geo[(geo["latitude"] != 0) & (geo["longitude"] != 0)]
    if geo.empty:
        return pd.DataFrame()

    geo = geo.groupby("XDOCK", as_index=False).agg(latitude=("latitude", "median"), longitude=("longitude", "median"))

    view = business_view.copy()
    if "Market / Xdock" not in view.columns:
        return pd.DataFrame()
    view["Market / Xdock"] = view["Market / Xdock"].astype(str).str.upper().str.strip()
    return view.merge(geo, left_on="Market / Xdock", right_on="XDOCK", how="left")


def render_executive_view(geo_view: pd.DataFrame, baselines: dict[str, float]) -> None:
    st.subheader("Executive view")

    exec_view = geo_view.copy()
    if exec_view.empty:
        st.info("No scored market view is available for the executive summary.")
        return

    exec_view["Estimated Opportunity"] = np.where(
        exec_view["CPS Gap vs Expected"].fillna(0).gt(0),
        exec_view["CPS Gap vs Expected"].fillna(0) * exec_view["Records With CPS"].fillna(0),
        0.0,
    )

    overpay_view = exec_view[exec_view["Signal / Classification"].eq("Overpay candidate")].copy()
    strict_view = exec_view[
        exec_view["Signal / Classification"].eq("Overpay candidate")
        & exec_view["Actual CPS"].notna()
        & exec_view["Expected CPS"].notna()
        & exec_view["Normalized Cost"].notna()
        & exec_view["Cleansheet Aggressive"].notna()
        & (exec_view["Actual CPS"] > exec_view["Expected CPS"])
        & (exec_view["Actual CPS"] > exec_view["Normalized Cost"])
        & (exec_view["Actual CPS"] > exec_view["Cleansheet Aggressive"])
    ].copy()

    total_opportunity = float(overpay_view["Estimated Opportunity"].fillna(0).sum())
    median_gap = overpay_view["CPS Gap vs Expected %"].median() if len(overpay_view) else np.nan
    high_conf_count = int(overpay_view["Confidence"].eq("High").sum()) if "Confidence" in overpay_view.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overpay markets", f"{len(overpay_view):,}")
    c2.metric("Strict overpay", f"{len(strict_view):,}")
    c3.metric("Estimated opportunity", money(total_opportunity, digits=0))
    c4.metric("Median overpay gap", plain_pct(median_gap) if pd.notna(median_gap) else "n/a")
    st.caption(f"High-confidence overpay markets: {high_conf_count:,}")

    if len(overpay_view):
        overpay_view["Top Root Cause"] = overpay_view.apply(lambda r: top_root_cause_label(r, baselines), axis=1)
        overpay_view["Recommendation 1"] = overpay_view.apply(lambda r: top_recommendation(r, baselines), axis=1)
        overpay_view["Action 1 Impact"] = overpay_view.apply(lambda r: top_action_impact(r, baselines), axis=1)
        overpay_view["Action 1 Confidence"] = overpay_view.apply(lambda r: top_action_confidence(r, baselines), axis=1)
        overpay_view["Recommendation 2"] = overpay_view.apply(lambda r: second_recommendation(r, baselines), axis=1)
        overpay_view["Action 2 Impact"] = overpay_view.apply(lambda r: second_action_impact(r, baselines), axis=1)
        overpay_view["Action 2 Confidence"] = overpay_view.apply(lambda r: second_action_confidence(r, baselines), axis=1)

    # Initialize selection variable at module scope
    selected_from_exec_queue = None
    
    map_col, queue_col = st.columns([1.45, 1.0])
    
    # Placeholder for map - will be updated after table selection
    map_placeholder = map_col.empty()
    
    with queue_col:
        st.markdown("### Priority queue")
        if len(overpay_view):
            opp_max = float(overpay_view["Estimated Opportunity"].fillna(0).max()) if len(overpay_view) else 0.0
            gap_max = float(overpay_view["CPS Gap vs Expected %"].fillna(0).max()) if len(overpay_view) else 0.0
            opp_max = max(opp_max, 1.0)
            gap_max = max(gap_max, 0.01)

            queue_cols = existing_cols(
                overpay_view,
                [
                    "Market / Xdock",
                    "Estimated Opportunity",
                    "CPS Gap vs Expected %",
                    "Confidence",
                    "Actual CPS",
                    "Expected CPS",
                    "Expected CPS CS Model",
                    "Top Root Cause",
                    "Recommendation 1",
                    "Action 1 Impact",
                    "Action 1 Confidence",
                    "Recommendation 2",
                    "Action 2 Impact",
                    "Action 2 Confidence",
                ],
            )
            queue_df = overpay_view.sort_values(["Estimated Opportunity", "CPS Gap vs Expected %"], ascending=[False, False])[queue_cols]
            st.caption(f"Showing {len(queue_df)} overpay markets. Click a row to open details and actions.")
            
            try:
                exec_event = st.dataframe(
                    queue_df,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="exec_queue_table",
                    column_config={
                        "Estimated Opportunity": st.column_config.ProgressColumn(min_value=0.0, max_value=opp_max, format="$%.0f"),
                        "Actual CPS": st.column_config.NumberColumn(format="$%.2f"),
                        "Expected CPS": st.column_config.NumberColumn(format="$%.2f"),
                        "Expected CPS CS Model": st.column_config.NumberColumn(format="$%.2f"),
                        "CPS Gap vs Expected %": st.column_config.ProgressColumn(min_value=0.0, max_value=gap_max, format="%.1f%%"),
                    },
                )
                selected_rows = selected_rows_from_event(exec_event)
                if selected_rows:
                    row_idx = selected_rows[0]
                    if 0 <= row_idx < len(queue_df):
                        selected_from_exec_queue = str(queue_df.iloc[row_idx]["Market / Xdock"])
                        st.session_state["exec_selected_market"] = selected_from_exec_queue
            except TypeError:
                st.dataframe(
                    queue_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Estimated Opportunity": st.column_config.ProgressColumn(min_value=0.0, max_value=opp_max, format="$%.0f"),
                        "Actual CPS": st.column_config.NumberColumn(format="$%.2f"),
                        "Expected CPS": st.column_config.NumberColumn(format="$%.2f"),
                        "Expected CPS CS Model": st.column_config.NumberColumn(format="$%.2f"),
                        "CPS Gap vs Expected %": st.column_config.ProgressColumn(min_value=0.0, max_value=gap_max, format="%.1f%%"),
                    },
                )

            if selected_from_exec_queue is None:
                persisted_exec_market = st.session_state.get("exec_selected_market")
                if (
                    persisted_exec_market is not None
                    and "Market / Xdock" in queue_df.columns
                    and persisted_exec_market in set(queue_df["Market / Xdock"].astype(str))
                ):
                    selected_from_exec_queue = persisted_exec_market

            # Auto-open popup when a new row is selected in the table
            if selected_from_exec_queue:
                last_opened_exec_market = st.session_state.get("exec_last_opened_market")
                if selected_from_exec_queue != last_opened_exec_market:
                    popup_row_df = overpay_view[
                        overpay_view["Market / Xdock"].astype(str).eq(selected_from_exec_queue)
                    ]
                    if not popup_row_df.empty:
                        st.session_state["exec_last_opened_market"] = selected_from_exec_queue
                        open_recommendation_popup(
                            "Executive priority queue drilldown",
                            popup_row_df.iloc[0],
                            baselines,
                        )
        else:
            st.info("No overpay candidates with the current filters.")

    # Render map dynamically based on table selection
    with map_placeholder.container():
        st.markdown("### Opportunity map")
        map_df = overpay_view.dropna(subset=["latitude", "longitude"]).copy()
        if len(map_df):
            # Determine if a row is selected and filter map accordingly
            selected_xdock_for_map = None
            if selected_from_exec_queue:
                selected_xdock_for_map = selected_from_exec_queue
            
            # Prepare map data with highlighting
            map_data_all = map_df.copy()
            if selected_xdock_for_map:
                # Filter to show all but highlight the selected one
                selected_row = map_df[map_df["Market / Xdock"].astype(str).eq(selected_xdock_for_map)]
                if not selected_row.empty:
                    map_data_all["Is Selected"] = map_data_all["Market / Xdock"].astype(str).eq(selected_xdock_for_map)
                    
                    # Use color and size to distinguish selected
                    color_max = float(map_data_all["Estimated Opportunity"].fillna(0).max()) if map_data_all["Estimated Opportunity"].notna().any() else 1.0
                    
                    # Create figure with all points
                    fig = px.scatter_map(
                        map_data_all,
                        lat="latitude",
                        lon="longitude",
                        color="Estimated Opportunity",
                        size="Estimated Opportunity",
                        size_max=26,
                        range_color=[0, color_max],
                        color_continuous_scale=["#BFD7EA", "#4F81BD", "#C8102E"],
                        hover_name="Market / Xdock",
                        hover_data={
                            "Confidence": True,
                            "Actual CPS": ":$.2f",
                            "Expected CPS": ":$.2f",
                            "Estimated Opportunity": ":$,.0f",
                            "latitude": False,
                            "longitude": False,
                            "Is Selected": False,
                        } | ({"Market Name": True} if "Market Name" in map_data_all.columns else {}) | ({"Expected CPS CS Model": ":$.2f"} if "Expected CPS CS Model" in map_data_all.columns else {}),
                        title="Overpay opportunity by xdock (Click table row to zoom)",
                    )
                    
                    # Zoom to selected xdock
                    zoom_lat = float(selected_row.iloc[0]["latitude"])
                    zoom_lon = float(selected_row.iloc[0]["longitude"])
                    
                    # Add emphasis to selected marker
                    fig.update_traces(
                        marker=dict(opacity=0.82, sizemode="area", line=dict(width=0)),
                    )
                    
                    # Highlight selected point with a ring
                    fig.add_scattermapbox(
                        lat=[zoom_lat],
                        lon=[zoom_lon],
                        mode="markers",
                        marker=dict(size=20, color="rgba(200, 16, 46, 0)", line=dict(width=3, color=MCK_RED)),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                    
                    fig.update_layout(
                        height=520,
                        margin=dict(l=0, r=0, t=48, b=0),
                        map=dict(
                            style="open-street-map",
                            zoom=6.5,
                            center=dict(lat=zoom_lat, lon=zoom_lon),
                        ),
                    )
                else:
                    # Fallback if selected xdock not in map data
                    color_max = float(map_data_all["Estimated Opportunity"].fillna(0).max()) if map_data_all["Estimated Opportunity"].notna().any() else 1.0
                    center_lat = float(map_data_all["latitude"].median())
                    center_lon = float(map_data_all["longitude"].median())
                    
                    fig = px.scatter_map(
                        map_data_all,
                        lat="latitude",
                        lon="longitude",
                        color="Estimated Opportunity",
                        size="Estimated Opportunity",
                        size_max=26,
                        range_color=[0, color_max],
                        color_continuous_scale=["#BFD7EA", "#4F81BD", "#C8102E"],
                        hover_name="Market / Xdock",
                        hover_data={
                            "Confidence": True,
                            "Actual CPS": ":$.2f",
                            "Expected CPS": ":$.2f",
                            "Estimated Opportunity": ":$,.0f",
                            "latitude": False,
                            "longitude": False,
                        } | ({"Market Name": True} if "Market Name" in map_data_all.columns else {}) | ({"Expected CPS CS Model": ":$.2f"} if "Expected CPS CS Model" in map_data_all.columns else {}),
                        title="Overpay opportunity by xdock",
                    )
                    fig.update_traces(marker=dict(opacity=0.82, sizemode="area"))
                    fig.update_layout(
                        height=520,
                        margin=dict(l=0, r=0, t=48, b=0),
                        map=dict(
                            style="open-street-map",
                            zoom=3.2,
                            center=dict(lat=center_lat, lon=center_lon),
                        ),
                    )
            else:
                # No selection - show full map
                color_max = float(map_data_all["Estimated Opportunity"].fillna(0).max()) if map_data_all["Estimated Opportunity"].notna().any() else 1.0
                center_lat = float(map_data_all["latitude"].median())
                center_lon = float(map_data_all["longitude"].median())
                
                fig = px.scatter_map(
                    map_data_all,
                    lat="latitude",
                    lon="longitude",
                    color="Estimated Opportunity",
                    size="Estimated Opportunity",
                    size_max=26,
                    range_color=[0, color_max],
                    color_continuous_scale=["#BFD7EA", "#4F81BD", "#C8102E"],
                    hover_name="Market / Xdock",
                    hover_data={
                        "Confidence": True,
                        "Actual CPS": ":$.2f",
                        "Expected CPS": ":$.2f",
                        "Estimated Opportunity": ":$,.0f",
                        "latitude": False,
                        "longitude": False,
                    } | ({"Market Name": True} if "Market Name" in map_data_all.columns else {}) | ({"Expected CPS CS Model": ":$.2f"} if "Expected CPS CS Model" in map_data_all.columns else {}),
                    title="Overpay opportunity by xdock",
                )
                fig.update_traces(marker=dict(opacity=0.82, sizemode="area"))
                fig.update_layout(
                    height=520,
                    margin=dict(l=0, r=0, t=48, b=0),
                    map=dict(
                        style="open-street-map",
                        zoom=3.2,
                        center=dict(lat=center_lat, lon=center_lon),
                    ),
                )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Map is ready, but the app data does not yet include xdock latitude/longitude columns.")
            st.caption("Your notebook geocodes query can be used once ORIGIN_LATITUDE and ORIGIN_LONGITUDE are merged into the exported app dataset.")

    st.markdown("### Why these markets")
    left_col, right_col = st.columns([1.0, 1.0])
    with left_col:
        if len(overpay_view):
            root_cause_summary = (
                overpay_view["Top Root Cause"]
                .fillna("Unassigned")
                .value_counts()
                .rename_axis("Top Root Cause")
                .reset_index(name="Markets")
            )
            fig_root = px.bar(
                root_cause_summary.head(8),
                x="Markets",
                y="Top Root Cause",
                orientation="h",
                title="Top root causes across overpay markets",
                color_discrete_sequence=[MCK_BRIGHT],
            )
            fig_root.update_layout(height=360, showlegend=False, yaxis_title="", xaxis_title="Markets")
            st.plotly_chart(fig_root, use_container_width=True)
        else:
            st.info("Root-cause summary will appear when overpay markets are available.")
    with right_col:
        comparison_rows = baselines.get("model_comparison")
        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            display_cols = existing_cols(comparison_df, ["Approach", "Model Status", "Feature Count", "MAE Cost", "Median Abs Error Cost", "R2 Log Target"])
            st.dataframe(
                comparison_df[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MAE Cost": st.column_config.NumberColumn(format="$%.2f"),
                    "Median Abs Error Cost": st.column_config.NumberColumn(format="$%.2f"),
                    "R2 Log Target": st.column_config.NumberColumn(format="%.3f"),
                },
            )
        else:
            fallback_metrics = baselines.get("model_metrics")
            if isinstance(fallback_metrics, dict):
                fallback_df = pd.DataFrame(
                    [
                        {
                            "Approach": "Baseline model",
                            "Model Status": fallback_metrics.get("model_status", "model"),
                            "Feature Count": fallback_metrics.get("feature_count"),
                            "MAE Cost": fallback_metrics.get("mae_cost"),
                            "Median Abs Error Cost": fallback_metrics.get("median_abs_error_cost"),
                            "R2 Log Target": fallback_metrics.get("r2_log_target"),
                        }
                    ]
                )
                display_cols = existing_cols(fallback_df, ["Approach", "Model Status", "Feature Count", "MAE Cost", "Median Abs Error Cost", "R2 Log Target"])
                st.dataframe(
                    fallback_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "MAE Cost": st.column_config.NumberColumn(format="$%.2f"),
                        "Median Abs Error Cost": st.column_config.NumberColumn(format="$%.2f"),
                        "R2 Log Target": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
                st.caption("CS model comparison is not present in this artifact. Showing baseline model metrics.")
            else:
                st.info("Model comparison metrics are not available for this run.")

# --------------------------------------------------------------------------- #
# Loading and preparation
# --------------------------------------------------------------------------- #
def _load_via_spark() -> pd.DataFrame | None:
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
    except Exception:
        return None

    if DATABRICKS_TABLE:
        return spark.table(DATABRICKS_TABLE).toPandas()
    if DATABRICKS_PATH:
        if DATABRICKS_PATH.lower().endswith(".parquet"):
            return spark.read.parquet(DATABRICKS_PATH).toPandas()
        return spark.read.option("header", True).option("inferSchema", True).csv(DATABRICKS_PATH).toPandas()
    return None


@st.cache_data(show_spinner="Loading cost data...")
def load_data() -> pd.DataFrame:
    prepared_path = os.environ.get("NORM_PREPARED_PATH", "").strip()
    if prepared_path:
        if prepared_path.lower().endswith(".pkl") and os.path.exists(prepared_path):
            return pd.read_pickle(prepared_path)
        if prepared_path.lower().endswith(".parquet") and os.path.exists(prepared_path):
            return pd.read_parquet(prepared_path)

    for local_prepared in [LOCAL_PREPARED_PICKLE_PATH, LOCAL_PREPARED_PARQUET_PATH]:
        if os.path.exists(local_prepared):
            if local_prepared.lower().endswith(".pkl"):
                return pd.read_pickle(local_prepared)
            return pd.read_parquet(local_prepared)

    df: pd.DataFrame | None = None
    if _on_databricks():
        df = _load_via_spark()
    if df is None:
        azure_conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        azure_container = os.environ.get("AZURE_STORAGE_CONTAINER", "data")
        azure_blob = os.environ.get("AZURE_STORAGE_BLOB_NAME", "LM_CS_slim.csv.gz")
        if azure_conn:
            from azure.storage.blob import BlobClient
            import io
            blob = BlobClient.from_connection_string(azure_conn, azure_container, azure_blob)
            data = blob.download_blob().readall()
            df = pd.read_csv(io.BytesIO(data), compression="gzip", low_memory=False)
        else:
            path = os.environ.get("NORM_DATA_PATH") or LOCAL_CSV_PATH
            df = pd.read_csv(path, low_memory=False)
    return prepare(df)


@st.cache_data(show_spinner="Preparing data...")
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Standardize stop-location identifier so downstream datasets are consistent.
    if "CNSLDTN_ID" not in df.columns and "TRACKING_ID" in df.columns:
        df["CNSLDTN_ID"] = df["TRACKING_ID"]

    xdock_col = _first_present(df, XDOCK_CANDIDATES)
    year_col = _first_present(df, YEAR_CANDIDATES)
    month_col = _first_present(df, MONTH_CANDIDATES)

    if xdock_col is None:
        raise KeyError("Could not find xdock/market column. Expected one of XDOCK, XDOCK_x, XDOCK_y, DC_CD.")
    df = df.rename(columns={xdock_col: "XDOCK"})

    if year_col is not None:
        df = df.rename(columns={year_col: "YEAR"})
    if month_col is not None:
        df = df.rename(columns={month_col: "MONTH"})

    numeric_cols = [
        TARGET,
        "YEAR",
        "MONTH",
        "normalized_cost",
        "normalized_cost_geography",
        "normalized_cost_shipment",
        "normalized_cost_density",
        "GEOGRAPHY_MULTIPLIER",
        "shipment_norm_multiplier_qt",
        "miles_per_stop_norm_multiplier_qt",
        "geo_mean",
        "STOP_COUNT_VAL_ROUTE_LVL",
        "TOTE_COUNT_VAL_ROUTE_LVL",
        "STOP_COUNT_VAL",
        "TOTE_COUNT_VAL",
        "ROUTE_COUNT_VAL",
        "DISTANCE_VAL",
        "LASTMILE_TOTAL_COST",
        "LASTMILE_BASE_COST",
        "LASTMILE_FUEL_COST",
        "LASTMILE_MISC_COST",
        "Cleansheet Cost Per Stop Conservative",
        "Cleansheet Cost Per Stop Aggressive",
    ]
    df = numeric_coerce(df, numeric_cols)

    for c in existing_cols(df, ["XDOCK", "FILL_DC_CD", "CARRIER_SCAC_CD", "ASN_TYPE_CD", "DELIVERY_TYPE", "CUST_BUS_TYP_DSCR"]):
        df[c] = df[c].astype("string").fillna("Unknown")

    if "FILL_DC_CD" in df.columns:
        df["FILL_DC_CD"] = df["FILL_DC_CD"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    else:
        df["FILL_DC_CD"] = df["XDOCK"].astype(str).str.extract(r"XD_(\d+)_", expand=False).fillna("Unknown")

    if "YEAR" in df.columns:
        df = df[df["YEAR"].notna()].copy()
        df["YEAR"] = df["YEAR"].astype(int)
    if "MONTH" in df.columns:
        df = df[df["MONTH"].notna()].copy()
        df["MONTH"] = df["MONTH"].astype(int)

    df = df[df["XDOCK"].notna()].copy()
    df["MARKET"] = df["XDOCK"].map(market_name)
    df["paid_flag"] = df[TARGET].fillna(0) > 0
    return df


def _is_default_scorecard_request(
    *,
    use_period_filter: bool,
    dc_options: list[str],
    sel_dcs: list[str],
    cust_type_options: list[str],
    sel_cust_types: list[str],
    min_group_n: int,
    overpay_strong: float,
    overpay_possible: float,
    underpay_strong: float,
    underpay_possible: float,
    use_p25_p75: bool,
) -> bool:
    full_scope = (not use_period_filter) and (set(sel_dcs) == set(dc_options)) and (
        set(sel_cust_types) == set(cust_type_options)
    )
    default_thresholds = (
        int(min_group_n) == 30
        and abs(float(overpay_possible) - 0.10) < 1e-9
        and abs(float(overpay_strong) - 0.20) < 1e-9
        and abs(float(underpay_possible) - (-0.10)) < 1e-9
        and abs(float(underpay_strong) - (-0.20)) < 1e-9
        and (not bool(use_p25_p75))
    )
    return full_scope and default_thresholds


@st.cache_data(show_spinner=False)
def load_precomputed_scorecard() -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    env_path = os.environ.get("NORM_SCORECARD_PATH", "").strip()
    candidates = [env_path, LOCAL_SCORECARD_ARTIFACT_PATH] if env_path else [LOCAL_SCORECARD_ARTIFACT_PATH]
    for path in candidates:
        if not path or (not os.path.exists(path)):
            continue
        try:
            payload = pd.read_pickle(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        scorecard = payload.get("scorecard")
        business_view = payload.get("business_view")
        model_metrics = payload.get("model_metrics")
        if isinstance(scorecard, pd.DataFrame) and isinstance(business_view, pd.DataFrame) and isinstance(model_metrics, dict):
            return scorecard, business_view, model_metrics
    return None

# --------------------------------------------------------------------------- #
# Modeling / scorecard engine
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Aggregating and fitting expected-cost model...")
def build_scorecard(
    df: pd.DataFrame,
    group_cols: list[str],
    min_group_n: int,
    overpay_strong: float,
    overpay_possible: float,
    underpay_strong: float,
    underpay_possible: float,
    use_p25_p75: bool,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Keep only required columns before aggregation to avoid duplicating the full raw file.
    metric_cols = existing_cols(
        df,
        [
            "normalized_cost",
            "normalized_cost_geography",
            "normalized_cost_shipment",
            "normalized_cost_density",
            "GEOGRAPHY_MULTIPLIER",
            "shipment_norm_multiplier_qt",
            "miles_per_stop_norm_multiplier_qt",
            "geo_mean",
            "STOP_COUNT_VAL_ROUTE_LVL",
            "TOTE_COUNT_VAL_ROUTE_LVL",
            "STOP_COUNT_VAL",
            "TOTE_COUNT_VAL",
            "ROUTE_COUNT_VAL",
            "DISTANCE_VAL",
            "LASTMILE_TOTAL_COST",
            "LASTMILE_BASE_COST",
            "LASTMILE_FUEL_COST",
            "LASTMILE_MISC_COST",
            "Cleansheet Cost Per Stop Conservative",
            "Cleansheet Cost Per Stop Aggressive",
        ],
    )
    keep = list(dict.fromkeys(group_cols + ["XDOCK", "MARKET", "FILL_DC_CD", "ROUTE_ID", TARGET, "paid_flag"] + metric_cols))
    work = df[existing_cols(df, keep)].copy()

    # Reduce memory pressure.
    for c in group_cols:
        if c in work.columns:
            work[c] = work[c].astype("category")
    for c in ["paid_flag"]:
        work[c] = work[c].astype("int8")
    for c in existing_cols(work, [TARGET] + metric_cols):
        work[c] = pd.to_numeric(work[c], errors="coerce").astype("float32")

    counts = (
        work.groupby(group_cols, dropna=False, observed=True)
        .agg(records=(TARGET, "size"), paid_records=("paid_flag", "sum"))
        .reset_index()
    )

    paid = work.loc[work["paid_flag"].eq(1)]
    if paid.empty:
        raise ValueError("No positive paid cost rows available after filters.")

    # Pre-aggregate to route level first (mean CPS per route), then to market level (median of route means).
    # This matches the notebook's Fabio logic: each route gets equal weight regardless of stop count.
    if "ROUTE_ID" in paid.columns:
        route_agg_cols = group_cols + ["ROUTE_ID"]
        route_agg_cols = existing_cols(paid, route_agg_cols)
        route_level = paid.groupby(route_agg_cols, dropna=False, observed=True).agg(
            **{TARGET: (TARGET, "mean"),
               **{c: (c, "mean") for c in existing_cols(paid, metric_cols)}}
        ).reset_index()
        paid_for_agg = route_level
    else:
        paid_for_agg = paid

    agg_dict = {
        "actual_median_cps": (TARGET, "median"),
        "actual_mean_cps": (TARGET, "mean"),
        "actual_p75_cps": (TARGET, lambda x: x.quantile(0.75)),
        "actual_p90_cps": (TARGET, lambda x: x.quantile(0.90)),
    }
    for c in metric_cols:
        agg_dict[f"median_{clean_col_name(c)}"] = (c, "median")

    paid_agg = paid_for_agg.groupby(group_cols, dropna=False, observed=True).agg(**agg_dict).reset_index()
    market = counts.merge(paid_agg, on=group_cols, how="left")

    market["paid_rate"] = safe_div(market["paid_records"], market["records"])
    for c in group_cols:
        market[c] = market[c].astype(str)

    cons_col = "median_cleansheet_cost_per_stop_conservative"
    aggr_col = "median_cleansheet_cost_per_stop_aggressive"
    if cons_col in market.columns and aggr_col in market.columns:
        has_both_cleansheet = market[cons_col].notna() & market[aggr_col].notna()
        market["cleansheet_mid_cost"] = np.where(
            has_both_cleansheet,
            (market[cons_col] + market[aggr_col]) / 2,
            np.nan,
        )
        market["cleansheet_range_spread_pct"] = np.abs(safe_div(market[aggr_col], market[cons_col]) - 1)
    else:
        market["cleansheet_mid_cost"] = np.nan
        market["cleansheet_range_spread_pct"] = np.nan

    # Model dataset.
    model_df = market[
        market["actual_median_cps"].notna()
        & (market["actual_median_cps"] > 0)
        & (market["paid_records"] >= min_group_n)
    ].copy()

    numeric_features = existing_cols(
        model_df,
        [
            "paid_records",
            "records",
            "paid_rate",
            "median_geography_multiplier",
            "median_shipment_norm_multiplier_qt",
            "median_miles_per_stop_norm_multiplier_qt",
            "median_geo_mean",
            "median_stop_count_val_route_lvl",
            "median_tote_count_val_route_lvl",
            "median_stop_count_val",
            "median_tote_count_val",
            "median_route_count_val",
            "median_distance_val",
            "median_lastmile_total_cost",
            "median_lastmile_base_cost",
            "median_lastmile_fuel_cost",
            "median_lastmile_misc_cost",
        ],
    )
    categorical_features = existing_cols(model_df, ["XDOCK", "FILL_DC_CD"])
    # Do not include the same column as both the grouping identifier and a feature when model rows are very sparse.
    categorical_features = [c for c in categorical_features if c in group_cols]
    feature_cols = numeric_features + categorical_features

    metrics = {
        "groups_total": int(len(market)),
        "model_groups": int(len(model_df)),
    }

    quantiles = [0.10, 0.50, 0.90] if not use_p25_p75 else [0.10, 0.25, 0.50, 0.75, 0.90]
    score_eligible = market["actual_median_cps"].notna() & (market["actual_median_cps"] > 0)

    if len(model_df):
        y = np.log1p(model_df["actual_median_cps"].astype(float))
        weights = np.sqrt(model_df["paid_records"].clip(lower=1).astype(float))
        train_idx, test_idx = train_test_split(model_df.index.to_numpy(), test_size=0.20, random_state=random_state)
    else:
        y = pd.Series(dtype=float)
        weights = pd.Series(dtype=float)
        train_idx = np.array([], dtype=int)
        test_idx = np.array([], dtype=int)

    def fit_expected_cost_variant(
        approach_name: str,
        score_prefix: str,
        numeric_variant_features: list[str],
        categorical_variant_features: list[str],
        uses_cleansheet_midpoint: bool,
    ) -> dict:
        variant_feature_cols = numeric_variant_features + categorical_variant_features
        variant_metrics = {
            "Approach": approach_name,
            "Model Status": "model",
            "Feature Count": int(len(variant_feature_cols)),
            "Model Groups": int(len(model_df)),
            "Uses Cleansheet Midpoint": uses_cleansheet_midpoint,
            "Quantiles": ", ".join(f"P{int(q * 100):02d}" for q in quantiles),
            "MAE Cost": np.nan,
            "Median Abs Error Cost": np.nan,
            "R2 Log Target": np.nan,
        }

        qcols = [f"{score_prefix}_p{int(q * 100):02d}" for q in quantiles]
        fallback_p50 = model_df["actual_median_cps"].median() if len(model_df) else market["actual_median_cps"].median()
        fallback_p10 = model_df["actual_median_cps"].quantile(0.10) if len(model_df) else market["actual_median_cps"].quantile(0.10)
        fallback_p90 = model_df["actual_median_cps"].quantile(0.90) if len(model_df) else market["actual_median_cps"].quantile(0.90)
        fallback_p25 = model_df["actual_median_cps"].quantile(0.25) if len(model_df) else market["actual_median_cps"].quantile(0.25)
        fallback_p75 = model_df["actual_median_cps"].quantile(0.75) if len(model_df) else market["actual_median_cps"].quantile(0.75)

        if len(model_df) < 20 or len(variant_feature_cols) == 0:
            variant_metrics["Model Status"] = "fallback benchmark"
            for q in quantiles:
                col = f"{score_prefix}_p{int(q * 100):02d}"
                if q == 0.10:
                    market[col] = fallback_p10
                elif q == 0.25:
                    market[col] = fallback_p25
                elif q == 0.50:
                    market[col] = fallback_p50
                elif q == 0.75:
                    market[col] = fallback_p75
                else:
                    market[col] = fallback_p90
            market[qcols] = np.sort(market[qcols].to_numpy(), axis=1)
            return variant_metrics

        def make_preprocessor() -> ColumnTransformer:
            num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            )
            return ColumnTransformer(
                [("num", num_pipe, numeric_variant_features), ("cat", cat_pipe, categorical_variant_features)],
                remainder="drop",
            )

        def make_model(q: float) -> Pipeline:
            return Pipeline(
                [
                    ("prep", make_preprocessor()),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            loss="quantile",
                            quantile=q,
                            max_iter=160,
                            learning_rate=0.06,
                            max_leaf_nodes=31,
                            l2_regularization=0.05,
                            random_state=random_state,
                        ),
                    ),
                ]
            )

        X_train = model_df.loc[train_idx, variant_feature_cols].copy()
        X_test = model_df.loc[test_idx, variant_feature_cols].copy()
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
        w_train = weights.loc[train_idx]

        models = {}
        for q in quantiles:
            model = make_model(q)
            model.fit(X_train, y_train, model__sample_weight=w_train)
            models[q] = model

        pred_test = np.expm1(models[0.50].predict(X_test)).clip(min=0)
        y_test_cost = np.expm1(y_test)
        variant_metrics.update(
            {
                "MAE Cost": float(mean_absolute_error(y_test_cost, pred_test)),
                "Median Abs Error Cost": float(median_absolute_error(y_test_cost, pred_test)),
                "R2 Log Target": float(r2_score(y_test, models[0.50].predict(X_test))),
            }
        )

        for q, model in models.items():
            col = f"{score_prefix}_p{int(q * 100):02d}"
            market[col] = np.nan
            market.loc[score_eligible, col] = np.expm1(
                model.predict(market.loc[score_eligible, variant_feature_cols])
            ).clip(min=0)
        market[qcols] = np.sort(market[qcols].to_numpy(), axis=1)
        return variant_metrics

    baseline_metrics = fit_expected_cost_variant(
        approach_name="Baseline model",
        score_prefix="pred",
        numeric_variant_features=numeric_features,
        categorical_variant_features=categorical_features,
        uses_cleansheet_midpoint=False,
    )

    challenger_numeric_features = list(numeric_features)
    challenger_available = "cleansheet_mid_cost" in model_df.columns and model_df["cleansheet_mid_cost"].notna().any()
    if challenger_available:
        challenger_numeric_features.append("cleansheet_mid_cost")

    challenger_metrics = fit_expected_cost_variant(
        approach_name="CS model",
        score_prefix="alt_pred",
        numeric_variant_features=challenger_numeric_features,
        categorical_variant_features=categorical_features,
        uses_cleansheet_midpoint=challenger_available,
    )
    if not challenger_available:
        challenger_metrics["Model Status"] = "not available"

    metrics.update(
        {
            "feature_count": baseline_metrics["Feature Count"],
            "model_status": baseline_metrics["Model Status"],
            "quantiles": baseline_metrics["Quantiles"],
            "mae_cost": baseline_metrics["MAE Cost"],
            "median_abs_error_cost": baseline_metrics["Median Abs Error Cost"],
            "r2_log_target": baseline_metrics["R2 Log Target"],
            "model_comparison": [baseline_metrics, challenger_metrics],
        }
    )

    market["model_residual"] = market["actual_median_cps"] - market["pred_p50"]
    market["model_residual_pct"] = safe_div(market["actual_median_cps"], market["pred_p50"]) - 1
    market["normalization_sensitivity_pct"] = safe_div(market.get("median_normalized_cost", np.nan), market["actual_median_cps"]) - 1
    market["norm_sensitivity_abs_pct"] = np.abs(market["normalization_sensitivity_pct"])

    market["cleansheet_cons_gap_pct"] = safe_div(market["actual_median_cps"], market[cons_col]) - 1 if cons_col in market.columns else np.nan
    market["cleansheet_aggr_gap_pct"] = safe_div(market["actual_median_cps"], market[aggr_col]) - 1 if aggr_col in market.columns else np.nan

    # Cleansheet sensitivity mirrors normalized-cost sensitivity: compare benchmark cost to actual CPS.
    market["cleansheet_sensitivity_pct"] = safe_div(market["cleansheet_mid_cost"], market["actual_median_cps"]) - 1
    market["cleansheet_sensitivity_abs_pct"] = np.abs(market["cleansheet_sensitivity_pct"])
    market["combined_sensitivity_abs_pct"] = market[["norm_sensitivity_abs_pct", "cleansheet_sensitivity_abs_pct"]].mean(axis=1, skipna=True)

    market["confidence_score"] = 0
    market.loc[market["paid_records"] >= min_group_n, "confidence_score"] += 1
    market.loc[market["paid_records"] >= min_group_n * 3, "confidence_score"] += 1
    market.loc[market["combined_sensitivity_abs_pct"].fillna(0) <= 0.40, "confidence_score"] += 1
    market["confidence"] = np.select(
        [market["confidence_score"] >= 3, market["confidence_score"] == 2, market["confidence_score"] == 1],
        ["High", "Medium", "Low"],
        default="Very Low",
    )

    market["classification"] = "Normal / inside expected band"
    strong_conf = market["confidence"].isin(["High", "Medium"])
    if use_p25_p75:
        overpay_possible_condition = (market["model_residual_pct"] >= overpay_possible) & (market["actual_median_cps"] > market["pred_p75"])
        underpay_possible_condition = (market["model_residual_pct"] <= underpay_possible) & (market["actual_median_cps"] < market["pred_p25"])
    else:
        overpay_possible_condition = market["model_residual_pct"] >= overpay_possible
        underpay_possible_condition = market["model_residual_pct"] <= underpay_possible

    market.loc[(market["model_residual_pct"] >= overpay_strong) & (market["actual_median_cps"] > market["pred_p90"]) & strong_conf, "classification"] = "Overpay candidate"
    market.loc[market["classification"].eq("Normal / inside expected band") & overpay_possible_condition, "classification"] = "Overpay candidate"
    market.loc[(market["model_residual_pct"] <= underpay_strong) & (market["actual_median_cps"] < market["pred_p10"]) & strong_conf, "classification"] = "Strong underpay candidate"
    market.loc[market["classification"].eq("Normal / inside expected band") & underpay_possible_condition, "classification"] = "Possible underpay"
    market.loc[(market["paid_records"] < min_group_n) | market["confidence"].eq("Very Low") | market["actual_median_cps"].isna() | market["pred_p50"].isna(), "classification"] = "Not enough evidence"

    market["cleansheet_overpay_support"] = market["cleansheet_aggr_gap_pct"].notna() & (market["cleansheet_aggr_gap_pct"] > 0)
    market["cleansheet_underpay_support"] = market["cleansheet_cons_gap_pct"].notna() & (market["cleansheet_cons_gap_pct"] < 0)

    # --- Classification reason ---
    def _cls_reason(row):
        parts = []
        gap_pct = row.get("model_residual_pct", 0) or 0
        actual = row.get("actual_median_cps", None)
        p90 = row.get("pred_p90", None)
        p10 = row.get("pred_p10", None)
        conf = row.get("confidence", "")
        paid = row.get("paid_records", 0) or 0
        cls = row.get("classification", "")

        if cls == "Not enough evidence":
            if paid < min_group_n:
                parts.append(f"low volume ({int(paid)} records)")
            if conf == "Very Low":
                parts.append("very low confidence")
            if pd.isna(actual):
                parts.append("no actual CPS")
        elif "overpay" in cls.lower():
            parts.append(f"gap +{gap_pct:.1%} vs expected")
            if actual is not None and p90 is not None:
                if actual > p90:
                    parts.append(f"actual ${actual:.2f} > P90 ${p90:.2f}")
                else:
                    parts.append(f"actual ${actual:.2f} within P90 ${p90:.2f}")
            if row.get("cleansheet_overpay_support"):
                parts.append(f"cleansheet also high ({row.get('cleansheet_aggr_gap_pct', 0):.1%} above aggressive)")
        elif "underpay" in cls.lower():
            parts.append(f"gap {gap_pct:.1%} vs expected")
            if actual is not None and p10 is not None:
                if actual < p10:
                    parts.append(f"actual ${actual:.2f} < P10 ${p10:.2f}")
                else:
                    parts.append(f"actual ${actual:.2f} within P10 ${p10:.2f}")
            if row.get("cleansheet_underpay_support"):
                parts.append(f"cleansheet also low ({row.get('cleansheet_cons_gap_pct', 0):.1%} vs conservative)")
        else:
            parts.append(f"gap {gap_pct:.1%} within thresholds")
            if actual is not None and p90 is not None:
                parts.append(f"actual ${actual:.2f} within band [${p10:.2f}–${p90:.2f}]")
        return "; ".join(parts)

    # --- Confidence reason ---
    def _conf_reason(row):
        parts = []
        paid = row.get("paid_records", 0) or 0
        sens = row.get("combined_sensitivity_abs_pct", None)
        if paid >= min_group_n * 3:
            parts.append(f"high volume ({int(paid)} records)")
        elif paid >= min_group_n:
            parts.append(f"adequate volume ({int(paid)} records)")
        else:
            parts.append(f"low volume ({int(paid)} records)")
        if sens is not None:
            if sens <= 0.40:
                parts.append(f"stable sensitivity ({sens:.0%})")
            else:
                parts.append(f"unstable sensitivity ({sens:.0%})")
        return "; ".join(parts)

    market["Classification Reason"] = market.apply(_cls_reason, axis=1)
    market["Confidence Reason"] = market.apply(_conf_reason, axis=1)

    market["business_note"] = np.select(
        [
            market["classification"].str.contains("overpay", case=False, na=False) & market["cleansheet_overpay_support"],
            market["classification"].str.contains("underpay", case=False, na=False) & market["cleansheet_underpay_support"],
            market["classification"].eq("Not enough evidence"),
            market["combined_sensitivity_abs_pct"].fillna(0) > 0.40,
        ],
        [
            "Model and cleansheet both point high",
            "Model and cleansheet both point low",
            "Low volume or weak basis",
            "Sensitive to normalization and cleansheet assumptions",
        ],
        default="Model-based residual signal",
    )

    # Business-ready aliases.
    market["Market / Xdock"] = market["XDOCK"] if "XDOCK" in market.columns else market[group_cols[0]]
    market["Market Name"] = market["Market / Xdock"].map(market_name)
    market["Actual CPS"] = market["actual_median_cps"]
    market["Expected CPS"] = market["pred_p50"]
    market["Expected CPS CS Model"] = market["alt_pred_p50"] if "alt_pred_p50" in market.columns else np.nan
    market["Expected Low CPS P10"] = market["pred_p10"]
    market["Expected High CPS P90"] = market["pred_p90"]
    market["CPS Gap vs Expected"] = market["model_residual"]
    market["CPS Gap vs Expected %"] = market["model_residual_pct"]
    market["Normalized Cost"] = market["median_normalized_cost"] if "median_normalized_cost" in market.columns else np.nan
    market["Cleansheet Conservative"] = market[cons_col] if cons_col in market.columns else np.nan
    market["Cleansheet Aggressive"] = market[aggr_col] if aggr_col in market.columns else np.nan
    market["Signal / Classification"] = market["classification"]
    market["Confidence"] = market["confidence"]
    market["Business Note"] = market["business_note"]
    market["Classification Reason"] = market["Classification Reason"]
    market["Confidence Reason"] = market["Confidence Reason"]
    market["Records With CPS"] = market["paid_records"]
    market["Total Records"] = market["records"]
    market["Gap vs Conservative Cleansheet %"] = market["cleansheet_cons_gap_pct"]
    market["Gap vs Aggressive Cleansheet %"] = market["cleansheet_aggr_gap_pct"]
    market["Normalization Sensitivity"] = market["norm_sensitivity_abs_pct"]
    market["Cleansheet Sensitivity"] = market["cleansheet_sensitivity_abs_pct"]
    market["Combined Sensitivity"] = market["combined_sensitivity_abs_pct"]

    display_cols = existing_cols(
        market,
        [
            "Market / Xdock",
            "Market Name",
            "Signal / Classification",
            "Confidence",
            "Classification Reason",
            "Confidence Reason",
            "Business Note",
            "Total Records",
            "Records With CPS",
            "Actual CPS",
            "Expected CPS",
            "Expected Low CPS P10",
            "Expected High CPS P90",
            "CPS Gap vs Expected",
            "CPS Gap vs Expected %",
            "Normalized Cost",
            "Normalization Sensitivity",
            "Cleansheet Sensitivity",
            "Combined Sensitivity",
            "Cleansheet Conservative",
            "Cleansheet Aggressive",
            "Gap vs Conservative Cleansheet %",
            "Gap vs Aggressive Cleansheet %",
            "median_distance_val",
            "median_stop_count_val_route_lvl",
            "median_tote_count_val_route_lvl",
        ],
    )
    business_view = market[display_cols].copy()
    business_view = business_view.sort_values(["Signal / Classification", "CPS Gap vs Expected %", "Records With CPS"], ascending=[True, False, False])
    gc.collect()
    return market, business_view, metrics

# --------------------------------------------------------------------------- #
# Explanation engine
# --------------------------------------------------------------------------- #
def classification_summary_text(business_view: pd.DataFrame, metrics: dict) -> list[str]:
    out = []
    total = len(business_view)
    counts = business_view["Signal / Classification"].value_counts()
    overpay = int(counts.get("Overpay candidate", 0))
    strong_under = int(counts.get("Strong underpay candidate", 0))
    poss_under = int(counts.get("Possible underpay", 0))
    not_enough = int(counts.get("Not enough evidence", 0))

    out.append(
        f"The tool scored {total:,} market groups. It identified {overpay:,} overpay candidates."
    )
    if strong_under + poss_under > 0:
        out.append(f"It also found {strong_under + poss_under:,} underpay signals. Treat underpay as a validation queue, not an immediate savings opportunity, because low cost can also indicate missing charges or incomplete invoices.")
    if not_enough > 0:
        out.append(f"{not_enough:,} groups are marked as not enough evidence because of low volume, weak confidence, or missing expected CPS.")
    if metrics.get("model_status") == "fallback benchmark":
        out.append("The model fell back to a benchmark method because there were not enough market groups or features to train a stable expected-cost model.")
    else:
        out.append("Expected CPS is the model-estimated median cost per stop for a comparable market profile. The overpay signal is based on actual CPS versus expected CPS, not normalized cost alone.")
    return out


def selected_class_text(detail: pd.DataFrame, selected_class: str) -> list[str]:
    if detail.empty:
        return [f"No market groups currently fall into {selected_class}."]
    top = detail.iloc[0]
    out = [
        f"For {selected_class}, the largest gap is {top.get('Market Name', top.get('Market / Xdock', 'n/a'))} at {pct(top.get('CPS Gap vs Expected %'))} versus expected CPS.",
        f"Top actual CPS is {money(top.get('Actual CPS'))} versus expected CPS of {money(top.get('Expected CPS'))}; records with CPS count is {int(top.get('Records With CPS', 0)):,}.",
    ]
    if pd.notna(top.get("Cleansheet Aggressive", np.nan)):
        out.append(f"Against aggressive cleansheet, the top group is {pct(top.get('Gap vs Aggressive Cleansheet %'))} from benchmark.")
    if pd.notna(top.get("Normalization Sensitivity", np.nan)) and top.get("Normalization Sensitivity") > 0.40:
        out.append("The top group is sensitive to normalization assumptions, so review the multiplier inputs before making a pricing decision.")
    return out


def row_investigation_text(row: pd.Series) -> list[str]:
    cls = row.get("Signal / Classification", "n/a")
    gap = row.get("CPS Gap vs Expected %", np.nan)
    actual = row.get("Actual CPS", np.nan)
    expected = row.get("Expected CPS", np.nan)
    paid = row.get("Records With CPS", np.nan)
    conf = row.get("Confidence", "n/a")
    out = [f"Signal: {cls} with {conf} confidence."]
    out.append(f"Actual CPS is {money(actual)} versus expected CPS of {money(expected)}, a gap of {pct(gap)} based on {int(paid):,} records with CPS." if pd.notna(paid) else f"Actual CPS is {money(actual)} versus expected CPS of {money(expected)}, a gap of {pct(gap)}.")
    if pd.notna(row.get("Normalized Cost", np.nan)):
        out.append(f"Normalized cost is {money(row.get('Normalized Cost'))}. Use it as supporting evidence, not the final decision rule.")
    if pd.notna(row.get("Cleansheet Conservative", np.nan)) or pd.notna(row.get("Cleansheet Aggressive", np.nan)):
        out.append(f"Cleansheet range: conservative {money(row.get('Cleansheet Conservative'))}, aggressive {money(row.get('Cleansheet Aggressive'))}.")
    if row.get("Business Note", ""):
        out.append(f"Business note: {row.get('Business Note')}")
    return out


def _fmt_issue_value(value, kind: str) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        if kind == "money":
            return money(value)
        if kind == "pct":
            return pct(value)
        if kind == "plain_pct":
            return plain_pct(value)
        if kind == "int":
            return f"{int(round(float(value))):,}"
        if kind == "num":
            return f"{float(value):,.1f}"
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def _root_cause_items(row: pd.Series, baselines: dict[str, float]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []

    def add(label: str, detail: str, severity: int, recommendation: str):
        items.append(
            {
                "label": label,
                "detail": detail,
                "severity": str(severity),
                "recommendation": recommendation,
            }
        )

    records_with_cps = row.get("Records With CPS", 0) or 0
    min_group_n = int(baselines.get("min_group_n", 0) or 0)
    if records_with_cps < min_group_n:
        add(
            "Low evidence / thin volume",
            f"{_fmt_issue_value(records_with_cps, 'int')} paid rows vs minimum {min_group_n:,}.",
            5,
            "Validate invoices, missing cost capture, and data completeness before commercial action.",
        )

    distance = row.get("median_distance_val", np.nan)
    baseline_distance = baselines.get("median_distance_val", np.nan)
    if pd.notna(distance) and pd.notna(baseline_distance) and baseline_distance > 0 and distance >= baseline_distance * 1.15:
        add(
            "Long route length",
            f"Median distance {_fmt_issue_value(distance, 'num')} is above peer median {_fmt_issue_value(baseline_distance, 'num')}.",
            4,
            "Review route sequencing, stop consolidation, and whether the market should be served from a different DC or delivery pattern.",
        )

    miles_per_stop = row.get("median_miles_per_stop_norm_multiplier_qt", np.nan)
    baseline_miles_per_stop = baselines.get("median_miles_per_stop_norm_multiplier_qt", np.nan)
    if pd.notna(miles_per_stop) and pd.notna(baseline_miles_per_stop) and baseline_miles_per_stop > 0 and miles_per_stop >= baseline_miles_per_stop * 1.15:
        add(
            "Inefficient miles-per-stop profile",
            f"Miles-per-stop multiplier {_fmt_issue_value(miles_per_stop, 'num')} is above peer median {_fmt_issue_value(baseline_miles_per_stop, 'num')}.",
            4,
            "Consolidate stops, remove deadhead, and re-time routes to improve stop density.",
        )

    stop_count = row.get("median_stop_count_val_route_lvl", np.nan)
    baseline_stop_count = baselines.get("median_stop_count_val_route_lvl", np.nan)
    tote_count = row.get("median_tote_count_val_route_lvl", np.nan)
    baseline_tote_count = baselines.get("median_tote_count_val_route_lvl", np.nan)
    if pd.notna(stop_count) and pd.notna(baseline_stop_count) and baseline_stop_count > 0 and stop_count <= baseline_stop_count * 0.85:
        add(
            "Low stop density",
            f"Median route stops {_fmt_issue_value(stop_count, 'num')} are below peer median {_fmt_issue_value(baseline_stop_count, 'num')}.",
            4,
            "Increase drop density, combine shipments, or shift volume into fuller routes where feasible.",
        )
    if pd.notna(tote_count) and pd.notna(baseline_tote_count) and baseline_tote_count > 0 and tote_count <= baseline_tote_count * 0.85:
        add(
            "Low tote density",
            f"Median route totes {_fmt_issue_value(tote_count, 'num')} are below peer median {_fmt_issue_value(baseline_tote_count, 'num')}.",
            3,
            "Review consolidation rules, shipment packaging, and how loads are being split across stops.",
        )

    geo_multiplier = row.get("median_geography_multiplier", np.nan)
    baseline_geo_multiplier = baselines.get("median_geography_multiplier", np.nan)
    geo_mean = row.get("median_geo_mean", np.nan)
    baseline_geo_mean = baselines.get("median_geo_mean", np.nan)
    if pd.notna(geo_multiplier) and pd.notna(baseline_geo_multiplier) and baseline_geo_multiplier > 0 and geo_multiplier >= baseline_geo_multiplier * 1.10:
        add(
            "Geography / lane complexity",
            f"Geography multiplier {_fmt_issue_value(geo_multiplier, 'num')} is above peer median {_fmt_issue_value(baseline_geo_multiplier, 'num')}.",
            4,
            "Reassess DC assignment, territory design, and whether a nearer xdock or alternate carrier mix would reduce travel burden.",
        )
    elif pd.notna(geo_mean) and pd.notna(baseline_geo_mean) and baseline_geo_mean > 0 and geo_mean >= baseline_geo_mean * 1.10:
        add(
            "Geography / density pressure",
            f"Geo mean {_fmt_issue_value(geo_mean, 'num')} is above peer median {_fmt_issue_value(baseline_geo_mean, 'num')}.",
            3,
            "Check whether geography and density assumptions are structurally making this market expensive to serve.",
        )

    base_cost = row.get("median_lastmile_base_cost", np.nan)
    fuel_cost = row.get("median_lastmile_fuel_cost", np.nan)
    misc_cost = row.get("median_lastmile_misc_cost", np.nan)
    total_cost = row.get("median_lastmile_total_cost", np.nan)
    if pd.notna(total_cost) and total_cost > 0:
        base_share = base_cost / total_cost if pd.notna(base_cost) else np.nan
        fuel_share = fuel_cost / total_cost if pd.notna(fuel_cost) else np.nan
        misc_share = misc_cost / total_cost if pd.notna(misc_cost) else np.nan
        if pd.notna(base_share) and base_share >= 0.55:
            add(
                "Base cost pressure",
                f"Base cost is {plain_pct(base_share)} of total cost.",
                3,
                "Review contract rate structure, fixed-fee components, and market-level pricing assumptions.",
            )
        if pd.notna(fuel_share) and fuel_share >= 0.25:
            add(
                "Fuel cost pressure",
                f"Fuel cost is {plain_pct(fuel_share)} of total cost.",
                3,
                "Check fuel surcharge logic, route length, and whether distance-driven savings are available.",
            )
        if pd.notna(misc_share) and misc_share >= 0.15:
            add(
                "Miscellaneous cost pressure",
                f"Misc cost is {plain_pct(misc_share)} of total cost.",
                3,
                "Audit accessorials, exception charges, and other non-core billable items.",
            )

    sensitivity = row.get("Combined Sensitivity", np.nan)
    if pd.notna(sensitivity) and sensitivity > 0.40:
        add(
            "Benchmark sensitivity",
            f"Combined sensitivity is {_fmt_issue_value(sensitivity, 'pct')}.",
            3,
            "Treat normalization and cleansheet outputs as directional only until the multiplier inputs are reviewed.",
        )

    if str(row.get("Business Note", "")) == "Model and cleansheet both point high":
        add(
            "Triangulated overpay signal",
            "Model, normalized cost, and cleansheet are aligned.",
            5,
            "Prioritize this market for commercial review and sourcing action.",
        )

    if "underpay" in str(row.get("Signal / Classification", "")).lower():
        add(
            "Low-cost signal",
            "The market is trading below expected cost.",
            2,
            "Validate that the low cost is real before treating it as a savings opportunity; underbilling can look like underpay.",
        )

    if not items and pd.notna(row.get("Actual CPS", np.nan)) and pd.notna(row.get("Expected CPS", np.nan)):
        add(
            "Residual overage",
            f"Actual CPS {_fmt_issue_value(row.get('Actual CPS'), 'money')} versus expected CPS {_fmt_issue_value(row.get('Expected CPS'), 'money')}.",
            1,
            "Review the largest route exceptions, service pattern, and market-level operating assumptions.",
        )

    items.sort(key=lambda x: int(x["severity"]), reverse=True)
    return items


def root_cause_summary(row: pd.Series, baselines: dict[str, float]) -> str:
    items = _root_cause_items(row, baselines)
    if not items:
        return "No strong root-cause signal detected."
    top = items[0]
    return f"{top['label']}: {top['detail']}"


def recommendation_text(row: pd.Series, baselines: dict[str, float]) -> list[str]:
    items = _root_cause_items(row, baselines)
    labels = {item["label"] for item in items}
    recs: list[str] = []

    if {"Low evidence / thin volume"} & labels:
        recs.append("Start with invoice and data-quality validation before any pricing or network action.")
    if {"Long route length", "Inefficient miles-per-stop profile", "Low stop density", "Low tote density"} & labels:
        recs.append("Review route consolidation, stop density, dispatch sequencing, and whether the market should be rebalanced to a closer service point.")
    if {"Geography / lane complexity", "Geography / density pressure"} & labels:
        recs.append("Reassess fill DC allocation and service-point assignment for structural travel reduction opportunities.")
    if {"Base cost pressure", "Fuel cost pressure", "Miscellaneous cost pressure"} & labels:
        recs.append("Audit the pricing structure and billable charges, then renegotiate the highest-cost components.")
    if "Benchmark sensitivity" in labels:
        recs.append("Verify normalization inputs and cleansheet assumptions before making a final commercial call.")
    if "Triangulated overpay signal" in labels:
        recs.append("Prioritize this xdock for sourcing review because the model and benchmark layers agree.")
    if not recs:
        recs.append("Use the market-level exception summary to identify the largest route or billing drivers, then validate against the underlying shipment detail.")
    if "underpay" in str(row.get("Signal / Classification", "")).lower():
        recs.append("If the market is under expected cost, confirm there is no missing charge capture before treating it as savings.")
    return recs


def _impact_band(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def ranked_recommendations(row: pd.Series, baselines: dict[str, float], use_hybrid: bool = True) -> list[dict[str, str]]:
    profile = str(baselines.get("recommender_profile", "Benchmark-heavy")).strip()
    if profile not in {"Benchmark-heavy", "Balanced"}:
        profile = "Benchmark-heavy"

    actual = float(row.get("Actual CPS", np.nan)) if pd.notna(row.get("Actual CPS", np.nan)) else np.nan
    expected = float(row.get("Expected CPS", np.nan)) if pd.notna(row.get("Expected CPS", np.nan)) else np.nan
    normalized = float(row.get("Normalized Cost", np.nan)) if pd.notna(row.get("Normalized Cost", np.nan)) else np.nan
    clean_aggr = float(row.get("Cleansheet Aggressive", np.nan)) if pd.notna(row.get("Cleansheet Aggressive", np.nan)) else np.nan

    model_gap = max((actual / expected) - 1, 0.0) if pd.notna(actual) and pd.notna(expected) and expected > 0 else 0.0
    norm_gap = max((actual / normalized) - 1, 0.0) if pd.notna(actual) and pd.notna(normalized) and normalized > 0 else 0.0
    clean_gap = max((actual / clean_aggr) - 1, 0.0) if pd.notna(actual) and pd.notna(clean_aggr) and clean_aggr > 0 else 0.0

    normalized_ship = float(row.get("median_normalized_cost_shipment", np.nan)) if pd.notna(row.get("median_normalized_cost_shipment", np.nan)) else np.nan
    normalized_density = float(row.get("median_normalized_cost_density", np.nan)) if pd.notna(row.get("median_normalized_cost_density", np.nan)) else np.nan
    ship_norm_gap = max((actual / normalized_ship) - 1, 0.0) if pd.notna(actual) and pd.notna(normalized_ship) and normalized_ship > 0 else 0.0
    density_norm_gap = max((actual / normalized_density) - 1, 0.0) if pd.notna(actual) and pd.notna(normalized_density) and normalized_density > 0 else 0.0

    action_probabilities = recommendation_action_probabilities(row) if use_hybrid else {}

    distance = float(row.get("median_distance_val", 0) or 0)
    distance_base = float(baselines.get("median_distance_val", 0) or 0)
    distance_excess = max((distance / distance_base) - 1, 0.0) if distance_base > 0 else 0.0

    miles_mult = float(row.get("median_miles_per_stop_norm_multiplier_qt", 0) or 0)
    miles_base = float(baselines.get("median_miles_per_stop_norm_multiplier_qt", 0) or 0)
    miles_excess = max((miles_mult / miles_base) - 1, 0.0) if miles_base > 0 else 0.0

    shipment_mult = float(row.get("median_shipment_norm_multiplier_qt", 0) or 0)
    shipment_base = float(baselines.get("median_shipment_norm_multiplier_qt", 0) or 0)
    shipment_excess = max((shipment_mult / shipment_base) - 1, 0.0) if shipment_base > 0 else 0.0

    geo_mult = float(row.get("median_geography_multiplier", 0) or 0)
    geo_base = float(baselines.get("median_geography_multiplier", 0) or 0)
    geo_excess = max((geo_mult / geo_base) - 1, 0.0) if geo_base > 0 else 0.0

    stop_count = float(row.get("median_stop_count_val_route_lvl", 0) or 0)
    stop_base = float(baselines.get("median_stop_count_val_route_lvl", 0) or 0)
    stop_deficit = max((stop_base / stop_count) - 1, 0.0) if stop_count > 0 and stop_base > 0 else 0.0

    fuel_share = 0.0
    base_share = 0.0
    misc_share = 0.0
    total_cost = row.get("median_lastmile_total_cost", np.nan)
    if pd.notna(total_cost) and total_cost > 0:
        fuel = float(row.get("median_lastmile_fuel_cost", 0) or 0)
        base = float(row.get("median_lastmile_base_cost", 0) or 0)
        misc = float(row.get("median_lastmile_misc_cost", 0) or 0)
        fuel_share = max(fuel / total_cost, 0.0)
        base_share = max(base / total_cost, 0.0)
        misc_share = max(misc / total_cost, 0.0)

    benchmark_alignment = min(1.0, (0.45 * clean_gap) + (0.35 * norm_gap) + (0.20 * model_gap))

    if profile == "Balanced":
        action_rows = [
            {
                "Action": "Renegotiate rate card / sourcing",
                "Score": min(1.0, (0.35 * clean_gap) + (0.20 * norm_gap) + (0.35 * model_gap) + (0.10 * base_share)),
                "Reason": "Model gap and benchmarks both support a pricing review opportunity.",
            },
            {
                "Action": "Route consolidation and stop-density lift",
                "Score": min(1.0, (0.20 * norm_gap) + (0.25 * miles_excess) + (0.20 * density_norm_gap) + (0.20 * stop_deficit) + (0.15 * distance_excess)),
                "Reason": "Operational route profile and normalized-cost pressure are both elevated.",
            },
            {
                "Action": "Optimize fill DC allocation",
                "Score": min(1.0, (0.20 * clean_gap) + (0.20 * distance_excess) + (0.20 * miles_excess) + (0.20 * geo_excess) + (0.20 * model_gap)),
                "Reason": "DC allocation and model residuals indicate structural fulfillment-assignment opportunity.",
            },
            {
                "Action": "Fuel program and surcharge validation",
                "Score": min(1.0, (0.35 * fuel_share) + (0.20 * clean_gap) + (0.20 * distance_excess) + (0.10 * geo_excess) + (0.15 * model_gap)),
                "Reason": "Fuel mix and expected-cost overage indicate surcharge and distance review.",
            },
            {
                "Action": "Invoice and accessorial audit",
                "Score": min(1.0, (0.50 * misc_share) + (0.30 * model_gap) + (0.20 * norm_gap)),
                "Reason": "Misc-charge concentration and model overage indicate billing-quality audit priority.",
            },
        ]
    else:
        action_rows = [
            {
                "Action": "Renegotiate rate card / sourcing",
                "Score": min(1.0, (0.50 * clean_gap) + (0.25 * norm_gap) + (0.15 * model_gap) + (0.10 * base_share)),
                "Reason": "Cleansheet and normalized cost indicate market pricing above benchmark.",
            },
            {
                "Action": "Route consolidation and stop-density lift",
                "Score": min(1.0, (0.20 * norm_gap) + (0.30 * miles_excess) + (0.20 * density_norm_gap) + (0.15 * stop_deficit) + (0.15 * distance_excess)),
                "Reason": "Normalized-cost pressure plus route profile suggests network inefficiency.",
            },
            {
                "Action": "Optimize fill DC allocation",
                "Score": min(1.0, (0.30 * clean_gap) + (0.20 * distance_excess) + (0.20 * miles_excess) + (0.15 * geo_excess) + (0.15 * shipment_excess)),
                "Reason": "Cleansheet and travel burden suggest structural fill-DC assignment redesign.",
            },
            {
                "Action": "Fuel program and surcharge validation",
                "Score": min(1.0, (0.40 * fuel_share) + (0.20 * clean_gap) + (0.15 * distance_excess) + (0.15 * geo_excess) + (0.10 * shipment_excess)),
                "Reason": "Fuel share and benchmark gaps indicate surcharge and route-length opportunity.",
            },
            {
                "Action": "Invoice and accessorial audit",
                "Score": min(1.0, (0.45 * misc_share) + (0.25 * norm_gap) + (0.15 * model_gap) + (0.15 * ship_norm_gap)),
                "Reason": "Misc-charge concentration and benchmark gaps suggest billing-quality leakage.",
            },
        ]

    for row_item in action_rows:
        base_score = float(row_item["Score"])
        ml_probability = float(action_probabilities.get(row_item["Action"], np.nan)) if action_probabilities else np.nan
        if pd.notna(ml_probability):
            row_item["ML Probability"] = ml_probability
            row_item["Score"] = float((0.70 * base_score) + (0.30 * ml_probability))
        else:
            row_item["ML Probability"] = np.nan
            row_item["Score"] = base_score
        row_item["Impact"] = _impact_band(float(row_item["Score"]))
        confidence_basis = (0.45 * benchmark_alignment) + (0.25 * base_score)
        if pd.notna(ml_probability):
            confidence_basis += 0.30 * ml_probability
        else:
            confidence_basis += 0.30 * base_score
        row_item["Confidence"] = _impact_band(float(confidence_basis))

    ranked = sorted(action_rows, key=lambda x: x["Score"], reverse=True)
    return ranked


def ranked_recommendations_rules(row: pd.Series, baselines: dict[str, float]) -> list[dict[str, str]]:
    return ranked_recommendations(row, baselines, use_hybrid=False)


def top_root_cause_label(row: pd.Series, baselines: dict[str, float]) -> str:
    items = _root_cause_items(row, baselines)
    return items[0]["label"] if items else "n/a"


def top_recommendation(row: pd.Series, baselines: dict[str, float]) -> str:
    ranked = ranked_recommendations(row, baselines)
    return ranked[0]["Action"] if ranked else "n/a"


def top_action_impact(row: pd.Series, baselines: dict[str, float]) -> str:
    ranked = ranked_recommendations(row, baselines)
    return ranked[0]["Impact"] if ranked else "n/a"


def top_action_confidence(row: pd.Series, baselines: dict[str, float]) -> str:
    ranked = ranked_recommendations(row, baselines)
    return ranked[0]["Confidence"] if ranked else "n/a"


def second_recommendation(row: pd.Series, baselines: dict[str, float]) -> str:
    ranked = ranked_recommendations(row, baselines)
    return ranked[1]["Action"] if len(ranked) > 1 else "n/a"


def second_action_impact(row: pd.Series, baselines: dict[str, float]) -> str:
    ranked = ranked_recommendations(row, baselines)
    return ranked[1]["Impact"] if len(ranked) > 1 else "n/a"


def second_action_confidence(row: pd.Series, baselines: dict[str, float]) -> str:
    ranked = ranked_recommendations(row, baselines)
    return ranked[1]["Confidence"] if len(ranked) > 1 else "n/a"


def _popup_raw_records(row: pd.Series, raw_df: pd.DataFrame | None) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    records = raw_df.copy()
    selected_market = str(row.get("Market / Xdock", "")).strip().upper()
    if selected_market and "XDOCK" in records.columns:
        xdock_norm = records["XDOCK"].astype(str).str.strip().str.upper()
        records = records[xdock_norm.eq(selected_market)]

    selected_carrier = str(row.get("Carrier", "")).strip()
    if selected_carrier and selected_carrier not in {"All carriers", "n/a"} and "CARRIER_SCAC_CD" in records.columns:
        carrier_norm = records["CARRIER_SCAC_CD"].astype(str).str.strip().str.upper()
        records = records[carrier_norm.eq(selected_carrier.upper())]

    # If filtering produced no rows, return market-level rows as a fallback.
    if records.empty and selected_market and "XDOCK" in raw_df.columns:
        raw_xdock_norm = raw_df["XDOCK"].astype(str).str.strip().str.upper()
        records = raw_df[raw_xdock_norm.eq(selected_market)].copy()

    return records


def render_recommendation_detail_body(row: pd.Series, baselines: dict[str, float], raw_df: pd.DataFrame | None = None) -> None:
    ranked = ranked_recommendations(row, baselines)
    st.markdown(f"**Market / Xdock:** {row.get('Market / Xdock', 'n/a')}")
    st.markdown(
        f"- Actual CPS: {money(row.get('Actual CPS'))}"
        f" | Expected CPS: {money(row.get('Expected CPS'))}"
        f" | Gap: {pct(row.get('CPS Gap vs Expected %'))}"
    )
    st.markdown(
        f"- Normalized Cost: {money(row.get('Normalized Cost'))}"
        f" | Cleansheet Aggressive: {money(row.get('Cleansheet Aggressive'))}"
        f" | Records With CPS: {int(row.get('Records With CPS', 0)):,}"
    )
    st.markdown(f"- Top root cause: {root_cause_summary(row, baselines)}")

    bench_rows = [
        {"Metric": "Actual", "Value": row.get("Actual CPS", np.nan)},
        {"Metric": "Expected", "Value": row.get("Expected CPS", np.nan)},
        {"Metric": "Normalized", "Value": row.get("Normalized Cost", np.nan)},
        {"Metric": "CS Agg", "Value": row.get("Cleansheet Aggressive", np.nan)},
    ]
    if "Expected CPS CS Model" in row.index:
        bench_rows.append({"Metric": "CS Model", "Value": row.get("Expected CPS CS Model", np.nan)})
    bench_df = pd.DataFrame(bench_rows)
    bench_df["Value"] = pd.to_numeric(bench_df["Value"], errors="coerce")
    bench_df = bench_df.dropna(subset=["Value"])
    if len(bench_df) >= 2:
        bench_order = ["Actual", "Expected", "Normalized", "CS Agg", "CS Model"]
        bench_df["Metric"] = pd.Categorical(bench_df["Metric"], categories=bench_order, ordered=True)
        bench_df = bench_df.sort_values("Metric")
        fig_bench = px.bar(
            bench_df,
            x="Value",
            y="Metric",
            orientation="h",
            color="Metric",
            color_discrete_map={
                "Actual": "#C8102E",
                "Expected": "#5B7FA6",
                "Normalized": "#7AA6C2",
                "CS Agg": "#9EB7C9",
                "CS Model": "#B9CADA",
            },
        )
        fig_bench.update_traces(texttemplate="$%{x:,.2f}", textposition="outside", cliponaxis=False)
        fig_bench.update_layout(
            showlegend=False,
            height=170,
            margin=dict(l=0, r=0, t=4, b=0),
            xaxis_title="CPS ($)",
            yaxis_title=None,
            xaxis=dict(showgrid=True, gridcolor="#E6EAF0", zeroline=False),
            yaxis=dict(categoryorder="array", categoryarray=bench_order),
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    if ranked:
        display_rows = []
        for idx, action in enumerate(ranked[:5], start=1):
            ml_prob = action.get("ML Probability", np.nan)
            display_rows.append(
                {
                    "Rank": idx,
                    "Action": action.get("Action", "n/a"),
                    "Impact": action.get("Impact", "n/a"),
                    "Confidence": action.get("Confidence", "n/a"),
                    "Hybrid Score": float(action.get("Score", np.nan)) if pd.notna(action.get("Score", np.nan)) else np.nan,
                    "ML Probability": (100.0 * float(ml_prob)) if pd.notna(ml_prob) else np.nan,
                    "Reason": action.get("Reason", ""),
                }
            )
        st.dataframe(
            pd.DataFrame(display_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hybrid Score": st.column_config.NumberColumn(format="%.3f"),
                "ML Probability": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    raw_records = _popup_raw_records(row, raw_df)
    st.markdown("### Source records")
    if raw_records.empty:
        st.info("No source rows found for this selection under current filters.")
        return

    st.caption(f"Matched rows: {len(raw_records):,}. Showing first 300.")
    preferred_cols = [
        "XDOCK",
        "FILL_DC_CD",
        "YEAR",
        "MONTH",
        "ROUTE_ID",
        "CNSLDTN_ID",
        "JOB_ID",
        "DLVRY_ACTL_DATETIME",
        "DELIVERY_TYPE",
        "CARRIER_SCAC_CD",
        TARGET,
        "normalized_cost",
        "normalized_cost_geography",
        "normalized_cost_shipment",
        "normalized_cost_density",
        "GEOGRAPHY_MULTIPLIER",
        "shipment_norm_multiplier_qt",
        "miles_per_stop_norm_multiplier_qt",
    ]
    preview_cols = existing_cols(raw_records, preferred_cols)
    preview_df = raw_records[preview_cols].head(300) if preview_cols else raw_records.head(300)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    selected_market = str(row.get("Market / Xdock", "selected_market")).replace("/", "_")
    st.download_button(
        "Download matched source records",
        raw_records.to_csv(index=False).encode("utf-8"),
        file_name=f"source_records_{selected_market}.csv",
        mime="text/csv",
        key=f"download_source_records_{selected_market}",
    )


if hasattr(st, "dialog"):
    @st.dialog("Recommendation details", width="large")
    def open_recommendation_popup(section_title: str, row: pd.Series, baselines: dict[str, float], raw_df: pd.DataFrame | None = None) -> None:
        st.markdown(f"### {section_title}")
        render_recommendation_detail_body(row, baselines, raw_df=raw_df)
else:
    def open_recommendation_popup(section_title: str, row: pd.Series, baselines: dict[str, float], raw_df: pd.DataFrame | None = None) -> None:
        st.markdown(f"### {section_title}")
        render_recommendation_detail_body(row, baselines, raw_df=raw_df)


def recommendation_feature_columns(df: pd.DataFrame) -> list[str]:
    return existing_cols(
        df,
        [
            "Actual CPS",
            "Expected CPS",
            "CPS Gap vs Expected %",
            "Normalized Cost",
            "Gap vs Conservative Cleansheet %",
            "Gap vs Aggressive Cleansheet %",
            "Normalization Sensitivity",
            "Cleansheet Sensitivity",
            "Combined Sensitivity",
            "Records With CPS",
            "Total Records",
            "Confidence Score",
            "median_distance_val",
            "median_normalized_cost_shipment",
            "median_normalized_cost_density",
            "median_shipment_norm_multiplier_qt",
            "median_miles_per_stop_norm_multiplier_qt",
            "median_stop_count_val_route_lvl",
            "median_tote_count_val_route_lvl",
            "median_geography_multiplier",
            "median_geo_mean",
            "median_lastmile_total_cost",
            "median_lastmile_base_cost",
            "median_lastmile_fuel_cost",
            "median_lastmile_misc_cost",
        ],
    )


def recommendation_confidence_score(row: pd.Series) -> float:
    confidence = str(row.get("Confidence", "")).strip()
    mapping = {"High": 3.0, "Medium": 2.0, "Low": 1.0, "Very Low": 0.0}
    return mapping.get(confidence, 0.0)


def recommendation_action_probabilities(row: pd.Series) -> dict[str, float]:
    pack = RECOMMENDER_MODEL_PACK
    if not pack:
        return {}

    action_priors = pack.get("action_priors", {}) or {}
    if pack.get("status") != "hybrid model":
        return {action: float(action_priors.get(action, 0.0)) for action in RECOMMENDATION_ACTIONS}

    model = pack.get("model")
    feature_cols = pack.get("feature_cols", [])
    if model is None or not feature_cols:
        return {action: float(action_priors.get(action, 0.0)) for action in RECOMMENDATION_ACTIONS}

    row_df = pd.DataFrame([row]).reindex(columns=feature_cols)
    try:
        probs = model.predict_proba(row_df)[0]
    except Exception:
        return {action: float(action_priors.get(action, 0.0)) for action in RECOMMENDATION_ACTIONS}

    actions = list(model.named_steps["model"].classes_)
    predicted = {action: float(prob) for action, prob in zip(actions, probs)}
    merged = {}
    for action in RECOMMENDATION_ACTIONS:
        merged[action] = float(predicted.get(action, action_priors.get(action, 0.0)))
    return merged


def _recommendation_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df.copy()
    required = existing_cols(
        train_df,
        ["Actual CPS", "Expected CPS", "CPS Gap vs Expected %", "Normalized Cost", "Records With CPS"],
    )
    if len(required) < 4:
        return train_df.iloc[0:0].copy()
    train_df = train_df[train_df[required].notna().all(axis=1)].copy()
    if "Signal / Classification" in train_df.columns:
        train_df = train_df[train_df["Signal / Classification"].notna()].copy()
    return train_df


@st.cache_data(show_spinner="Training hybrid recommender...")
def train_recommendation_agent(df: pd.DataFrame, baselines: dict[str, float], random_state: int = 42) -> dict:
    train_df = _recommendation_training_frame(df)
    train_df["Confidence Score"] = train_df.apply(recommendation_confidence_score, axis=1)
    feature_cols = recommendation_feature_columns(train_df)
    if len(train_df) < 40 or len(feature_cols) < 6:
                            raw_df=filtered,
        uniform_priors = {action: 1.0 / len(RECOMMENDATION_ACTIONS) for action in RECOMMENDATION_ACTIONS}
        return {
            "status": "fallback",
            "model": None,
    # Render map independently from table selection.
            "action_priors": uniform_priors,
            "metrics": {
                "training_rows": int(len(train_df)),
                "feature_count": int(len(feature_cols)),
            map_data_all = map_df.copy()
            color_max = float(map_data_all["Estimated Opportunity"].fillna(0).max()) if map_data_all["Estimated Opportunity"].notna().any() else 1.0
            center_lat = float(map_data_all["latitude"].median())
            center_lon = float(map_data_all["longitude"].median())

            fig = px.scatter_map(
                map_data_all,
                lat="latitude",
                lon="longitude",
                color="Estimated Opportunity",
                size="Estimated Opportunity",
                size_max=26,
                range_color=[0, color_max],
                color_continuous_scale=["#BFD7EA", "#4F81BD", "#C8102E"],
                hover_name="Market / Xdock",
                hover_data={
                    "Confidence": True,
                    "Actual CPS": ":$.2f",
                    "Expected CPS": ":$.2f",
                    "Estimated Opportunity": ":$,.0f",
                    "latitude": False,
                    "longitude": False,
                } | ({"Market Name": True} if "Market Name" in map_data_all.columns else {}) | ({"Expected CPS CS Model": ":$.2f"} if "Expected CPS CS Model" in map_data_all.columns else {}),
                title="Overpay opportunity by xdock",
            )
            fig.update_traces(marker=dict(opacity=0.82, sizemode="area"))
            fig.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=48, b=0),
                map=dict(
                    style="open-street-map",
                    zoom=3.2,
                    center=dict(lat=center_lat, lon=center_lon),
                ),
            )

      .kpi {{ flex: 1; background: {MCK_LIGHT}; border: 1px solid #E2E8F0; border-radius: 10px; padding: .6rem .8rem; }}
      .kpi .lbl {{ font-size: .68rem; text-transform: uppercase; letter-spacing: .45px; color: #68717A; white-space: nowrap; }}
      .kpi .val {{ font-size: 1.18rem; font-weight: 800; color: {MCK_BLUE}; line-height: 1.25; }}
      .kpi .sub {{ font-size: .72rem; color: #68717A; overflow: hidden; text-overflow: ellipsis; }}
            .mi-row {{
                display: grid;
                grid-template-columns: repeat(7, minmax(0, 1fr));
                gap: .4rem;
                margin: .25rem 0 .75rem 0;
            }}
            .mi-card {{
                background: #F8FBFE;
                border: 1px solid #DCE8F3;
                border-radius: 8px;
                padding: .35rem .5rem;
                min-width: 0;
            }}
            .mi-lbl {{
                font-size: .62rem;
                text-transform: uppercase;
                letter-spacing: .35px;
                color: #6D7782;
                line-height: 1.2;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .mi-val {{
                font-size: .96rem;
                font-weight: 800;
                color: {MCK_BLUE};
                line-height: 1.1;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .mi-sub {{
                font-size: .64rem;
                color: #6D7782;
                line-height: 1.2;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .drill-wrap {{
                overflow-x: auto;
                border: 1px solid #DCE8F3;
                border-radius: 10px;
                background: #FFFFFF;
            }}
            .drill-table {{
                border-collapse: separate;
                border-spacing: 0;
                width: 100%;
                min-width: 1280px;
                font-size: .82rem;
            }}
            .drill-table thead th {{
                position: sticky;
                top: 0;
                z-index: 4;
                background: #EAF3FB;
                color: #123D63;
                text-align: left;
                padding: .5rem .6rem;
                border-bottom: 1px solid #CFE2F3;
                white-space: nowrap;
            }}
            .drill-table tbody td {{
                padding: .45rem .6rem;
                border-bottom: 1px solid #EEF2F6;
                white-space: nowrap;
            }}
            .drill-table tbody tr:nth-child(even) {{
                background: #FAFCFE;
            }}
            .drill-table .row-link {{
                display: block;
                width: 100%;
                border: 0;
                background: transparent;
                color: inherit;
                text-decoration: none;
                padding: 0;
                margin: 0;
                text-align: left;
                font: inherit;
                cursor: pointer;
            }}
            .drill-table .row-link:focus {{
                outline: none;
            }}
            .drill-table tbody tr:focus-within td {{
                background: #FFF3CD;
            }}
            .drill-table tbody tr:focus-within td:first-child {{
                background: #FFE8A1;
            }}
            .drill-table th:first-child,
            .drill-table td:first-child {{
                position: sticky;
                left: 0;
                z-index: 3;
                background: #F4F9FE;
                box-shadow: 1px 0 0 #DCE8F3;
                font-weight: 700;
            }}
            .drill-table thead th:first-child {{
                z-index: 5;
            }}
            .gap-pos {{ color: inherit; font-weight: 600; }}
            .gap-neg {{ color: #2B7A0B; font-weight: 700; }}
            .gap-mid {{ color: #53565A; font-weight: 600; }}
            @media (max-width: 1200px) {{
                .mi-row {{
                    grid-template-columns: repeat(4, minmax(0, 1fr));
                }}
            }}
            @media (max-width: 800px) {{
                .mi-row {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
            }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Last Mile Cost Intelligence (LMCI)</div>', unsafe_allow_html=True)
st.markdown('<div class="app-rule"></div>', unsafe_allow_html=True)

try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

all_dc_options = sorted(data["FILL_DC_CD"].dropna().astype(str).unique()) if "FILL_DC_CD" in data.columns else []
all_cust_type_options = sorted(data["CUST_BUS_TYP_DSCR"].dropna().astype(str).unique()) if "CUST_BUS_TYP_DSCR" in data.columns else []

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f'<div class="logo-wrap"><img src="{MCK_LOGO}" alt="McKesson" /></div>', unsafe_allow_html=True)
    st.header("Analysis setup")

    use_period_filter = False
    if {"YEAR", "MONTH"}.issubset(data.columns):
        use_period_filter = st.toggle("Filter to year/month", value=False)

    filtered = data
    if use_period_filter:
        years = sorted(filtered["YEAR"].dropna().unique())
        sel_year = st.selectbox("Year", years, index=len(years) - 1)
        months = sorted(filtered.loc[filtered["YEAR"].eq(sel_year), "MONTH"].dropna().unique())
        sel_month = st.selectbox("Month", months, index=len(months) - 1)
        filtered = filtered[(filtered["YEAR"].eq(sel_year)) & (filtered["MONTH"].eq(sel_month))].copy()

    dc_options = sorted(filtered["FILL_DC_CD"].dropna().astype(str).unique()) if "FILL_DC_CD" in filtered.columns else []
    sel_dcs = st.multiselect("Fill DC filter", dc_options, default=dc_options)
    if sel_dcs:
        filtered = filtered[filtered["FILL_DC_CD"].astype(str).isin(sel_dcs)].copy()

    sel_cust_types: list[str] = all_cust_type_options.copy()
    if "CUST_BUS_TYP_DSCR" in filtered.columns:
        cust_type_options = sorted(filtered["CUST_BUS_TYP_DSCR"].dropna().astype(str).unique())
        sel_cust_types = st.multiselect("Customer type filter", cust_type_options, default=cust_type_options)
        if sel_cust_types:
            filtered = filtered[filtered["CUST_BUS_TYP_DSCR"].astype(str).isin(sel_cust_types)].copy()

    group_cols = ["XDOCK"]

    st.divider()
    st.header("Signal thresholds")
    min_group_n = st.number_input("Minimum paid rows", min_value=5, max_value=5000, value=30, step=5)
    overpay_possible = st.slider("Overpay floor threshold", 0.05, 0.50, 0.10, 0.01)
    overpay_strong = st.slider("Overpay high-confidence threshold", 0.10, 1.00, 0.20, 0.01)
    underpay_possible = -st.slider("Possible underpay threshold", 0.05, 0.50, 0.10, 0.01)
    underpay_strong = -st.slider("Strong underpay threshold", 0.10, 1.00, 0.20, 0.01)
    use_p25_p75 = st.toggle("Fit P25/P75 bands", value=False, help="Leave off for speed. P10/P50/P90 is enough for the decision tool.")
    recommender_profile = st.selectbox(
        "Recommendation profile",
        ["Benchmark-heavy", "Balanced"],
        index=0,
        help="Benchmark-heavy prioritizes normalized cost and cleansheet; Balanced gives more weight to expected-CPS residual and route profile.",
    )

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

use_precomputed_default = _is_default_scorecard_request(
    use_period_filter=bool(use_period_filter),
    dc_options=all_dc_options,
    sel_dcs=list(sel_dcs),
    cust_type_options=all_cust_type_options,
    sel_cust_types=list(sel_cust_types),
    min_group_n=int(min_group_n),
    overpay_strong=float(overpay_strong),
    overpay_possible=float(overpay_possible),
    underpay_strong=float(underpay_strong),
    underpay_possible=float(underpay_possible),
    use_p25_p75=bool(use_p25_p75),
)

scorecard = None
business_view = None
model_metrics = None
if use_precomputed_default:
    precomputed = load_precomputed_scorecard()
    if precomputed is not None:
        scorecard, business_view, model_metrics = precomputed

if scorecard is None or business_view is None or model_metrics is None:
    try:
        scorecard, business_view, model_metrics = build_scorecard(
            filtered,
            group_cols=group_cols,
            min_group_n=int(min_group_n),
            overpay_strong=float(overpay_strong),
            overpay_possible=float(overpay_possible),
            underpay_strong=float(underpay_strong),
            underpay_possible=float(underpay_possible),
            use_p25_p75=bool(use_p25_p75),
        )
    except Exception as exc:
        st.error(f"Could not build scorecard: {exc}")
        st.stop()

# --------------------------------------------------------------------------- #
# KPI cards and automatic explanation
# --------------------------------------------------------------------------- #
counts = business_view["Signal / Classification"].value_counts()
overpay_count = int(counts.get("Overpay candidate", 0))
underpay = int(counts.get("Strong underpay candidate", 0) + counts.get("Possible underpay", 0))
not_enough = int(counts.get("Not enough evidence", 0))

high_conf = int((business_view["Confidence"].eq("High")).sum()) if "Confidence" in business_view.columns else 0
if business_view["CPS Gap vs Expected %"].notna().any():
    worst_idx = business_view["CPS Gap vs Expected %"].idxmax()
    worst = business_view.loc[worst_idx]
    worst_label = worst.get("Market Name", worst.get("Market / Xdock", "n/a"))
    worst_gap = worst.get("CPS Gap vs Expected %")
else:
    worst_label, worst_gap = "n/a", np.nan

cards = [
    ("Scored groups", f"{len(business_view):,}", "grain: Market (XDOCK)"),
    ("Overpay candidates", f"{overpay_count:,}", "single combined bucket"),
    ("Underpay signals", f"{underpay:,}", "validate missing charges"),
    ("Highest gap", pct(worst_gap), str(worst_label)[:32]),
    ("High confidence", f"{high_conf:,}", f"not enough evidence: {not_enough:,}"),
]
kpi_html = '<div class="kpi-row">' + "".join(
    f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val">{val}</div><div class="sub">{sub}</div></div>'
    for lbl, val, sub in cards
) + "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

rca_baselines = {
    "min_group_n": int(min_group_n),
    "recommender_profile": recommender_profile,
}
for col in [
    "median_distance_val",
    "median_shipment_norm_multiplier_qt",
    "median_miles_per_stop_norm_multiplier_qt",
    "median_stop_count_val_route_lvl",
    "median_tote_count_val_route_lvl",
    "median_geography_multiplier",
    "median_geo_mean",
]:
    if col in business_view.columns:
        rca_baselines[col] = business_view[col].median(skipna=True)

RECOMMENDER_MODEL_PACK = train_recommendation_agent(business_view, rca_baselines)
executive_geo_view = build_xdock_geo_view(filtered, business_view)
executive_context = dict(rca_baselines)
if isinstance(model_metrics, dict):
    executive_context["model_comparison"] = model_metrics.get("model_comparison")
    executive_context["model_metrics"] = model_metrics

with st.expander("Quick interpretation", expanded=False):
    st.markdown(
        f"- {overpay_count:,} overpay candidates, {underpay:,} underpay signals, and {not_enough:,} groups with not enough evidence."
    )
    st.markdown(
        "- Decision rule: use **Actual CPS vs Expected CPS** as the primary signal; use **normalized cost and cleansheet** as supporting evidence."
    )

# --------------------------------------------------------------------------- #
# Market investigation tool
# --------------------------------------------------------------------------- #
top_exec_tab, top_market_tab, top_reco_tab, top_business_tab = st.tabs(["Executive view", "Market investigation", "Recommendation agent", "Business views"])

with top_exec_tab:
    render_executive_view(executive_geo_view, executive_context)

with top_market_tab:
    st.subheader("Market investigation")
    market_options = business_view["Market / Xdock"].astype(str).tolist()
    if market_options:
        if "market_picker" not in st.session_state:
            st.session_state["market_picker"] = market_options[0]
        if st.session_state.get("market_picker") not in market_options:
            st.session_state["market_picker"] = market_options[0]

        selected_market = st.selectbox("Choose market / xdock", market_options, key="market_picker")
        row_df = business_view[business_view["Market / Xdock"].astype(str).eq(selected_market)]
        if not row_df.empty:
            row = row_df.iloc[0]
            classification = row.get("Signal / Classification", "n/a")
            confidence = row.get("Confidence", "n/a")
            cls_color = CLASS_COLORS.get(classification, MCK_GRAY)
            st.markdown(
                f'<div style="display:inline-block;padding:.35rem .9rem;border-radius:6px;'
                f'background:{cls_color};color:white;font-weight:700;font-size:.95rem;margin-bottom:.6rem;">'
                f'{classification} &nbsp;·&nbsp; {confidence} confidence</div>',
                unsafe_allow_html=True,
            )
            metrics = [
                ("Actual CPS", money(row.get("Actual CPS")), pct(row.get("CPS Gap vs Expected %"))),
                ("Expected CPS", money(row.get("Expected CPS")), "P50"),
                ("Expected high", money(row.get("Expected High CPS P90")), "P90"),
                ("Normalized", money(row.get("Normalized Cost")), plain_pct(row.get("Normalization Sensitivity"))),
                ("CS conservative", money(row.get("Cleansheet Conservative")), "benchmark"),
                ("CS aggressive", money(row.get("Cleansheet Aggressive")), "benchmark"),
                ("Records With CPS", f"{int(row.get('Records With CPS', 0)):,}", str(row.get("Confidence", ""))),
            ]
            mi_html = '<div class="mi-row">' + "".join(
                f'<div class="mi-card"><div class="mi-lbl">{lbl}</div><div class="mi-val">{val}</div><div class="mi-sub">{sub}</div></div>'
                for lbl, val, sub in metrics
            ) + "</div>"
            st.markdown(mi_html, unsafe_allow_html=True)
            st.markdown('<div class="tool-note">' + "<br>".join(row_investigation_text(row)) + "</div>", unsafe_allow_html=True)

            st.caption("Use the Recommender agent tab for ranked overpay queues, strict all-3-reference-cost filtering, and action drilldown.")

# --------------------------------------------------------------------------- #
# Business charts
# --------------------------------------------------------------------------- #
with top_business_tab:
    st.subheader("Business views")
    tab0, tab00, tab01, tab4, tab9, tab1, tab2, tab3, tab5, tab6, tab7, tab8, tab10 = st.tabs(
        [
            "Method Flow",
            "Methodology",
            "Signal summary",
            "Cleansheet triangulation",
            "Signal Buckets",
            "Actual vs expected",
            "Top overpay",
            "Cost band",
            "Cost Comparison",
            "Triangulation V1",
            "Triangulation V2",
            "Triangulation V3",
            "Customer Layer",
        ]
    )

hover_cols = existing_cols(
    business_view,
    [
        "Market / Xdock",
        "Market Name",
        "Signal / Classification",
        "Confidence",
        "Business Note",
        "Records With CPS",
        "Actual CPS",
        "Expected CPS",
        "Expected Low CPS P10",
        "Expected High CPS P90",
        "CPS Gap vs Expected %",
        "Normalized Cost",
        "Cleansheet Conservative",
        "Cleansheet Aggressive",
    ],
)

with tab0:

    st.subheader("Decision Framework")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.info(
            """
            **Model Inputs**

            Route Ops · Geography

            Normalization · Volume

            Cost Structure
            """
        )

    with c2:
        st.success(
            """
            **Expected CPS**

            Quantile Gradient Boosting Model

            P10 / P50 / P90
            """
        )

    with c3:
        st.warning(
            """
            **Model Validation**

            MAE

            Median Error

            R²
            """
        )

    with c4:
        st.error(
            """
            **Opportunity Signal**

            Actual CPS

            vs

            Expected CPS
            """
        )

    with c5:
        st.success(
            """
            **Recommendation**

            Overpay

            Underpay

            Review
            """
        )

    st.markdown("### Business Validation")

    v1, v2, v3 = st.columns(3)

    with v1:
        st.metric(
            "Confidence",
            "Trust Level"
        )

    with v2:
        st.metric(
            "Normalized Cost",
            "Internal Validation"
        )

    with v3:
        st.metric(
            "Cleansheet",
            "External Validation"
        )

    st.info(
        """
Layer 1: Model Validation → Is Expected CPS accurate?

Layer 2: Business Validation → Should we trust the opportunity?
        """
    )

    with st.expander("Model input features (by category)", expanded=False):
        st.markdown(
            """
| Category | Features |
| --- | --- |
| **Route Operations** | DISTANCE_VAL, STOP_COUNT_VAL_ROUTE_LVL, TOTE_COUNT_VAL_ROUTE_LVL, STOP_COUNT_VAL, TOTE_COUNT_VAL, ROUTE_COUNT_VAL |
| **Volume & Quality (count-based)** | paid_records, records, paid_rate |
| **Geography** | GEOGRAPHY_MULTIPLIER, geo_mean |
| **Normalization** | shipment_norm_multiplier_qt, miles_per_stop_norm_multiplier_qt, normalized_cost |
| **Cost Structure** | LASTMILE_TOTAL_COST, LASTMILE_BASE_COST, LASTMILE_FUEL_COST, LASTMILE_MISC_COST |
| **Categorical** | XDOCK and FILL_DC_CD |
            """
        )

    st.dataframe(
        pd.DataFrame([model_metrics]),
        hide_index=True,
        use_container_width=True
    )
    
with tab00:

    st.subheader("Decision Methodology")

    st.info(
        """
Expected CPS identifies the signal.
Normalized Cost, Cleansheet, and Confidence help validate the business opportunity.
        """
    )

    st.markdown("""
### Objective

Identify markets whose Cost Per Stop (CPS) is materially above or below what would be expected for a comparable operating profile.

The objective is not to find the highest-cost markets. The objective is to identify markets where actual CPS differs significantly from expected CPS after accounting for operational drivers.

---

### 1. Market-Level Aggregation

Shipment-level records are aggregated to market/xdock grain (`XDOCK`).

The model uses operational characteristics such as:

- `DISTANCE_VAL`
- `STOP_COUNT_VAL_ROUTE_LVL`, `TOTE_COUNT_VAL_ROUTE_LVL`, `STOP_COUNT_VAL`, `TOTE_COUNT_VAL`
- `ROUTE_COUNT_VAL` and market-level record counts (`paid_records`, `records`) for count-based volume
- `GEOGRAPHY_MULTIPLIER`, `geo_mean`
- `shipment_norm_multiplier_qt`, `miles_per_stop_norm_multiplier_qt`
- `LASTMILE_TOTAL_COST`, `LASTMILE_BASE_COST`, `LASTMILE_FUEL_COST`, `LASTMILE_MISC_COST`

Multiplier meaning (1-line): `GEOGRAPHY_MULTIPLIER` = lane/geography burden, `shipment_norm_multiplier_qt` = shipment volume intensity, `miles_per_stop_norm_multiplier_qt` = stop-density burden; higher values mean structurally harder/more expensive operating conditions versus baseline.

---

### 2. Expected CPS Estimation

A Qualtile Gradient Boosting Model predicts:

- P10 Expected CPS (model-predicted)
- P50 Expected CPS (model-predicted)
- P90 Expected CPS (model-predicted)

P50 is the model's prediction of what a comparable market should cost. It is treated as the primary expected cost.

P10 and P90 define the predicted operating range for comparable markets.

---

### 3. Gap Calculation

Primary signal:

Gap % = (Actual CPS / Expected CPS) - 1

Examples:

- +20% = Actual CPS is 20% above Expected CPS
- -15% = Actual CPS is 15% below Expected CPS

---

### 4. Validation Approach

#### Model Validation

Expected CPS is evaluated against Actual CPS using:

- MAE ($)
- Median Absolute Error ($)
- R²

These metrics measure how accurately the model predicts market cost.

#### Business Validation

After a market is flagged, supporting evidence is reviewed:

- Normalized Cost
- Conservative Cleansheet
- Aggressive Cleansheet
- Confidence

These metrics do not validate the model.

They help determine whether the overpay or underpay signal is credible and worth investigating.

---

### 5. Decision Hierarchy

1. Expected CPS (Primary Signal)
2. Actual CPS vs Expected CPS Gap
3. Confidence
4. Normalized Cost
5. Cleansheet Benchmarks

Expected CPS drives the classification.

Normalized Cost and Cleansheet provide supporting evidence to strengthen or challenge the recommendation.

---

### 6. Confidence Logic

Confidence is a rule-based score from 0 to 3 points. One point is added for each condition met:

1. Paid records are at least the minimum threshold (`paid_records >= min_group_n`)
2. Paid records are at least 3x the minimum threshold (`paid_records >= 3 * min_group_n`)
3. Combined sensitivity is stable (`combined_sensitivity_abs_pct <= 40%`)

Combined sensitivity is the average of:

- Normalization sensitivity: `abs((Normalized Cost / Actual CPS) - 1)`
- Cleansheet sensitivity: `abs((Cleansheet Midpoint / Actual CPS) - 1)`

Where Cleansheet Midpoint is the midpoint between conservative and aggressive cleansheet CPS when both are available.

Confidence labels:

- High: score >= 3
- Medium: score = 2
- Low: score = 1
- Very Low: score = 0

Very Low confidence groups are treated as Not enough evidence in final classification.

---

### 7. Classification Rules

Classification is assigned in priority order:

#### Overpay Candidate (Combined Bucket)
This bucket combines strong and possible overpay logic.

Strong-overpay trigger:
1. Gap % ≥ strong overpay threshold (default: +20%)
2. Actual CPS > P90 Expected CPS (actual exceeds the model's upper predicted range)
3. Confidence is High or Medium

#### Strong Underpay Candidate
All three conditions must be met:
1. Gap % ≤ strong underpay threshold (default: -20%)
2. Actual CPS < P10 Expected CPS (actual is below the model's lower predicted range)
3. Confidence is High or Medium

Possible-overpay trigger (also mapped into the same Overpay Candidate bucket):
1. Gap % ≥ possible overpay threshold (default: +10%)
2. (If P25/P75 enabled) Actual CPS > P75 Expected CPS
3. Group not already classified by higher-priority rules

#### Possible Underpay
1. Gap % ≤ possible underpay threshold (default: -10%)
2. (If P25/P75 enabled) Actual CPS < P25 Expected CPS
3. Group not already classified as Strong Underpay

#### Not Enough Evidence
Any of:
- Paid records below minimum threshold
- Confidence is Very Low
- Actual CPS is missing
- Expected P50 is missing

#### Normal / Inside Expected Band
All other groups that do not meet any of the above conditions.

---

### 8. Sensitivity Calculation

Sensitivity measures how much the supporting benchmarks (normalized cost, cleansheet) diverge from actual CPS.
A large divergence means the benchmarks and the actual rate tell very different stories, reducing confidence.

**Normalization Sensitivity:**
```
norm_sensitivity = abs((Normalized Cost / Actual CPS) - 1)
```
Example: Actual CPS = $30, Normalized Cost = $24 → sensitivity = abs(24/30 - 1) = 20%

**Cleansheet Sensitivity:**
```
cleansheet_sensitivity = abs((Cleansheet Midpoint / Actual CPS) - 1)
```
Where Cleansheet Midpoint = (Conservative + Aggressive) / 2

**Combined Sensitivity:**
```
combined_sensitivity = average(norm_sensitivity, cleansheet_sensitivity)
```

If combined sensitivity ≤ 40% → benchmarks are aligned with actual CPS → +1 confidence point.
If combined sensitivity > 40% → benchmarks diverge significantly → business note flags it as "Sensitive to normalization and cleansheet assumptions".

---

### 9. Recommender Agent (Current Implementation)

The recommender agent is a hybrid layer that ranks actions after the market is classified.

#### What it does today

- The app first produces a rules-based action ranking for each xdock.
- The ML layer then learns the top agent action from the current scored view as a pseudo-label.
- At inference time, the ML model predicts action probabilities and blends them with the rules-based scores.

#### Current ML target

- The training target is `Agent Top Action`, which is the top action produced by the current rules engine.
- This means the recommender is currently learning the agent’s behavior, not real commercial outcomes.
- It is not yet trained on labels like accepted recommendation, realized savings, or before/after CPS.

#### ML features used

- Actual CPS, Expected CPS, CPS Gap vs Expected %
- Normalized Cost and cleansheet gaps
- Records With CPS, Total Records, Confidence Score
- Route-level medians such as distance, miles-per-stop, stop count, tote count, geography multiplier, and cost mix

#### Model type

- Multinomial Logistic Regression
- Median imputation
- Standard scaling
- 80/20 train-test split
- Balanced class weights

#### Blend logic

- Rules-based score remains the primary signal.
- ML action probability is blended into the action score.
- If the current filtered data is too small or too sparse, the app falls back to rules-only.

#### Interpretation

- This is a hybrid recommender, not a true outcome-trained recommender yet.
- To make it data-based in the commercial sense, the app needs a real target table with review decisions or realized savings.
    """)

    st.subheader("Current Model Statistics")

    comparison_rows = model_metrics.get("model_comparison") if isinstance(model_metrics, dict) else None
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        baseline_row = comparison_df.iloc[0]
        challenger_row = comparison_df.iloc[1] if len(comparison_df) > 1 else None

        m1, m2, m3 = st.columns(3)
        with m1:
            delta_text = None
            if challenger_row is not None and pd.notna(challenger_row.get("MAE Cost")) and pd.notna(baseline_row.get("MAE Cost")):
                delta_text = f"{challenger_row['MAE Cost'] - baseline_row['MAE Cost']:+.2f} vs baseline"
            st.metric("CS Model MAE", money(challenger_row.get("MAE Cost")) if challenger_row is not None else "n/a", delta=delta_text)
        with m2:
            delta_text = None
            if challenger_row is not None and pd.notna(challenger_row.get("Median Abs Error Cost")) and pd.notna(baseline_row.get("Median Abs Error Cost")):
                delta_text = f"{challenger_row['Median Abs Error Cost'] - baseline_row['Median Abs Error Cost']:+.2f} vs baseline"
            st.metric(
                "CS Model Median Error",
                money(challenger_row.get("Median Abs Error Cost")) if challenger_row is not None else "n/a",
                delta=delta_text,
            )
        with m3:
            delta_text = None
            if challenger_row is not None and pd.notna(challenger_row.get("R2 Log Target")) and pd.notna(baseline_row.get("R2 Log Target")):
                delta_text = f"{challenger_row['R2 Log Target'] - baseline_row['R2 Log Target']:+.3f} vs baseline"
            st.metric(
                "CS Model R2",
                f"{challenger_row.get('R2 Log Target'):.3f}" if challenger_row is not None and pd.notna(challenger_row.get("R2 Log Target")) else "n/a",
                delta=delta_text,
            )

        st.caption("Baseline model remains the production Expected CPS signal. The CS model adds the average of conservative and aggressive cleansheet as an extra feature and is scored on the same train/test split for comparison.")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "MAE Cost": st.column_config.NumberColumn(format="$%.2f"),
                "Median Abs Error Cost": st.column_config.NumberColumn(format="$%.2f"),
                "R2 Log Target": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    else:
        st.dataframe(
            pd.DataFrame([model_metrics]),
            use_container_width=True,
            hide_index=True,
        )

with top_reco_tab:
    st.subheader("Recommender agent")
    
    # Combined search for both tables
    combined_search_query = st.text_input(
        "Search overpay queues",
        value="",
        key="combined_overpay_search_query",
        placeholder="Type market, root cause, recommendation, or confidence",
    )

    overpay_queue = business_view[business_view["Signal / Classification"].eq("Overpay candidate")].copy()
    if len(overpay_queue):
        overpay_queue["Estimated Opportunity"] = np.where(
            overpay_queue["CPS Gap vs Expected"].fillna(0).gt(0),
            overpay_queue["CPS Gap vs Expected"].fillna(0) * overpay_queue["Records With CPS"].fillna(0),
            0.0,
        )
        overpay_queue["Top Root Cause"] = overpay_queue.apply(lambda r: top_root_cause_label(r, rca_baselines), axis=1)
        overpay_queue["Recommendation 1"] = overpay_queue.apply(lambda r: top_recommendation(r, rca_baselines), axis=1)
        overpay_queue["Action 1 Impact"] = overpay_queue.apply(lambda r: top_action_impact(r, rca_baselines), axis=1)
        overpay_queue["Action 1 Confidence"] = overpay_queue.apply(lambda r: top_action_confidence(r, rca_baselines), axis=1)
        overpay_queue["Recommendation 2"] = overpay_queue.apply(lambda r: second_recommendation(r, rca_baselines), axis=1)
        overpay_queue["Action 2 Impact"] = overpay_queue.apply(lambda r: second_action_impact(r, rca_baselines), axis=1)
        overpay_queue["Action 2 Confidence"] = overpay_queue.apply(lambda r: second_action_confidence(r, rca_baselines), axis=1)
        opp_max_reco = float(overpay_queue["Estimated Opportunity"].fillna(0).max())
        gap_max_reco = float(overpay_queue["CPS Gap vs Expected %"].fillna(0).max())
        opp_max_reco = max(opp_max_reco, 1.0)
        gap_max_reco = max(gap_max_reco, 0.01)

        st.markdown("### Overpay queue (Expected CPS gap driven)")
        queue_cols = existing_cols(
            overpay_queue,
            [
                "Market / Xdock",
                "Estimated Opportunity",
                "CPS Gap vs Expected %",
                "Confidence",
                "Actual CPS",
                "Expected CPS",
                "Normalized Cost",
                "Cleansheet Aggressive",
                "Records With CPS",
                "Top Root Cause",
                "Recommendation 1",
                "Action 1 Impact",
                "Action 1 Confidence",
                "Recommendation 2",
                "Action 2 Impact",
                "Action 2 Confidence",
            ],
        )
        overpay_queue = overpay_queue.sort_values(["CPS Gap vs Expected %", "Records With CPS"], ascending=[False, False])
        overpay_show = filter_table_rows(overpay_queue[queue_cols], combined_search_query).head(200)
        st.caption(f"Showing {len(overpay_show)} of {len(overpay_queue)} rows")

        overpay_market_options = overpay_queue["Market / Xdock"].astype(str).tolist()
        if overpay_market_options:
            pending_overpay_picker = st.session_state.pop("overpay_picker_pending", None)
            if pending_overpay_picker in overpay_market_options:
                st.session_state["overpay_picker"] = pending_overpay_picker
            if "overpay_picker" not in st.session_state:
                st.session_state["overpay_picker"] = overpay_market_options[0]
            if st.session_state.get("overpay_picker") not in overpay_market_options:
                st.session_state["overpay_picker"] = overpay_market_options[0]

            selected_overpay_market = st.selectbox(
                "Drilldown market from overpay queue",
                overpay_market_options,
                key="overpay_picker",
            )

            if st.button("Open overpay drilldown popup", key="open_overpay_popup_btn"):
                selected_row_df = overpay_queue[
                    overpay_queue["Market / Xdock"].astype(str).eq(selected_overpay_market)
                ]
                if not selected_row_df.empty:
                    open_recommendation_popup("Overpay queue drilldown", selected_row_df.iloc[0], rca_baselines, raw_df=filtered)

        selected_from_overpay_queue = None
        try:
            overpay_event = st.dataframe(
                overpay_show,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="overpay_queue_table",
                column_config={
                    "Estimated Opportunity": st.column_config.ProgressColumn(min_value=0.0, max_value=opp_max_reco, format="$%.0f"),
                    "Actual CPS": st.column_config.NumberColumn(format="$%.2f"),
                    "Expected CPS": st.column_config.NumberColumn(format="$%.2f"),
                    "Normalized Cost": st.column_config.NumberColumn(format="$%.2f"),
                    "Cleansheet Aggressive": st.column_config.NumberColumn(format="$%.2f"),
                    "CPS Gap vs Expected %": st.column_config.ProgressColumn(min_value=0.0, max_value=gap_max_reco, format="%.1f%%"),
                },
            )
            selected_rows = selected_rows_from_event(overpay_event)
            if selected_rows:
                row_idx = selected_rows[0]
                if 0 <= row_idx < len(overpay_show):
                    selected_from_overpay_queue = str(overpay_show.iloc[row_idx]["Market / Xdock"])
                    st.session_state["overpay_selected_market"] = selected_from_overpay_queue
        except TypeError:
            st.dataframe(
                overpay_show,
                use_container_width=True,
                hide_index=True,
            )

        if selected_from_overpay_queue is None:
            persisted_overpay_market = st.session_state.get("overpay_selected_market")
            if (
                persisted_overpay_market is not None
                and "Market / Xdock" in overpay_show.columns
                and persisted_overpay_market in set(overpay_show["Market / Xdock"].astype(str))
            ):
                selected_from_overpay_queue = persisted_overpay_market

        # Auto-open popup when a new row is selected in the table.
        if selected_from_overpay_queue:
            last_opened_overpay_market = st.session_state.get("overpay_last_opened_market")
            if selected_from_overpay_queue != last_opened_overpay_market:
                popup_row_df = overpay_queue[
                    overpay_queue["Market / Xdock"].astype(str).eq(selected_from_overpay_queue)
                ]
                if not popup_row_df.empty:
                    st.session_state["overpay_last_opened_market"] = selected_from_overpay_queue
                    open_recommendation_popup(
                        "Overpay queue drilldown",
                        popup_row_df.iloc[0],
                        rca_baselines,
                        raw_df=filtered,
                    )

        # Buttons for row selection popup
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Open selected row details",
                key="open_selected_overpay_popup",
                disabled=not bool(selected_from_overpay_queue),
            ):
                popup_row_df = overpay_queue[overpay_queue["Market / Xdock"].astype(str).eq(selected_from_overpay_queue)]
                if not popup_row_df.empty:
                    open_recommendation_popup("Overpay queue drilldown", popup_row_df.iloc[0], rca_baselines, raw_df=filtered)
        with col2:
            if st.button("Deselect row", key="deselect_overpay_popup", disabled=not bool(selected_from_overpay_queue)):
                st.session_state["overpay_queue_table"] = {"selection": {"rows": []}}
                st.session_state["overpay_selected_market"] = None
                st.session_state["overpay_last_opened_market"] = None
                st.rerun()

        # sync row-click selection to picker for next rerun
        if overpay_market_options and selected_from_overpay_queue and selected_from_overpay_queue in overpay_market_options:
            if st.session_state.get("overpay_picker") != selected_from_overpay_queue:
                st.session_state["overpay_picker_pending"] = selected_from_overpay_queue
    else:
        st.info("No overpay candidates with the current filters.")

    strict_overpay_queue = business_view[
        business_view["Signal / Classification"].eq("Overpay candidate")
        & business_view["Actual CPS"].notna()
        & business_view["Expected CPS"].notna()
        & business_view["Normalized Cost"].notna()
        & business_view["Cleansheet Aggressive"].notna()
        & (business_view["Actual CPS"] > business_view["Expected CPS"])
        & (business_view["Actual CPS"] > business_view["Normalized Cost"])
        & (business_view["Actual CPS"] > business_view["Cleansheet Aggressive"])
    ].copy()

    st.markdown("### Strict overpay queue (confirmed by all 3 reference costs)")
    st.caption(
        "Rules: Overpay candidate and Actual CPS > Expected CPS (model reference), Actual CPS > Normalized Cost (peer-normalized reference), and Actual CPS > Cleansheet Aggressive (cleansheet reference)."
    )
    if len(strict_overpay_queue):
        strict_overpay_queue["Top Root Cause"] = strict_overpay_queue.apply(lambda r: top_root_cause_label(r, rca_baselines), axis=1)
        strict_overpay_queue["Recommendation 1"] = strict_overpay_queue.apply(lambda r: top_recommendation(r, rca_baselines), axis=1)
        strict_overpay_queue["Recommendation 2"] = strict_overpay_queue.apply(lambda r: second_recommendation(r, rca_baselines), axis=1)
        strict_overpay_queue["Action 1 Impact"] = strict_overpay_queue.apply(lambda r: top_action_impact(r, rca_baselines), axis=1)
        strict_overpay_queue["Action 1 Confidence"] = strict_overpay_queue.apply(lambda r: top_action_confidence(r, rca_baselines), axis=1)
        strict_overpay_queue["Action 2 Impact"] = strict_overpay_queue.apply(lambda r: second_action_impact(r, rca_baselines), axis=1)
        strict_overpay_queue["Action 2 Confidence"] = strict_overpay_queue.apply(lambda r: second_action_confidence(r, rca_baselines), axis=1)

        strict_overpay_queue = strict_overpay_queue.sort_values(["CPS Gap vs Expected %", "Records With CPS"], ascending=[False, False])
        strict_cols = existing_cols(
            strict_overpay_queue,
            [
                "Market / Xdock",
                "Confidence",
                "CPS Gap vs Expected %",
                "Actual CPS",
                "Expected CPS",
                "Normalized Cost",
                "Cleansheet Aggressive",
                "Records With CPS",
                "Top Root Cause",
                "Recommendation 1",
                "Action 1 Impact",
                "Action 1 Confidence",
                "Recommendation 2",
                "Action 2 Impact",
                "Action 2 Confidence",
            ],
        )
        strict_show = strict_overpay_queue[strict_cols]
        strict_show = filter_table_rows(strict_show, combined_search_query).head(200)
        st.caption(f"Showing {len(strict_show)} of {len(strict_overpay_queue)} rows")

        strict_market_options = strict_overpay_queue["Market / Xdock"].astype(str).tolist()
        if strict_market_options:
            pending_strict_picker = st.session_state.pop("strict_overpay_picker_pending", None)
            if pending_strict_picker in strict_market_options:
                st.session_state["strict_overpay_picker"] = pending_strict_picker
            if "strict_overpay_picker" not in st.session_state:
                st.session_state["strict_overpay_picker"] = strict_market_options[0]
            if st.session_state.get("strict_overpay_picker") not in strict_market_options:
                st.session_state["strict_overpay_picker"] = strict_market_options[0]

            selected_strict_market = st.selectbox(
                "Drilldown market from strict queue",
                strict_market_options,
                key="strict_overpay_picker",
            )

            if st.button("Open strict drilldown popup", key="open_strict_popup_btn"):
                selected_row_df = strict_overpay_queue[
                    strict_overpay_queue["Market / Xdock"].astype(str).eq(selected_strict_market)
                ]
                if not selected_row_df.empty:
                    open_recommendation_popup("Strict overpay drilldown", selected_row_df.iloc[0], rca_baselines, raw_df=filtered)

        selected_from_queue = None
        try:
            queue_event = st.dataframe(
                strict_show,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="strict_overpay_queue_table",
            )
            selected_rows = selected_rows_from_event(queue_event)
            if selected_rows:
                row_idx = selected_rows[0]
                if 0 <= row_idx < len(strict_show):
                    selected_from_queue = str(strict_show.iloc[row_idx]["Market / Xdock"])
                    st.session_state["strict_selected_market"] = selected_from_queue
        except TypeError:
            st.dataframe(
                strict_show,
                use_container_width=True,
                hide_index=True,
            )

        if selected_from_queue is None:
            persisted_strict_market = st.session_state.get("strict_selected_market")
            if (
                persisted_strict_market is not None
                and "Market / Xdock" in strict_show.columns
                and persisted_strict_market in set(strict_show["Market / Xdock"].astype(str))
            ):
                selected_from_queue = persisted_strict_market

        # Auto-open popup when a new row is selected in the table.
        if selected_from_queue:
            last_opened_strict_market = st.session_state.get("strict_last_opened_market")
            if selected_from_queue != last_opened_strict_market:
                popup_row_df = strict_overpay_queue[
                    strict_overpay_queue["Market / Xdock"].astype(str).eq(selected_from_queue)
                ]
                if not popup_row_df.empty:
                    st.session_state["strict_last_opened_market"] = selected_from_queue
                    open_recommendation_popup(
                        "Strict overpay drilldown",
                        popup_row_df.iloc[0],
                        rca_baselines,
                        raw_df=filtered,
                    )

        # Buttons for row selection popup
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Open selected row details",
                key="open_selected_strict_popup",
                disabled=not bool(selected_from_queue),
            ):
                popup_row_df = strict_overpay_queue[strict_overpay_queue["Market / Xdock"].astype(str).eq(selected_from_queue)]
                if not popup_row_df.empty:
                    open_recommendation_popup("Strict overpay drilldown", popup_row_df.iloc[0], rca_baselines, raw_df=filtered)
        with col2:
            if st.button("Deselect row", key="deselect_strict_popup", disabled=not bool(selected_from_queue)):
                st.session_state["strict_overpay_queue_table"] = {"selection": {"rows": []}}
                st.session_state["strict_selected_market"] = None
                st.session_state["strict_last_opened_market"] = None
                st.rerun()

        # sync row-click selection to picker for next rerun
        if strict_market_options and selected_from_queue and selected_from_queue in strict_market_options:
            if st.session_state.get("strict_overpay_picker") != selected_from_queue:
                st.session_state["strict_overpay_picker_pending"] = selected_from_queue
    else:
        st.info("No xdock meets the strict all-3-reference-cost condition with the current filters.")

with tab01:
    st.subheader("1. Signal summary and drilldown")

    classification_counts = (
        business_view["Signal / Classification"]
        .value_counts()
        .rename_axis("Signal / Classification")
        .reset_index(name="Market Groups")
    )

    fig_counts = px.bar(
        classification_counts,
        x="Signal / Classification",
        y="Market Groups",
        color="Signal / Classification",
        text="Market Groups",
        color_discrete_map=CLASS_COLORS,
        title="Market Classification Summary",
    )
    fig_counts.update_layout(
        xaxis_title="Signal / Classification",
        yaxis_title="Market Groups",
        showlegend=False,
    )

    class_options = classification_counts["Signal / Classification"].tolist()
    def_class = "Overpay candidate" if "Overpay candidate" in class_options else class_options[0]

    selected_class = None
    selected_from_chart = False
    try:
        event = st.plotly_chart(
            fig_counts,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="classification_bar",
        )
        pts = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
        if pts:
            selected_class = pts[0].get("x")
            selected_from_chart = True
    except TypeError:
        st.plotly_chart(fig_counts, use_container_width=True)

    if "signal_class_picker" not in st.session_state:
        st.session_state["signal_class_picker"] = def_class
    if selected_from_chart and selected_class in class_options:
        st.session_state["signal_class_picker"] = selected_class
        st.session_state["signal_chart_active"] = True
    elif st.session_state.get("signal_chart_active", False):
        # If the chart selection is cleared (blank click / Esc), fall back to default class.
        st.session_state["signal_class_picker"] = def_class
        st.session_state["signal_chart_active"] = False

    selected_class = st.selectbox(
        "Select a signal to inspect",
        class_options,
        index=class_options.index(st.session_state["signal_class_picker"]),
        key="signal_class_picker",
    )
    if selected_from_chart:
        st.success(f"Selected from chart: {selected_class}")

    detail = business_view[business_view["Signal / Classification"].eq(selected_class)].copy()
    if "CPS Gap vs Expected %" in detail.columns:
        detail = detail.sort_values("CPS Gap vs Expected %", ascending=False)

    st.caption(
        f"{selected_class}: {len(detail):,} groups. Review gap %, confidence, and cleansheet values for quick triage."
    )

    key_cols = [
        "Market / Xdock",
        "Signal / Classification",
        "Confidence",
        "Classification Reason",
        "Confidence Reason",
        "Records With CPS",
        "Actual CPS",
        "Expected CPS",
        "CPS Gap vs Expected %",
        "Normalized Cost",
        "Cleansheet Conservative",
        "Cleansheet Aggressive",
    ]
    show_cols = existing_cols(detail, key_cols)
    detail_show = detail[show_cols].copy()

    def _fmt_cell(col: str, val) -> str:
        if col in {"Actual CPS", "Expected CPS", "Normalized Cost", "Cleansheet Conservative", "Cleansheet Aggressive"}:
            return money(val)
        if col == "CPS Gap vs Expected %":
            return pct(val)
        if col == "Records With CPS":
            try:
                return f"{int(val):,}"
            except Exception:
                return "n/a"
        if pd.isna(val):
            return "n/a"
        return str(val)

    headers_html = "".join(f"<th>{html.escape(c)}</th>" for c in show_cols)
    rows_html = []
    for _, r in detail_show.iterrows():
        row_cells = []
        for c in show_cols:
            cell_val = _fmt_cell(c, r.get(c))
            cell_link = f"<button type='button' class='row-link'>{html.escape(cell_val)}</button>"
            if c == "CPS Gap vs Expected %":
                gv = r.get(c)
                if pd.notna(gv) and gv >= 0.10:
                    cls = "gap-pos"
                elif pd.notna(gv) and gv <= -0.10:
                    cls = "gap-neg"
                else:
                    cls = "gap-mid"
                row_cells.append(f"<td class='{cls}'>{cell_link}</td>")
            else:
                row_cells.append(f"<td>{cell_link}</td>")
        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")

    table_html = (
        "<div class='drill-wrap'><table class='drill-table'>"
        f"<thead><tr>{headers_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    st.download_button(
        "Download selected details",
        detail_show.to_csv(index=False).encode("utf-8"),
        file_name=f"market_details_{selected_class.replace(' ', '_').replace('/', '_')}.csv",
        mime="text/csv",
    )

with tab1:
    plot_df = business_view[business_view["Actual CPS"].notna() & business_view["Expected CPS"].notna()].copy()
    if len(plot_df):
        fig = px.scatter(
            plot_df,
            x="Expected CPS",
            y="Actual CPS",
            color="Signal / Classification",
            size="Records With CPS",
            hover_data=hover_cols,
            color_discrete_map=CLASS_COLORS,
            title="Actual CPS vs Model-Expected CPS",
        )
        max_val = np.nanmax([plot_df["Actual CPS"].max(), plot_df["Expected CPS"].max()])
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Actual = Expected", line=dict(color="black", dash="dash")))
        fig.update_layout(xaxis_title="Expected CPS", yaxis_title="Actual median CPS")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dots above the diagonal cost more than expected after controlling for market profile. Larger bubbles have more records with CPS.")

with tab2:
    top_overpay = business_view[business_view["Signal / Classification"].eq("Overpay candidate")].copy()
    top_overpay = top_overpay.sort_values("CPS Gap vs Expected %", ascending=False).head(30)
    if len(top_overpay):
        top_overpay["Label"] = make_label(top_overpay)
        fig = px.bar(
            top_overpay.sort_values("CPS Gap vs Expected %"),
            x="CPS Gap vs Expected %",
            y="Label",
            color="Signal / Classification",
            orientation="h",
            hover_data=hover_cols,
            color_discrete_map=CLASS_COLORS,
            title="Top overpay candidates by gap vs expected CPS",
        )
        fig.update_layout(height=760, xaxis_title="Actual CPS / Expected CPS - 1", yaxis_title="Market group")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Use this as the primary sourcing review queue. Strong candidates are above the high expected band and pass confidence checks.")
    else:
        st.info("No overpay candidates with the current filters.")

with tab3:
    band_sample = business_view[business_view["CPS Gap vs Expected %"].notna()].sort_values("CPS Gap vs Expected %", ascending=False).head(30).copy()
    if len(band_sample):
        band_sample["Label"] = make_label(band_sample)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=band_sample["Label"], y=band_sample["Actual CPS"], name="Actual CPS"))
        fig.add_trace(go.Scatter(x=band_sample["Label"], y=band_sample["Expected CPS"], mode="lines+markers", name="Expected P50"))
        fig.add_trace(go.Scatter(x=band_sample["Label"], y=band_sample["Expected High CPS P90"], mode="lines+markers", name="Expected P90"))
        fig.add_trace(go.Scatter(x=band_sample["Label"], y=band_sample["Expected Low CPS P10"], mode="lines+markers", name="Expected P10"))
        fig.update_layout(title="Actual CPS vs expected cost range", xaxis_title="Market group", yaxis_title="Cost per stop", xaxis_tickangle=-45, height=760, barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("If actual CPS is above P90, the market is above a reasonable high-cost outcome for comparable profiles.")

with tab4:

    clean = business_view[
        business_view["Gap vs Aggressive Cleansheet %"].notna()
        & business_view["CPS Gap vs Expected %"].notna()
    ].copy()

    if len(clean):

        # -----------------------------------------
        # Dynamic clipping
        # -----------------------------------------

        x_low = clean["Gap vs Aggressive Cleansheet %"].quantile(0.01)
        x_high = clean["Gap vs Aggressive Cleansheet %"].quantile(0.99)

        y_low = clean["CPS Gap vs Expected %"].quantile(0.01)
        y_high = clean["CPS Gap vs Expected %"].quantile(0.99)

        clean_plot = clean.copy()

        # Preserve actual values
        clean_plot["Actual Cleansheet Gap"] = (
            clean["Gap vs Aggressive Cleansheet %"]
        )

        clean_plot["Actual Model Gap"] = (
            clean["CPS Gap vs Expected %"]
        )

        # Visualization clipping only
        clean_plot["Gap vs Aggressive Cleansheet %"] = (
            clean_plot["Gap vs Aggressive Cleansheet %"]
            .clip(x_low, x_high)
        )

        clean_plot["CPS Gap vs Expected %"] = (
            clean_plot["CPS Gap vs Expected %"]
            .clip(y_low, y_high)
        )

        fig = px.scatter(
            clean_plot,
            x="Gap vs Aggressive Cleansheet %",
            y="CPS Gap vs Expected %",
            color="Signal / Classification",
            size="Records With CPS",
            hover_data={
                "Market / Xdock": True,
                "Signal / Classification": True,

                "Actual CPS": ":,.2f",
                "Expected CPS": ":,.2f",

                "Cleansheet Aggressive": ":,.2f",

                "Records With CPS": ":,.0f",

                # hide noisy fields
                "Confidence": False,
                "Business Note": False,
                "Market Name": False,

                "Gap vs Aggressive Cleansheet %": False,
                "CPS Gap vs Expected %": False,

                "Total Records": False,
                "Normalization Sensitivity": False,
                "Expected Low CPS P10": False,
                "Expected High CPS P90": False,
                "Normalized Cost": False,
                "Cleansheet Conservative": False,
            },
            color_discrete_map=CLASS_COLORS,
            title="Model Gap vs Aggressive Cleansheet Gap",
        )

        # -----------------------------------------
        # Label biggest markets
        # -----------------------------------------

        top_labels = (
            clean_plot
            .sort_values(
                "Records With CPS",
                ascending=False
            )
            .head(15)
        )

        fig.add_trace(
            go.Scatter(
                x=top_labels[
                    "Gap vs Aggressive Cleansheet %"
                ],
                y=top_labels[
                    "CPS Gap vs Expected %"
                ],
                text=top_labels[
                    "Market / Xdock"
                ],
                mode="text",
                textposition="top center",
                showlegend=False,
            )
        )

        # -----------------------------------------
        # Quadrants
        # -----------------------------------------

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black"
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black"
        )

        fig.update_xaxes(
            range=[x_low, x_high],
            tickformat=".0%",
            title="Gap vs Aggressive Cleansheet (%)"
        )

        fig.update_yaxes(
            range=[y_low, y_high],
            tickformat=".0%",
            title="Gap vs Expected CPS (%)"
        )

        fig.update_traces(
            marker=dict(
                sizemin=8,
                opacity=0.85,
                line=dict(
                    width=1,
                    color="white"
                ),
            )
        )

        fig.update_layout(
            height=850,
            legend_title="Classification"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------------------
        # Opportunity table
        # -----------------------------------------

        st.markdown("### Top Quadrant Opportunities")

        top_quad = (
            clean
            .sort_values(
                [
                    "CPS Gap vs Expected %",
                    "Records With CPS"
                ],
                ascending=[False, False]
            )
            .head(25)
        )

        st.dataframe(
            top_quad[
                [
                    "Market / Xdock",
                    "Signal / Classification",
                    "Actual CPS",
                    "Expected CPS",
                    "CPS Gap vs Expected %",
                    "Gap vs Aggressive Cleansheet %",
                    "Records With CPS",
                    "Confidence",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            """
Upper Right

• Model and Cleansheet both indicate potential overpayment.

Upper Left

• Model indicates higher-than-expected cost but Cleansheet does not.

Lower Left

• Model and Cleansheet both support below-benchmark cost.

Lower Right

• Cleansheet indicates higher-than-benchmark cost but the model does not.

The chart is scaled using the 1st and 99th percentiles to remove excess white space while preserving the overall market distribution.
            """
        )

    else:

        st.info(
            "No aggressive cleansheet values available for the current view."
        )

with tab5:

    st.subheader("Market Cost Comparison")

    comparison_markets = sorted(
        business_view["Market / Xdock"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_market = st.selectbox(
        "Select Market / Xdock",
        comparison_markets,
        key="cost_comparison_market"
    )

    market_details = (
        business_view[
            business_view["Market / Xdock"]
            .astype(str)
            .eq(selected_market)
        ]
        .copy()
    )

    if len(market_details):

        row = market_details.iloc[0]

        # --------------------------------------------------
        # Comparison Chart
        # --------------------------------------------------

        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Actual CPS",
                    "Expected CPS",
                    "Normalized Cost",
                    "Conservative Cleansheet",
                    "Aggressive Cleansheet",
                ],
                "Value": [
                    row.get("Actual CPS", np.nan),
                    row.get("Expected CPS", np.nan),
                    row.get("Normalized Cost", np.nan),
                    row.get("Cleansheet Conservative", np.nan),
                    row.get("Cleansheet Aggressive", np.nan),
                ],
            }
        ).dropna()

        if len(comparison_df):

            comparison_df["Color"] = np.where(
                comparison_df["Metric"] == "Actual CPS",
                "#C8102E",
                "#00447C"
            )

            fig = px.bar(
                comparison_df,
                x="Metric",
                y="Value",
                color="Color",
                color_discrete_map="identity",
                text="Value",
                title=f"Cost Comparison - {selected_market}"
            )

            fig.update_traces(
                texttemplate="$%{y:,.2f}",
                textposition="outside"
            )

            expected_cps = row.get("Expected CPS", np.nan)

            if pd.notna(expected_cps):

                fig.add_hline(
                    y=expected_cps,
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"Expected CPS (${expected_cps:,.2f})"
                )

            fig.update_layout(
                height=600,
                showlegend=False,
                yaxis_title="Cost Per Stop ($)",
                xaxis_title="",
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No comparison values available for this market."
            )

        # --------------------------

with tab6:

    st.subheader("Triangulation V1")

    tri = business_view[
        business_view["Normalized Cost"].notna()
        & business_view["CPS Gap vs Expected %"].notna()
    ].copy()

    if len(tri):

        tri["Normalized Gap %"] = (
            tri["Actual CPS"]
            / tri["Normalized Cost"]
        ) - 1

        fig = px.scatter(
            tri,
            x="Normalized Gap %",
            y="CPS Gap vs Expected %",
            color="Signal / Classification",
            size="Records With CPS",
            hover_data=[
                "Market / Xdock",
                "Actual CPS",
                "Expected CPS",
                "Normalized Cost",
                "Records With CPS",
            ],
            color_discrete_map=CLASS_COLORS,
            title="Expected CPS Gap vs Normalized Cost Gap"
        )

        fig.add_hline(y=0, line_dash="dash")
        fig.add_vline(x=0, line_dash="dash")

        fig.update_layout(height=800)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info("No normalized cost values available.")

with tab7:

    st.subheader("Triangulation V2")

    tri = business_view.copy()

    tri = tri[
        tri["Normalized Cost"].notna()
        & tri["Gap vs Aggressive Cleansheet %"].notna()
        & tri["CPS Gap vs Expected %"].notna()
    ].copy()

    if len(tri):

        tri["Normalized Gap %"] = (
            tri["Actual CPS"]
            / tri["Normalized Cost"]
        ) - 1

        tri["Validation Score"] = (
            0.50 * tri["CPS Gap vs Expected %"]
            + 0.30 * tri["Gap vs Aggressive Cleansheet %"]
            + 0.20 * tri["Normalized Gap %"]
        )

        fig = px.scatter(
            tri,
            x="Validation Score",
            y="Actual CPS",
            size="Records With CPS",
            color="Signal / Classification",
            hover_data=[
                "Market / Xdock",
                "Actual CPS",
                "Expected CPS",
                "Normalized Cost",
                "Cleansheet Aggressive",
                "Records With CPS",
            ],
            color_discrete_map=CLASS_COLORS,
            title="Combined Validation Score"
        )

        fig.update_layout(height=800)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            """
Higher Validation Score indicates stronger agreement
between Expected CPS, Normalized Cost, and Cleansheet.
            """
        )

    else:

        st.info("Not enough data for validation score.")

with tab8:

    st.subheader("Triangulation V3")

    tri = business_view.copy()

    tri = tri[
        tri["Normalized Cost"].notna()
        & tri["Gap vs Aggressive Cleansheet %"].notna()
        & tri["CPS Gap vs Expected %"].notna()
    ].copy()

    if len(tri):

        tri["Normalized Gap %"] = (
            tri["Actual CPS"]
            / tri["Normalized Cost"]
        ) - 1

        fig = px.scatter(
            tri,
            x="Gap vs Aggressive Cleansheet %",
            y="CPS Gap vs Expected %",
            color="Normalized Gap %",
            size="Records With CPS",
            hover_data=[
                "Market / Xdock",
                "Actual CPS",
                "Expected CPS",
                "Normalized Cost",
                "Cleansheet Aggressive",
                "Records With CPS",
            ],
            color_continuous_scale="RdYlGn_r",
            title="Expected CPS + Cleansheet + Normalized Cost"
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="black"
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black"
        )

        fig.update_layout(
            height=850
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            """
X-axis = Gap vs Aggressive Cleansheet

Y-axis = Gap vs Expected CPS

Color = Gap vs Normalized Cost

Bubble Size = Records With CPS

Red bubbles indicate all three measures support a higher-cost signal.
            """
        )

    else:

        st.info(
            "No triangulation data available."
        )

with tab9:

    st.subheader("Signal Buckets: Model + Normalized Cost + Cleansheet")

    st.caption(
        """
This view groups markets into 8 possible signal combinations.

Each market is evaluated using three binary signals:

1. Actual CPS vs Expected CPS
2. Actual CPS vs Normalized Cost
3. Actual CPS vs Aggressive Cleansheet

This creates 2³ = 8 possible buckets.
        """
    )

    tri = business_view.copy()

    required_cols = [
        "Market / Xdock",
        "Actual CPS",
        "Expected CPS",
        "Normalized Cost",
        "Cleansheet Aggressive",
        "CPS Gap vs Expected %",
        "Records With CPS",
    ]

    missing_cols = [
        c for c in required_cols
        if c not in tri.columns
    ]

    if missing_cols:

        st.warning(
            f"Missing required columns for signal bucket view: {missing_cols}"
        )

    else:

        tri = tri[
            tri["Actual CPS"].notna()
            & tri["Expected CPS"].notna()
            & tri["Normalized Cost"].notna()
            & tri["Cleansheet Aggressive"].notna()
            & tri["CPS Gap vs Expected %"].notna()
            & tri["Records With CPS"].notna()
        ].copy()

        if len(tri) == 0:

            st.info(
                "Not enough complete data to build signal buckets. This view needs Actual CPS, Expected CPS, Normalized Cost, Aggressive Cleansheet, and Records With CPS."
            )

        else:

            # --------------------------------------------------
            # Build the three validation signals
            # --------------------------------------------------

            tri["Normalized Gap %"] = (
                tri["Actual CPS"] / tri["Normalized Cost"]
            ) - 1

            tri["Aggressive Cleansheet Gap %"] = (
                tri["Actual CPS"] / tri["Cleansheet Aggressive"]
            ) - 1

            tri["Model Signal"] = np.where(
                tri["CPS Gap vs Expected %"] > 0,
                "Above Expected",
                "Below Expected",
            )

            tri["Normalized Signal"] = np.where(
                tri["Normalized Gap %"] > 0,
                "Above Normalized",
                "Below Normalized",
            )

            tri["Cleansheet Signal"] = np.where(
                tri["Aggressive Cleansheet Gap %"] > 0,
                "Above Cleansheet",
                "Below Cleansheet",
            )

            tri["Signal Bucket"] = (
                tri["Model Signal"]
                + " | "
                + tri["Normalized Signal"]
                + " | "
                + tri["Cleansheet Signal"]
            )

            # --------------------------------------------------
            # Business-friendly bucket names
            # --------------------------------------------------

            bucket_map = {
                "Above Expected | Above Normalized | Above Cleansheet":
                    "All 3 Above: Strong Overpay Support",

                "Above Expected | Above Normalized | Below Cleansheet":
                    "Model + Normalized Above, Cleansheet Below",

                "Above Expected | Below Normalized | Above Cleansheet":
                    "Model + Cleansheet Above, Normalized Below",

                "Above Expected | Below Normalized | Below Cleansheet":
                    "Model Above Only",

                "Below Expected | Above Normalized | Above Cleansheet":
                    "Normalized + Cleansheet Above, Model Below",

                "Below Expected | Above Normalized | Below Cleansheet":
                    "Normalized Above Only",

                "Below Expected | Below Normalized | Above Cleansheet":
                    "Cleansheet Above Only",

                "Below Expected | Below Normalized | Below Cleansheet":
                    "All 3 Below: Strong Underpay / Low-Cost Support",
            }

            tri["Bucket Label"] = tri["Signal Bucket"].map(bucket_map)
            tri["Bucket Label"] = tri["Bucket Label"].fillna(tri["Signal Bucket"])

            # --------------------------------------------------
            # Score for ranking inside each bucket
            # --------------------------------------------------

            tri["Three-Signal Score"] = (
                0.40 * tri["CPS Gap vs Expected %"]
                + 0.30 * tri["Normalized Gap %"]
                + 0.30 * tri["Aggressive Cleansheet Gap %"]
            )

            tri["Volume Adjusted Score"] = (
                tri["Three-Signal Score"]
                * np.log1p(tri["Records With CPS"])
            )

            # --------------------------------------------------
            # Bucket summary
            # --------------------------------------------------

            bucket_summary = (
                tri
                .groupby(
                    [
                        "Bucket Label",
                        "Model Signal",
                        "Normalized Signal",
                        "Cleansheet Signal",
                    ],
                    dropna=False
                )
                .agg(
                    Markets=("Market / Xdock", "count"),
                    Records_With_CPS=("Records With CPS", "sum"),
                    Median_Model_Gap=("CPS Gap vs Expected %", "median"),
                    Median_Normalized_Gap=("Normalized Gap %", "median"),
                    Median_Cleansheet_Gap=("Aggressive Cleansheet Gap %", "median"),
                    Median_Three_Signal_Score=("Three-Signal Score", "median"),
                )
                .reset_index()
                .sort_values(
                    "Records_With_CPS",
                    ascending=False
                )
            )

            # --------------------------------------------------
            # Treemap: 8 signal buckets
            # --------------------------------------------------

            fig = px.treemap(
                bucket_summary,
                path=["Bucket Label"],
                values="Records_With_CPS",
                color="Median_Three_Signal_Score",
                color_continuous_scale="RdYlGn_r",
                hover_data={
                    "Markets": True,
                    "Records_With_CPS": ":,.0f",
                    "Median_Model_Gap": ":.1%",
                    "Median_Normalized_Gap": ":.1%",
                    "Median_Cleansheet_Gap": ":.1%",
                    "Median_Three_Signal_Score": ":.3f",
                },
                title="8 Signal Buckets: Model + Normalized Cost + Cleansheet",
            )

            fig.update_layout(
                height=750,
                margin=dict(
                    l=10,
                    r=10,
                    t=70,
                    b=10,
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info(
                """
Interpretation:

- **All 3 Above** = strongest overpay support.
- **All 3 Below** = strongest underpay or low-cost support.
- Mixed buckets indicate disagreement between the model, normalized cost, and cleansheet.
- Bucket size is based on Records With CPS, so larger boxes represent more operational evidence.
                """
            )

            # --------------------------------------------------
            # Bucket selector and detail table
            # --------------------------------------------------

            st.markdown("### Markets in Selected Bucket")

            bucket_options = (
                bucket_summary["Bucket Label"]
                .dropna()
                .astype(str)
                .tolist()
            )

            if len(bucket_options) == 0:

                st.info("No signal buckets available.")

            else:

                default_bucket = (
                    "All 3 Above: Strong Overpay Support"
                    if "All 3 Above: Strong Overpay Support" in bucket_options
                    else bucket_options[0]
                )

                selected_bucket = st.selectbox(
                    "Select bucket",
                    bucket_options,
                    index=bucket_options.index(default_bucket),
                    key="signal_bucket_selector",
                )

                bucket_detail = (
                    tri[
                        tri["Bucket Label"].eq(selected_bucket)
                    ]
                    .sort_values(
                        "Volume Adjusted Score",
                        ascending=False
                    )
                    .copy()
                )

                detail_cols = [
                    "Market / Xdock",
                    "Signal / Classification",
                    "Confidence",
                    "Actual CPS",
                    "Expected CPS",
                    "Normalized Cost",
                    "Cleansheet Aggressive",
                    "CPS Gap vs Expected %",
                    "Normalized Gap %",
                    "Aggressive Cleansheet Gap %",
                    "Records With CPS",
                    "Three-Signal Score",
                    "Volume Adjusted Score",
                    "Business Note",
                ]

                detail_cols = [
                    c for c in detail_cols
                    if c in bucket_detail.columns
                ]

                st.dataframe(
                    bucket_detail[detail_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Actual CPS": st.column_config.NumberColumn(format="$%.2f"),
                        "Expected CPS": st.column_config.NumberColumn(format="$%.2f"),
                        "Normalized Cost": st.column_config.NumberColumn(format="$%.2f"),
                        "Cleansheet Aggressive": st.column_config.NumberColumn(format="$%.2f"),
                        "CPS Gap vs Expected %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Normalized Gap %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Aggressive Cleansheet Gap %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Records With CPS": st.column_config.NumberColumn(format="%,d"),
                        "Three-Signal Score": st.column_config.NumberColumn(format="%.3f"),
                        "Volume Adjusted Score": st.column_config.NumberColumn(format="%.3f"),
                    },
                )

                st.download_button(
                    "Download selected bucket",
                    bucket_detail[detail_cols].to_csv(index=False).encode("utf-8"),
                    file_name=f"signal_bucket_{selected_bucket.replace(' ', '_').replace('/', '_').replace('|', '_')}.csv",
                    mime="text/csv",
                )


with tab10:

    st.subheader("Customer type layer")

    if "CUST_BUS_TYP_DSCR" not in filtered.columns:
        st.info("Customer type column (CUST_BUS_TYP_DSCR) is not present in this dataset.")
    else:
        cust_df = filtered.copy()
        cust_df["CUST_BUS_TYP_DSCR"] = cust_df["CUST_BUS_TYP_DSCR"].astype("string").fillna("Unknown")

        paid_rows = cust_df[cust_df[TARGET].notna() & (cust_df[TARGET] > 0)].copy()
        if paid_rows.empty:
            st.info("No paid rows available for customer-type analysis under current filters.")
        else:
            type_summary = (
                paid_rows.groupby("CUST_BUS_TYP_DSCR", dropna=False)
                .agg(
                    Records=(TARGET, "size"),
                    Median_CPS=(TARGET, "median"),
                    Mean_CPS=(TARGET, "mean"),
                    Total_Cost=(TARGET, "sum"),
                )
                .reset_index()
                .sort_values("Records", ascending=False)
            )

            total_records = max(int(type_summary["Records"].sum()), 1)
            total_cost = type_summary["Total_Cost"].sum()
            type_summary["Records_%"] = type_summary["Records"] / total_records
            type_summary["Cost_%"] = np.where(total_cost > 0, type_summary["Total_Cost"] / total_cost, np.nan)

            c1, c2, c3 = st.columns(3)
            c1.metric("Customer types", f"{type_summary['CUST_BUS_TYP_DSCR'].nunique():,}")
            c2.metric("Paid records", f"{total_records:,}")
            c3.metric("Total paid cost", money(total_cost))

            fig_mix = px.bar(
                type_summary,
                x="CUST_BUS_TYP_DSCR",
                y="Records",
                color="Median_CPS",
                color_continuous_scale="Blues",
                title="Customer type mix by paid records (color = median CPS)",
            )
            fig_mix.update_layout(height=520, xaxis_title="Customer Type", yaxis_title="Paid Records")
            st.plotly_chart(fig_mix, use_container_width=True)

            by_market_type = (
                paid_rows.groupby(["XDOCK", "CUST_BUS_TYP_DSCR"], dropna=False)
                .agg(
                    Records=(TARGET, "size"),
                    Actual_CPS=(TARGET, "median"),
                )
                .reset_index()
            )

            expected_by_type = (
                by_market_type.groupby("CUST_BUS_TYP_DSCR", dropna=False)["Actual_CPS"]
                .median()
                .rename("Expected_CPS_By_Type")
                .reset_index()
            )

            by_market_type = by_market_type.merge(expected_by_type, on="CUST_BUS_TYP_DSCR", how="left")
            by_market_type["Gap_vs_Type_Expected_%"] = safe_div(
                by_market_type["Actual_CPS"], by_market_type["Expected_CPS_By_Type"]
            ) - 1

            top_types = (
                by_market_type.groupby("CUST_BUS_TYP_DSCR", dropna=False)["Records"].sum()
                .sort_values(ascending=False)
                .head(8)
                .index
            )
            heat = by_market_type[by_market_type["CUST_BUS_TYP_DSCR"].isin(top_types)].copy()
            if not heat.empty:
                fig_heat = px.density_heatmap(
                    heat,
                    x="CUST_BUS_TYP_DSCR",
                    y="XDOCK",
                    z="Gap_vs_Type_Expected_%",
                    histfunc="avg",
                    color_continuous_scale="RdYlGn_r",
                    title="Market vs customer type gap to type baseline (median CPS)",
                )
                fig_heat.update_layout(height=560, xaxis_title="Customer Type", yaxis_title="Market / Xdock")
                st.plotly_chart(fig_heat, use_container_width=True)

            detail = by_market_type.sort_values("Records", ascending=False).copy()
            st.dataframe(
                detail[[
                    "XDOCK",
                    "CUST_BUS_TYP_DSCR",
                    "Records",
                    "Actual_CPS",
                    "Expected_CPS_By_Type",
                    "Gap_vs_Type_Expected_%",
                ]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Actual_CPS": st.column_config.NumberColumn(format="$%.2f"),
                    "Expected_CPS_By_Type": st.column_config.NumberColumn(format="$%.2f"),
                    "Gap_vs_Type_Expected_%": st.column_config.NumberColumn(format="%.1f%%"),
                    "Records": st.column_config.NumberColumn(format="%,d"),
                },
            )


# --------------------------------------------------------------------------- #
# Methodology and model diagnostics
# --------------------------------------------------------------------------- #
with st.expander("Methodology and model diagnostics"):
    st.markdown(
        """
        **Approach**
        - Aggregate shipment-level rows to market/xdock grain (`XDOCK`) before modeling.
        - Use positive paid rows to calculate market-level actual median CPS.
        - Fit a quantile gradient boosting model to estimate expected CPS bands.
        - P50 is the expected median CPS. P10/P90 define the low/high expected range.
                - Classification is based on actual CPS versus expected CPS, with confidence checks for volume and combined sensitivity.
                - Combined sensitivity = average of normalization sensitivity and cleansheet sensitivity.
                - Confidence points (0-3):
                    - +1 if paid records >= minimum threshold
                    - +1 if paid records >= 3x minimum threshold
                    - +1 if combined sensitivity <= 40%
                - Confidence labels: High (3), Medium (2), Low (1), Very Low (0).

        **Why this is more defensible than normalized cost alone**
        - Normalized cost is included as a supporting metric.
        - The primary signal asks whether a market is expensive relative to comparable operational profile, not simply whether its normalized cost is high.
        - Cleansheet is used as triangulation where available, not as the only source of truth.
        """
    )
    st.dataframe(pd.DataFrame([model_metrics]), use_container_width=True, hide_index=True)

# Full download.
st.download_button(
    "Download full business scorecard",
    business_view.to_csv(index=False).encode("utf-8"),
    file_name="lastmile_business_cost_scorecard.csv",
    mime="text/csv",
)
