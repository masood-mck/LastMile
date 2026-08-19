"""
Xdock Normalized-Cost Analytics
================================

Streamlit dashboard to compare crossdock (xdock) normalized cost for a chosen
year + month, and spot markets where we appear to be OVER- or UNDER-paying.

Because ``normalized_cost`` already adjusts for geography, density and shipment
mix, every market is comparable to a single baseline (the median/mean normalized
cost across markets). Markets above the baseline look like over-payment; markets
below it look like under-payment.

Runs in two environments:

  * Local          ->  reads the CSV at ``LOCAL_CSV_PATH`` (or ``NORM_DATA_PATH``).
  * Databricks     ->  reads a Unity Catalog / Hive table (``NORM_DATA_TABLE``)
                       or a DBFS / volume path (``NORM_DATA_PATH``) via Spark.

Run locally:
    streamlit run lm_cost_app.py

Run on Databricks (Databricks Apps or a driver-proxy terminal):
    export NORM_DATA_TABLE="catalog.schema.main_geog_dense_ship_mult"   # or
    export NORM_DATA_PATH="/Volumes/catalog/schema/vol/main_geog_dense_ship_mult.csv"
    streamlit run lm_cost_app.py
"""

from __future__ import annotations

import calendar
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Local default (relative to this file so it works regardless of the CWD).
# Prefer the gzipped file if present (smaller -> stays under the Apps 10 MB
# per-file limit); pandas.read_csv auto-decompresses based on the .gz suffix.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_GZ_PATH = os.path.join(_DATA_DIR, "LM_CS_slim.csv.gz")
_CSV_PATH = os.path.join(_DATA_DIR, "LM_CS_slim.csv")
LOCAL_CSV_PATH = _GZ_PATH if os.path.exists(_GZ_PATH) else _CSV_PATH
# Databricks defaults (override with env vars). Leave blank if unused.
DATABRICKS_TABLE = os.environ.get("NORM_DATA_TABLE", "")
DATABRICKS_PATH = os.environ.get("NORM_DATA_PATH", "")

# Canonical column names we rely on (with fall-backs for the duplicated cols).
XDOCK_CANDIDATES = ["XDOCK", "XDOCK_x", "XDOCK_y"]
YEAR_CANDIDATES = ["YEAR", "ROUTE_YEAR", "year"]
MONTH_CANDIDATES = ["MONTH", "ROUTE_MONTH", "month"]

st.set_page_config(page_title="Crossdock Normalized Cost", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (environment-aware)
# --------------------------------------------------------------------------- #
def _on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def _load_via_spark() -> pd.DataFrame | None:
    """Read from a Spark table or path when running on Databricks."""
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
    except Exception:
        return None

    if DATABRICKS_TABLE:
        return spark.table(DATABRICKS_TABLE).toPandas()
    if DATABRICKS_PATH:
        reader = spark.read.option("header", True).option("inferSchema", True)
        if DATABRICKS_PATH.lower().endswith(".parquet"):
            return spark.read.parquet(DATABRICKS_PATH).toPandas()
        return reader.csv(DATABRICKS_PATH).toPandas()
    return None


@st.cache_data(show_spinner="Loading normalized cost data...")
def load_data() -> pd.DataFrame:
    df: pd.DataFrame | None = None

    if _on_databricks():
        df = _load_via_spark()

    if df is None:
        # Local (or Databricks fallback to a directly-readable path).
        path = os.environ.get("NORM_DATA_PATH") or LOCAL_CSV_PATH
        df = pd.read_csv(path, low_memory=False)

    return _prepare(df)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


@st.cache_data(show_spinner="Preparing...")
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    xdock = _first_present(df, XDOCK_CANDIDATES)
    year = _first_present(df, YEAR_CANDIDATES)
    month = _first_present(df, MONTH_CANDIDATES)
    if not (xdock and year and month):
        missing = [
            name
            for name, col in [("xdock", xdock), ("year", year), ("month", month)]
            if col is None
        ]
        raise KeyError(f"Could not find required column(s): {missing}")

    df = df.rename(columns={xdock: "XDOCK", year: "YEAR", month: "MONTH"})

    # numeric coercion
    for col in [
        "YEAR",
        "MONTH",
        "cost per stop",
        "normalized_cost",
        "Cleansheet Cost Per Stop Conservative",
        "Cleansheet Cost Per Stop Aggressive",
        "GEOGRAPHY_MULTIPLIER",
        "shipment_norm_multiplier_qt",
        "miles_per_stop_norm_multiplier_qt",
        "geo_mean",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["XDOCK", "YEAR", "MONTH"])
    df["YEAR"] = df["YEAR"].astype(int)
    df["MONTH"] = df["MONTH"].astype(int)
    # fill DC: prefer the FILL_DC_CD column, else derive from the xdock prefix
    if "FILL_DC_CD" in df.columns:
        df["FILL_DC_CD"] = (
            df["FILL_DC_CD"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        )
    else:
        df["FILL_DC_CD"] = df["XDOCK"].astype(str).str.extract(r"XD_(\d+)_", expand=False)
    return df


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
AGG_FUNCS = {"Mean": "mean", "Median": "median"}

# McKesson brand palette
MCK_BLUE = "#00447C"     # primary deep blue
MCK_BRIGHT = "#0091DA"   # bright blue
MCK_GREEN = "#78BE20"    # green
MCK_RED = "#C8102E"      # red
MCK_GRAY = "#53565A"     # neutral
MCK_OLIVE = "#4A7C1F"    # darker green (aggressive cleansheet)

# Cost metrics: display label -> source column / aggregated column / colour.
METRIC_SRC = {
    "Cost per stop": "cost per stop",
    "Normalized cost": "normalized_cost",
    "Cleansheet conservative": "Cleansheet Cost Per Stop Conservative",
    "Cleansheet aggressive": "Cleansheet Cost Per Stop Aggressive",
}
METRIC_OUTCOL = {
    "Cost per stop": "cost_per_stop",
    "Normalized cost": "normalized_cost",
    "Cleansheet conservative": "cleansheet_conservative",
    "Cleansheet aggressive": "cleansheet_aggressive",
}
METRIC_COLORS = {
    "Cost per stop": MCK_BLUE,
    "Normalized cost": MCK_BRIGHT,
    "Cleansheet conservative": MCK_GREEN,
    "Cleansheet aggressive": MCK_OLIVE,
}

# Normalization multipliers (source column -> friendly label). These adjust the raw
# cost per stop into the comparable normalized_cost.
MULTIPLIER_LABELS = {
    "GEOGRAPHY_MULTIPLIER": "Geography",
    "miles_per_stop_norm_multiplier_qt": "Density (miles / stop)",
    "shipment_norm_multiplier_qt": "Shipment multiplier",
    "geo_mean": "Geo mean",
}
MULTIPLIER_COLS = list(MULTIPLIER_LABELS)


@st.cache_data(show_spinner="Aggregating...")
def by_xdock(df: pd.DataFrame, year: int, month: int, agg: str) -> pd.DataFrame:
    d = df[(df["YEAR"] == year) & (df["MONTH"] == month)]
    spec: dict = {"shipments": ("normalized_cost", "size")}
    for label, src in METRIC_SRC.items():
        if src in d.columns:
            spec[METRIC_OUTCOL[label]] = (src, agg)
    for mcol in MULTIPLIER_COLS:
        if mcol in d.columns:
            spec[mcol] = (mcol, "mean")
    out = d.groupby(["FILL_DC_CD", "XDOCK"], as_index=False).agg(**spec)
    return out.sort_values(["FILL_DC_CD", "normalized_cost"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      /* tighten the top so the page starts higher */
      .block-container { padding-top: 1.1rem; padding-bottom: 1rem; }
      header[data-testid="stHeader"] { height: 0; background: transparent; }
      /* app title */
      .app-title { font-size: 2.0rem; font-weight: 800; color: #00447C;
                   margin: 0 0 .1rem 0; letter-spacing: .2px; }
      .app-rule  { height: 3px; width: 64px; background: #78BE20;
                   border-radius: 2px; margin: 0 0 .7rem 0; }
      /* compact KPI cards */
      .kpi-row { display: flex; gap: .5rem; margin: 0 0 .7rem 0; }
      .kpi { flex: 1; background: #f7f9fb; border: 1px solid #e6ebf1;
             border-radius: 8px; padding: .4rem .7rem; }
      .kpi .lbl { font-size: .66rem; text-transform: uppercase;
                  letter-spacing: .4px; color: #7a7d82; white-space: nowrap; }
      .kpi .val { font-size: 1.1rem; font-weight: 700; color: #00447C;
                  line-height: 1.2; }
      .kpi .sub { font-size: .68rem; color: #7a7d82; white-space: nowrap;
                  overflow: hidden; text-overflow: ellipsis; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="app-title">Crossdock Cost — Market Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="app-rule"></div>', unsafe_allow_html=True)

try:
    data = load_data()
except Exception as exc:  # surface config problems clearly
    st.error(f"Could not load data: {exc}")
    st.stop()

# ---- sidebar: view toggle + filters ----
with st.sidebar:
    st.markdown("#### View")
    view = st.radio(
        "Chart view",
        ["Market comparison", "By Fill DC", "Gap analysis", "Normalization drivers"],
        label_visibility="collapsed",
    )
    st.divider()

    st.header("Filters")

    years = sorted(data["YEAR"].unique())
    sel_year = st.selectbox("Year", years, index=len(years) - 1)

    months_for_year = sorted(data.loc[data["YEAR"] == sel_year, "MONTH"].unique())
    month_labels = {m: f"{m:02d} — {calendar.month_name[m]}" for m in months_for_year}
    sel_month = st.selectbox(
        "Month",
        months_for_year,
        index=len(months_for_year) - 1,
        format_func=lambda m: month_labels[m],
    )

    agg_label = st.selectbox("Aggregation", list(AGG_FUNCS.keys()))
    agg = AGG_FUNCS[agg_label]

    baseline_label = st.selectbox(
        "Market baseline",
        ["Median", "Mean"],
        help="Reference each DC's markets are compared against.",
    )

    metric_opts = [m for m, src in METRIC_SRC.items() if src in data.columns]
    sel_metrics = st.multiselect(
        "Cost metrics (clustered chart)", metric_opts, default=metric_opts
    )

    dcs = sorted(data["FILL_DC_CD"].dropna().unique())
    sel_dcs = st.multiselect("Fill DC", dcs, default=dcs)

# ---- compute ----
summary = by_xdock(data, sel_year, sel_month, agg)
summary = summary[summary["FILL_DC_CD"].isin(sel_dcs)].copy()

if summary.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# Baseline computed PER Fill DC: every market vs. its own DC's median/mean.
baseline_func = "median" if baseline_label == "Median" else "mean"
summary["baseline"] = summary.groupby("FILL_DC_CD")["normalized_cost"].transform(baseline_func)
summary["deviation"] = summary["normalized_cost"] - summary["baseline"]
summary["deviation_pct"] = summary["deviation"] / summary["baseline"] * 100
summary["status"] = summary["deviation"].apply(
    lambda d: "Above baseline" if d > 0 else ("Below baseline" if d < 0 else "At baseline")
)

period_label = f"{calendar.month_name[sel_month]} {sel_year}"
STATUS_COLORS = {"Above baseline": MCK_RED, "Below baseline": MCK_GREEN, "At baseline": MCK_GRAY}
dc_list = sorted(summary["FILL_DC_CD"].unique())
dc_xdocks = {dc: set(summary.loc[summary["FILL_DC_CD"] == dc, "XDOCK"]) for dc in dc_list}


def _market(xdock: str) -> str:
    """Human-readable market name: drop the XD_<dc>_ prefix and carrier suffix."""
    s = re.sub(r"^XD_\d+_", "", str(xdock))
    s = re.sub(r"\.[A-Za-z0-9]+$", "", s)
    return s.replace("_", " ").title()


summary["MARKET"] = summary["XDOCK"].map(_market)

# ---- compact KPI strip (keeps the top small) ----
_worst = summary.loc[summary["deviation_pct"].idxmax()]
_kpis = [
    ("Period", period_label, f"{summary['XDOCK'].nunique()} markets"),
    ("Fill DCs", str(len(dc_list)), ""),
    ("Largest gap", f"+{_worst['deviation_pct']:.0f}%", _worst["MARKET"]),
]
_cards = "".join(
    f'<div class="kpi"><div class="lbl">{lbl}</div>'
    f'<div class="val">{val}</div><div class="sub">{sub}</div></div>'
    for lbl, val, sub in _kpis
)
st.markdown(f'<div class="kpi-row">{_cards}</div>', unsafe_allow_html=True)

order = summary.sort_values(["FILL_DC_CD", "normalized_cost"], ascending=[True, True])[
    "XDOCK"
].tolist()


def _dc_top_labels(fig: go.Figure, order_: list[str]) -> None:
    """DC separators + top labels for a vertical (x-categorical) chart."""
    pos = 0
    for j, dc in enumerate(dc_list):
        cats = [x for x in order_ if x in dc_xdocks[dc]]
        if not cats:
            continue
        x0, x1 = pos - 0.5, pos + len(cats) - 0.5
        if j > 0:
            fig.add_vline(x=x0, line_color="#d9d9d9", line_width=1)
        fig.add_annotation(
            xref="x", x=(x0 + x1) / 2, yref="paper", y=1.26,
            text=f"<b>DC {dc}</b>", showarrow=False,
            font={"size": 16, "color": MCK_BLUE},
        )
        pos += len(cats)


# =========================================================================== #
# CHARTS - selected from the left panel (View)
# =========================================================================== #
if view == "Market comparison":
    st.subheader("Cost per stop vs. normalized — by Crossdock")

    if not sel_metrics:
        st.info("Select at least one cost metric in the left panel.")
    else:
        sel_cols = [
            METRIC_OUTCOL[m] for m in sel_metrics if METRIC_OUTCOL[m] in summary.columns
        ]
        long = summary.melt(
            id_vars=["XDOCK", "MARKET", "FILL_DC_CD"],
            value_vars=sel_cols,
            var_name="_col",
            value_name="value",
        )
        col_to_label = {v: k for k, v in METRIC_OUTCOL.items()}
        long["Metric"] = long["_col"].map(col_to_label)
        fig1 = px.bar(
            long,
            x="XDOCK",
            y="value",
            color="Metric",
            barmode="group",
            color_discrete_map=METRIC_COLORS,
            category_orders={"XDOCK": order, "Metric": sel_metrics},
            custom_data=["MARKET", "FILL_DC_CD"],
            labels={"value": f"{agg_label} $ per stop", "XDOCK": ""},
        )
        fig1.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b> (DC %{customdata[1]})<br>"
                "%{fullData.name}: $%{y:.2f}<extra></extra>"
            )
        )
        fig1.update_xaxes(
            tickmode="array",
            tickvals=order,
            ticktext=[_market(x) for x in order],
            tickangle=-40,
            tickfont={"size": 10},
        )
        fig1.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
        _dc_top_labels(fig1, order)
        fig1.update_layout(
            height=470,
            margin={"l": 10, "r": 10, "t": 105, "b": 90},
            plot_bgcolor="white",
            bargap=0.25,
            bargroupgap=0.05,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": ""},
        )
        st.plotly_chart(fig1, use_container_width=True)

elif view == "By Fill DC":
    st.subheader("Average cost per stop vs. normalized cost — by Fill DC")

    val_cols = [c for c in ["cost_per_stop", "normalized_cost"] if c in summary.columns]
    dc_avg = summary.groupby("FILL_DC_CD", as_index=False)[val_cols].mean()
    dc_long = dc_avg.melt(
        id_vars="FILL_DC_CD", value_vars=val_cols, var_name="_col", value_name="value"
    )
    dc_label = {"cost_per_stop": "Cost per stop", "normalized_cost": "Normalized cost"}
    dc_long["Metric"] = dc_long["_col"].map(dc_label)
    fig3 = px.bar(
        dc_long,
        x="FILL_DC_CD",
        y="value",
        color="Metric",
        barmode="group",
        color_discrete_map=METRIC_COLORS,
        category_orders={"Metric": ["Cost per stop", "Normalized cost"]},
        text=dc_long["value"].map(lambda v: f"${v:,.2f}"),
        labels={"value": f"{agg_label} $ per stop", "FILL_DC_CD": "Fill DC"},
    )
    fig3.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="DC %{x}<br>%{fullData.name}: $%{y:.2f}<extra></extra>",
    )
    fig3.update_xaxes(type="category", title="Fill DC")
    fig3.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig3.update_layout(
        height=470,
        margin={"l": 10, "r": 10, "t": 40, "b": 40},
        plot_bgcolor="white",
        bargap=0.3,
        bargroupgap=0.05,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": ""},
    )
    st.plotly_chart(fig3, use_container_width=True)

elif view == "Normalization drivers":
    st.subheader("How normalization adjusts cost per stop, by Fill DC")
    st.caption(
        "Normalized cost = **cost per stop ÷ Geo mean**. Geo mean blends the three factors below. "
        "Each bar is the market-average factor (**1.0 = no change**). A factor **above 1.0 scales the "
        "market's cost DOWN** (it was structurally expensive); **below 1.0 scales it UP**."
    )

    mcols = [c for c in MULTIPLIER_COLS if c in summary.columns]
    if not mcols or "normalized_cost" not in summary.columns:
        st.info(
            "Multiplier columns are not in the current dataset. Regenerate LM_CS_slim.csv "
            "with the multiplier fields (GEOGRAPHY_MULTIPLIER, shipment / density multipliers) "
            "to enable this view."
        )
    else:
        DC_SEQ = [MCK_BLUE, MCK_BRIGHT, MCK_GREEN, MCK_OLIVE, MCK_RED, MCK_GRAY]
        dc_list = sorted(summary["FILL_DC_CD"].astype(str).unique())

        # average normalization factor per Fill DC (summary is already market-level)
        rows = []
        for dc in dc_list:
            sub = summary[summary["FILL_DC_CD"].astype(str) == dc]
            for c in mcols:
                val = sub[c].mean()
                if pd.isna(val):
                    continue
                rows.append({"Fill DC": dc, "Factor": MULTIPLIER_LABELS[c], "value": val})
        fac = pd.DataFrame(rows).dropna(subset=["value"])

        if fac.empty:
            st.info("Multiplier values are empty for the current selection.")
        else:
            y_order = [MULTIPLIER_LABELS[c] for c in mcols]  # Geo mean ends up on top

            # KPI: the DC whose Geo mean moves cost the most (normalized = cost / geo_mean)
            if "geo_mean" in mcols:
                gm = fac[fac["Factor"] == MULTIPLIER_LABELS["geo_mean"]].copy()
                gm["impact"] = (1.0 / gm["value"] - 1.0) * 100.0
                _row = gm.loc[gm["impact"].abs().idxmax()]
                _dir = "raises" if _row["impact"] >= 0 else "lowers"
                st.markdown(
                    '<div class="kpi-row"><div class="kpi">'
                    '<div class="lbl">Biggest normalization impact</div>'
                    f'<div class="val">Fill DC {_row["Fill DC"]} · {_row["impact"]:+.0f}% cost</div>'
                    f'<div class="sub">normalization {_dir} this DC\'s cost per stop '
                    f'(Geo mean {_row["value"]:.2f})</div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

            xmax = float(max(2.0, fac["value"].max() * 1.15))
            fig4 = px.bar(
                fac,
                x="value",
                y="Factor",
                orientation="h",
                color="Fill DC",
                barmode="group",
                color_discrete_sequence=DC_SEQ,
                category_orders={"Factor": y_order, "Fill DC": dc_list},
                text=fac["value"].map(lambda v: f"{v:.2f}"),
                labels={"value": "Average factor (1.0 = no change)", "Factor": ""},
            )
            fig4.update_traces(
                textposition="auto",
                cliponaxis=False,
                textfont={"size": 11},
                hovertemplate="%{y} — Fill DC %{fullData.name}: %{x:.2f}×<extra></extra>",
            )
            fig4.update_xaxes(
                range=[0, xmax], showgrid=True, gridcolor="#eee",
            )
            fig4.update_yaxes(tickfont={"size": 12})
            # shade relative to the 1.0 "no change" line so direction reads at a glance
            fig4.add_vrect(x0=0, x1=1, fillcolor=MCK_RED, opacity=0.06,
                           layer="below", line_width=0)
            fig4.add_vrect(x0=1, x1=xmax, fillcolor=MCK_GREEN, opacity=0.06,
                           layer="below", line_width=0)
            fig4.add_vline(x=1.0, line_width=2, line_color="#444", line_dash="dash")
            fig4.add_annotation(
                x=0.5, y=-0.5, xref="x", yref="paper", yshift=-30,
                text="\u25c4 Raises normalized cost", showarrow=False,
                font={"size": 12, "color": MCK_RED},
            )
            fig4.add_annotation(
                x=(1 + xmax) / 2, y=-0.5, xref="x", yref="paper", yshift=-30,
                text="Lowers normalized cost \u25ba", showarrow=False,
                font={"size": 12, "color": MCK_GREEN},
            )
            fig4.update_layout(
                height=max(360, 40 * len(y_order) * max(1, len(dc_list)) + 120),
                margin={"l": 10, "r": 40, "t": 20, "b": 70},
                plot_bgcolor="white",
                bargap=0.30,
                bargroupgap=0.08,
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": "Fill DC"},
            )
            st.plotly_chart(fig4, use_container_width=True)

else:
    st.subheader("Gap analysis — vs. each Fill DC's own baseline")
    st.caption("Gap from each DC's own baseline. **Red = above baseline**, **green = below baseline**.")

    s2 = summary.sort_values(["FILL_DC_CD", "deviation_pct"], ascending=[True, True])
    order2 = s2["XDOCK"].tolist()
    fig2 = px.bar(
        s2,
        x="deviation_pct",
        y="XDOCK",
        orientation="h",
        color="status",
        color_discrete_map=STATUS_COLORS,
        text=s2["deviation_pct"].map(lambda v: f"{v:+.0f}%"),
        custom_data=["MARKET", "FILL_DC_CD", "normalized_cost", "shipments"],
        labels={"deviation_pct": "% gap from DC baseline", "status": ""},
    )
    fig2.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{customdata[0]}</b> (DC %{customdata[1]})<br>%{x:+.1f}% gap vs DC base<br>"
            "cost $%{customdata[2]:.2f}<br>%{customdata[3]:,} shipments<extra></extra>"
        ),
    )
    fig2.update_yaxes(
        categoryorder="array",
        categoryarray=order2,
        tickmode="array",
        tickvals=order2,
        ticktext=[_market(x) for x in order2],
        tickfont={"size": 11},
    )
    pos = 0
    for j, dc in enumerate(dc_list):
        cats = [x for x in order2 if x in dc_xdocks[dc]]
        if not cats:
            continue
        y0, y1 = pos - 0.5, pos + len(cats) - 0.5
        if j % 2 == 1:
            fig2.add_shape(
                type="rect", xref="paper", x0=0, x1=1, yref="y", y0=y0, y1=y1,
                fillcolor="#f4f4f4", line_width=0, layer="below",
            )
        fig2.add_annotation(
            xref="paper", x=1.0, y=(y0 + y1) / 2, yref="y",
            text=f"<b>DC {dc}</b>", showarrow=False, xanchor="left",
            font={"size": 12, "color": MCK_GRAY},
        )
        pos += len(cats)
    fig2.add_vline(x=0, line_color="#444", line_width=1.5)
    fig2.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig2.update_layout(
        height=max(460, 24 * len(order2) + 90),
        margin={"l": 10, "r": 90, "t": 20, "b": 40},
        plot_bgcolor="white",
        bargap=0.28,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": ""},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---- detail (collapsed, not a chart) ----
with st.expander("Detail table & download"):
    cols = ["FILL_DC_CD", "MARKET", "XDOCK"]
    cols += [
        c
        for c in [
            "cost_per_stop",
            "normalized_cost",
            "cleansheet_conservative",
            "cleansheet_aggressive",
        ]
        if c in summary.columns
    ]
    cols += ["baseline", "deviation", "deviation_pct", "status", "shipments"]
    table = summary[cols].rename(
        columns={
            "normalized_cost": f"{agg_label}_normalized_cost",
            "baseline": f"dc_{baseline_label.lower()}_baseline",
            "deviation": "deviation_$",
            "deviation_pct": "deviation_%",
            "shipments": "shipment_count",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"crossdock_cost_{sel_year}_{sel_month:02d}.csv",
        mime="text/csv",
    )
