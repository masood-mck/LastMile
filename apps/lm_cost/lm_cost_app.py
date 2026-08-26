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
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
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
    "Strong overpay candidate": MCK_RED,
    "Possible overpay": "#F05A28",
    "Normal / inside expected band": MCK_BLUE,
    "Possible underpay": MCK_GREEN,
    "Strong underpay candidate": MCK_OLIVE,
    "Not enough evidence": MCK_GRAY,
}

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
    if "Carrier" in df.columns:
        label = label + " | " + df["Carrier"].astype(str)
    elif "CARRIER_SCAC_CD" in df.columns:
        label = label + " | " + df["CARRIER_SCAC_CD"].astype(str)
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
    df["zero_cost_flag"] = df[TARGET].fillna(0) <= 0
    return df

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
    keep = list(dict.fromkeys(group_cols + ["XDOCK", "MARKET", "FILL_DC_CD", TARGET, "paid_flag", "zero_cost_flag"] + metric_cols))
    work = df[existing_cols(df, keep)].copy()

    # Reduce memory pressure.
    for c in group_cols:
        if c in work.columns:
            work[c] = work[c].astype("category")
    for c in ["paid_flag", "zero_cost_flag"]:
        work[c] = work[c].astype("int8")
    for c in existing_cols(work, [TARGET] + metric_cols):
        work[c] = pd.to_numeric(work[c], errors="coerce").astype("float32")

    counts = (
        work.groupby(group_cols, dropna=False, observed=True)
        .agg(records=(TARGET, "size"), paid_records=("paid_flag", "sum"), zero_records=("zero_cost_flag", "sum"))
        .reset_index()
    )

    paid = work.loc[work["paid_flag"].eq(1)]
    if paid.empty:
        raise ValueError("No positive paid cost rows available after filters.")

    agg_dict = {
        "actual_median_cps": (TARGET, "median"),
        "actual_mean_cps": (TARGET, "mean"),
        "actual_p75_cps": (TARGET, lambda x: x.quantile(0.75)),
        "actual_p90_cps": (TARGET, lambda x: x.quantile(0.90)),
    }
    for c in metric_cols:
        agg_dict[f"median_{clean_col_name(c)}"] = (c, "median")

    paid_agg = paid.groupby(group_cols, dropna=False, observed=True).agg(**agg_dict).reset_index()
    market = counts.merge(paid_agg, on=group_cols, how="left")

    market["zero_rate"] = safe_div(market["zero_records"], market["records"])
    market["paid_rate"] = safe_div(market["paid_records"], market["records"])
    for c in group_cols:
        market[c] = market[c].astype(str)

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
            "zero_rate",
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
    categorical_features = existing_cols(model_df, ["XDOCK", "CARRIER_SCAC_CD", "FILL_DC_CD"])
    # Do not include the same column as both the grouping identifier and a feature when model rows are very sparse.
    categorical_features = [c for c in categorical_features if c in group_cols]
    feature_cols = numeric_features + categorical_features

    metrics = {
        "groups_total": int(len(market)),
        "model_groups": int(len(model_df)),
        "feature_count": int(len(feature_cols)),
        "model_status": "model",
    }

    if len(model_df) < 20 or len(feature_cols) == 0:
        # Fallback: robust benchmark. Still gives a tool result instead of failing.
        p50 = model_df["actual_median_cps"].median() if len(model_df) else market["actual_median_cps"].median()
        p10 = model_df["actual_median_cps"].quantile(0.10) if len(model_df) else market["actual_median_cps"].quantile(0.10)
        p90 = model_df["actual_median_cps"].quantile(0.90) if len(model_df) else market["actual_median_cps"].quantile(0.90)
        market["pred_p10"] = p10
        market["pred_p50"] = p50
        market["pred_p90"] = p90
        if use_p25_p75:
            market["pred_p25"] = model_df["actual_median_cps"].quantile(0.25)
            market["pred_p75"] = model_df["actual_median_cps"].quantile(0.75)
        else:
            market["pred_p25"] = np.nan
            market["pred_p75"] = np.nan
        metrics["model_status"] = "fallback benchmark"
        metrics["mae_cost"] = np.nan
        metrics["median_abs_error_cost"] = np.nan
        metrics["r2_log_target"] = np.nan
    else:
        X = model_df[feature_cols].copy()
        y = np.log1p(model_df["actual_median_cps"].astype(float))
        weights = np.sqrt(model_df["paid_records"].clip(lower=1).astype(float))

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.20, random_state=random_state
        )

        def make_preprocessor():
            num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            )
            return ColumnTransformer(
                [("num", num_pipe, numeric_features), ("cat", cat_pipe, categorical_features)],
                remainder="drop",
            )

        def make_model(q: float):
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

        quantiles = [0.10, 0.50, 0.90] if not use_p25_p75 else [0.10, 0.25, 0.50, 0.75, 0.90]
        models = {}
        for q in quantiles:
            m = make_model(q)
            m.fit(X_train, y_train, model__sample_weight=w_train)
            models[q] = m

        pred_test = np.expm1(models[0.50].predict(X_test)).clip(min=0)
        y_test_cost = np.expm1(y_test)
        metrics.update(
            {
                "quantiles": str(quantiles),
                "mae_cost": float(mean_absolute_error(y_test_cost, pred_test)),
                "median_abs_error_cost": float(median_absolute_error(y_test_cost, pred_test)),
                "r2_log_target": float(r2_score(y_test, models[0.50].predict(X_test))),
            }
        )

        score_eligible = market["actual_median_cps"].notna() & (market["actual_median_cps"] > 0)
        for q, m in models.items():
            col = f"pred_p{int(q * 100):02d}"
            market[col] = np.nan
            market.loc[score_eligible, col] = np.expm1(m.predict(market.loc[score_eligible, feature_cols])).clip(min=0)
        if "pred_p25" not in market.columns:
            market["pred_p25"] = np.nan
        if "pred_p75" not in market.columns:
            market["pred_p75"] = np.nan
        qcols = ["pred_p10", "pred_p25", "pred_p50", "pred_p75", "pred_p90"] if use_p25_p75 else ["pred_p10", "pred_p50", "pred_p90"]
        market[qcols] = np.sort(market[qcols].to_numpy(), axis=1)

    market["model_residual"] = market["actual_median_cps"] - market["pred_p50"]
    market["model_residual_pct"] = safe_div(market["actual_median_cps"], market["pred_p50"]) - 1
    market["normalization_sensitivity_pct"] = safe_div(market.get("median_normalized_cost", np.nan), market["actual_median_cps"]) - 1
    market["norm_sensitivity_abs_pct"] = np.abs(market["normalization_sensitivity_pct"])

    cons_col = "median_cleansheet_cost_per_stop_conservative"
    aggr_col = "median_cleansheet_cost_per_stop_aggressive"
    market["cleansheet_cons_gap_pct"] = safe_div(market["actual_median_cps"], market[cons_col]) - 1 if cons_col in market.columns else np.nan
    market["cleansheet_aggr_gap_pct"] = safe_div(market["actual_median_cps"], market[aggr_col]) - 1 if aggr_col in market.columns else np.nan

    # Cleansheet sensitivity mirrors normalized-cost sensitivity: compare benchmark cost to actual CPS.
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

    market["cleansheet_sensitivity_pct"] = safe_div(market["cleansheet_mid_cost"], market["actual_median_cps"]) - 1
    market["cleansheet_sensitivity_abs_pct"] = np.abs(market["cleansheet_sensitivity_pct"])
    market["combined_sensitivity_abs_pct"] = market[["norm_sensitivity_abs_pct", "cleansheet_sensitivity_abs_pct"]].mean(axis=1, skipna=True)

    market["confidence_score"] = 0
    market.loc[market["paid_records"] >= min_group_n, "confidence_score"] += 1
    market.loc[market["paid_records"] >= min_group_n * 3, "confidence_score"] += 1
    market.loc[market["zero_rate"] <= 0.35, "confidence_score"] += 1
    market.loc[market["combined_sensitivity_abs_pct"].fillna(0) <= 0.40, "confidence_score"] += 1
    market["confidence"] = np.select(
        [market["confidence_score"] >= 4, market["confidence_score"] == 3, market["confidence_score"] == 2],
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

    market.loc[(market["model_residual_pct"] >= overpay_strong) & (market["actual_median_cps"] > market["pred_p90"]) & strong_conf, "classification"] = "Strong overpay candidate"
    market.loc[market["classification"].eq("Normal / inside expected band") & overpay_possible_condition, "classification"] = "Possible overpay"
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
        zero_rate = row.get("zero_rate", None)
        sens = row.get("combined_sensitivity_abs_pct", None)
        if paid >= min_group_n * 3:
            parts.append(f"high volume ({int(paid)} records)")
        elif paid >= min_group_n:
            parts.append(f"adequate volume ({int(paid)} records)")
        else:
            parts.append(f"low volume ({int(paid)} records)")
        if zero_rate is not None:
            if zero_rate <= 0.35:
                parts.append(f"low zero-cost rate ({zero_rate:.0%})")
            else:
                parts.append(f"high zero-cost rate ({zero_rate:.0%})")
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
            "Low volume, high zero rate, or weak basis",
            "Sensitive to normalization and cleansheet assumptions",
        ],
        default="Model-based residual signal",
    )

    # Business-ready aliases.
    market["Market / Xdock"] = market["XDOCK"] if "XDOCK" in market.columns else market[group_cols[0]]
    market["Market Name"] = market["Market / Xdock"].map(market_name)
    if "CARRIER_SCAC_CD" in market.columns:
        market["Carrier"] = market["CARRIER_SCAC_CD"]
    else:
        market["Carrier"] = "All carriers"

    market["Actual CPS"] = market["actual_median_cps"]
    market["Expected CPS"] = market["pred_p50"]
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
    market["Zero Cost Rate"] = market["zero_rate"]
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
            "Carrier",
            "Signal / Classification",
            "Confidence",
            "Classification Reason",
            "Confidence Reason",
            "Business Note",
            "Total Records",
            "Records With CPS",
            "Zero Cost Rate",
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
    strong_over = int(counts.get("Strong overpay candidate", 0))
    poss_over = int(counts.get("Possible overpay", 0))
    strong_under = int(counts.get("Strong underpay candidate", 0))
    poss_under = int(counts.get("Possible underpay", 0))
    not_enough = int(counts.get("Not enough evidence", 0))

    out.append(
        f"The tool scored {total:,} market groups. It identified {strong_over:,} strong overpay candidates and {poss_over:,} possible overpay candidates."
    )
    if strong_under + poss_under > 0:
        out.append(f"It also found {strong_under + poss_under:,} underpay signals. Treat underpay as a validation queue, not an immediate savings opportunity, because low cost can also indicate missing charges or incomplete invoices.")
    if not_enough > 0:
        out.append(f"{not_enough:,} groups are marked as not enough evidence because of low volume, weak confidence, missing expected CPS, or high zero-cost exposure.")
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

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.1rem; padding-bottom: 1rem; }}
      header[data-testid="stHeader"] {{ height: 0; background: transparent; }}
            section[data-testid="stSidebar"] {{
                background: {MCK_NAVY};
                border-right: 1px solid rgba(255, 255, 255, 0.18);
            }}
            section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
                padding-top: .35rem;
            }}
            .logo-wrap {{
                background:#FFFFFF;
                border-radius:10px;
                padding:12px 14px;
                margin:.2rem 0 .8rem 0;
                text-align:left;
            }}
            .logo-wrap img {{
                width:148px;
                height:auto;
                display:block;
            }}
            section[data-testid="stSidebar"] h1,
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] div {{
                color: #FFFFFF;
            }}
            section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] {{
                background: transparent !important;
            }}
            section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
                background: rgba(255, 255, 255, 0.08) !important;
                color: #FFFFFF !important;
                border-color: rgba(255, 255, 255, 0.24) !important;
            }}
            section[data-testid="stSidebar"] div[data-baseweb="select"] input {{
                color: #FFFFFF !important;
            }}
            section[data-testid="stSidebar"] div[data-baseweb="tag"] {{
                background: rgba(255, 255, 255, 0.14) !important;
                color: #FFFFFF !important;
                border: 1px solid rgba(255, 255, 255, 0.22) !important;
            }}
            section[data-testid="stSidebar"] div[data-baseweb="popover"] {{
                background: #FFFFFF !important;
                color: {MCK_NAVY} !important;
            }}
            section[data-testid="stSidebar"] hr {{
                border-color: rgba(255, 255, 255, 0.22);
            }}
      .app-title {{ font-size: 2.05rem; font-weight: 850; color: {MCK_BLUE}; margin: 0 0 .15rem 0; }}
      .app-rule {{ height: 3px; width: 72px; background: {MCK_GREEN}; border-radius: 2px; margin: 0 0 .8rem 0; }}
      .tool-note {{ background: #F7FAFD; border-left: 5px solid {MCK_BRIGHT}; padding: .85rem 1rem; border-radius: 6px; margin: .6rem 0; }}
      .kpi-row {{ display: flex; gap: .6rem; margin: .4rem 0 1rem 0; }}
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

    if "CUST_BUS_TYP_DSCR" in filtered.columns:
        cust_type_options = sorted(filtered["CUST_BUS_TYP_DSCR"].dropna().astype(str).unique())
        sel_cust_types = st.multiselect("Customer type filter", cust_type_options, default=cust_type_options)
        if sel_cust_types:
            filtered = filtered[filtered["CUST_BUS_TYP_DSCR"].astype(str).isin(sel_cust_types)].copy()

    grain = st.radio(
        "Scoring grain",
        ["Market only", "Market + carrier"],
        index=0,
        help="Market only is best for executive market review. Market + carrier is better for sourcing/vendor drilldown.",
    )
    group_cols = ["XDOCK"]
    if grain == "Market + carrier" and "CARRIER_SCAC_CD" in filtered.columns:
        group_cols = ["XDOCK", "CARRIER_SCAC_CD"]

    st.divider()
    st.header("Signal thresholds")
    min_group_n = st.number_input("Minimum paid rows", min_value=5, max_value=5000, value=30, step=5)
    overpay_possible = st.slider("Possible overpay threshold", 0.05, 0.50, 0.10, 0.01)
    overpay_strong = st.slider("Strong overpay threshold", 0.10, 1.00, 0.20, 0.01)
    underpay_possible = -st.slider("Possible underpay threshold", 0.05, 0.50, 0.10, 0.01)
    underpay_strong = -st.slider("Strong underpay threshold", 0.10, 1.00, 0.20, 0.01)
    use_p25_p75 = st.toggle("Fit P25/P75 bands", value=False, help="Leave off for speed. P10/P50/P90 is enough for the decision tool.")

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

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
strong_over = int(counts.get("Strong overpay candidate", 0))
possible_over = int(counts.get("Possible overpay", 0))
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
    ("Scored groups", f"{len(business_view):,}", f"grain: {grain}"),
    ("Overpay candidates", f"{strong_over + possible_over:,}", f"strong: {strong_over:,}"),
    ("Underpay signals", f"{underpay:,}", "validate missing charges"),
    ("Highest gap", pct(worst_gap), str(worst_label)[:32]),
    ("High confidence", f"{high_conf:,}", f"not enough evidence: {not_enough:,}"),
]
kpi_html = '<div class="kpi-row">' + "".join(
    f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val">{val}</div><div class="sub">{sub}</div></div>'
    for lbl, val, sub in cards
) + "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

with st.expander("Quick interpretation", expanded=False):
    st.markdown(
        f"- {strong_over + possible_over:,} overpay candidates, {underpay:,} underpay signals, and {not_enough:,} groups with not enough evidence."
    )
    st.markdown(
        "- Decision rule: use **Actual CPS vs Expected CPS** as the primary signal; use **normalized cost and cleansheet** as supporting evidence."
    )

# --------------------------------------------------------------------------- #
# Market investigation tool
# --------------------------------------------------------------------------- #
st.subheader("1. Market investigation")
market_options = business_view["Market / Xdock"].astype(str).tolist()
if market_options:
    selected_market = st.selectbox("Choose market / xdock", market_options)
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

# --------------------------------------------------------------------------- #
# Business charts
# --------------------------------------------------------------------------- #
st.subheader("2. Business views")
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
        "Carrier",
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
| **Route Operations** | Distance, Stops (route-level), Totes (route-level), Stops (shipment), Totes (shipment), Route Count |
| **Volume & Quality** | Paid Records, Total Records, Zero Rate, Paid Rate |
| **Geography** | Geography Multiplier, Geo Mean |
| **Normalization** | Shipment Norm Multiplier, Miles-per-Stop Norm Multiplier |
| **Cost Structure** | Total Cost, Base Cost, Fuel Cost, Misc Cost |
| **Categorical** | Market (XDOCK), Carrier SCAC, Fill DC (when in scoring grain) |
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

Shipment-level records are aggregated to the selected scoring grain:

- Market only
- Market + Carrier

The model uses operational characteristics such as:

- Distance
- Stops
- Totes
- Shipment volume
- Geography multipliers
- Density multipliers
- Cost structure variables

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

Confidence is a rule-based score from 0 to 4 points. One point is added for each condition met:

1. Paid records are at least the minimum threshold (`paid_records >= min_group_n`)
2. Paid records are at least 3x the minimum threshold (`paid_records >= 3 * min_group_n`)
3. Zero-cost rate is acceptable (`zero_rate <= 35%`)
4. Combined sensitivity is stable (`combined_sensitivity_abs_pct <= 40%`)

Combined sensitivity is the average of:

- Normalization sensitivity: `abs((Normalized Cost / Actual CPS) - 1)`
- Cleansheet sensitivity: `abs((Cleansheet Midpoint / Actual CPS) - 1)`

Where Cleansheet Midpoint is the midpoint between conservative and aggressive cleansheet CPS when both are available.

Confidence labels:

- High: score >= 4
- Medium: score = 3
- Low: score = 2
- Very Low: score <= 1

Very Low confidence groups are treated as Not enough evidence in final classification.

---

### 7. Classification Rules

Classification is assigned in priority order:

#### Strong Overpay Candidate
All three conditions must be met:
1. Gap % ≥ strong overpay threshold (default: +20%)
2. Actual CPS > P90 Expected CPS (actual exceeds the model's upper predicted range)
3. Confidence is High or Medium

#### Strong Underpay Candidate
All three conditions must be met:
1. Gap % ≤ strong underpay threshold (default: -20%)
2. Actual CPS < P10 Expected CPS (actual is below the model's lower predicted range)
3. Confidence is High or Medium

#### Possible Overpay
1. Gap % ≥ possible overpay threshold (default: +10%)
2. (If P25/P75 enabled) Actual CPS > P75 Expected CPS
3. Group not already classified as Strong Overpay

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
    """)

    st.subheader("Current Model Statistics")

    st.dataframe(
        pd.DataFrame([model_metrics]),
        use_container_width=True,
        hide_index=True,
    )

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
    def_class = "Strong overpay candidate" if "Strong overpay candidate" in class_options else class_options[0]

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
        "Carrier",
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
    top_overpay = business_view[business_view["Signal / Classification"].isin(["Strong overpay candidate", "Possible overpay"])].copy()
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
                "Carrier": True,
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
                "Zero Cost Rate": False,
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
                    "Carrier",
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
                "Carrier",
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
                "Carrier",
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
                "Carrier",
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
                    "Carrier",
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
            if "CARRIER_SCAC_CD" in paid_rows.columns:
                paid_rows["Carrier"] = paid_rows["CARRIER_SCAC_CD"].astype(str)
            else:
                paid_rows["Carrier"] = "All carriers"

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
        - Aggregate shipment-level rows to the selected scoring grain before modeling.
        - Use positive paid rows to calculate market-level actual median CPS.
        - Fit a quantile gradient boosting model to estimate expected CPS bands.
        - P50 is the expected median CPS. P10/P90 define the low/high expected range.
                - Classification is based on actual CPS versus expected CPS, with confidence checks for volume, zero-cost rate, and combined sensitivity.
                - Combined sensitivity = average of normalization sensitivity and cleansheet sensitivity.
                - Confidence points (0-4):
                    - +1 if paid records >= minimum threshold
                    - +1 if paid records >= 3x minimum threshold
                    - +1 if zero-cost rate <= 35%
                    - +1 if combined sensitivity <= 40%
                - Confidence labels: High (4), Medium (3), Low (2), Very Low (0-1).

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
