"""
Crossdock Cost Decision Tool
============================

Interactive Streamlit tool for last-mile / xdock market cost review.
It compares actual CPS to model-expected CPS, normalized cost, and cleansheet benchmarks.

Run locally:
    streamlit run lm_cost_app_enhanced.py

Databricks Apps / Databricks terminal:
    export NORM_DATA_TABLE="catalog.schema.table_name"
    # or
    export NORM_DATA_PATH="/Volumes/catalog/schema/vol/LM_CS_slim.csv"
    streamlit run lm_cost_app_enhanced.py
"""

from __future__ import annotations

import os
import re
import gc
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
st.set_page_config(page_title="Crossdock Cost Decision Tool", layout="wide")

# --------------------------------------------------------------------------- #
# Paths / env
# --------------------------------------------------------------------------- #
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_GZ_PATH = os.path.join(_DATA_DIR, "LM_CS_slim.csv.gz")
_CSV_PATH = os.path.join(_DATA_DIR, "LM_CS_slim.csv")
LOCAL_CSV_PATH = _GZ_PATH if os.path.exists(_GZ_PATH) else _CSV_PATH
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
        path = os.environ.get("NORM_DATA_PATH") or LOCAL_CSV_PATH
        df = pd.read_csv(path, low_memory=False)
    return prepare(df)


@st.cache_data(show_spinner="Preparing data...")
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

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

    for c in existing_cols(df, ["XDOCK", "FILL_DC_CD", "CARRIER_SCAC_CD", "ASN_TYPE_CD", "DELIVERY_TYPE"]):
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

    market["confidence_score"] = 0
    market.loc[market["paid_records"] >= min_group_n, "confidence_score"] += 1
    market.loc[market["paid_records"] >= min_group_n * 3, "confidence_score"] += 1
    market.loc[market["zero_rate"] <= 0.35, "confidence_score"] += 1
    market.loc[market["norm_sensitivity_abs_pct"].fillna(0) <= 0.40, "confidence_score"] += 1
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
    market["business_note"] = np.select(
        [
            market["classification"].str.contains("overpay", case=False, na=False) & market["cleansheet_overpay_support"],
            market["classification"].str.contains("underpay", case=False, na=False) & market["cleansheet_underpay_support"],
            market["classification"].eq("Not enough evidence"),
            market["norm_sensitivity_abs_pct"].fillna(0) > 0.40,
        ],
        [
            "Model and cleansheet both point high",
            "Model and cleansheet both point low",
            "Low volume, high zero rate, or weak basis",
            "Sensitive to multiplier assumptions",
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
    market["Paid Records"] = market["paid_records"]
    market["Total Records"] = market["records"]
    market["Zero Cost Rate"] = market["zero_rate"]
    market["Gap vs Conservative Cleansheet %"] = market["cleansheet_cons_gap_pct"]
    market["Gap vs Aggressive Cleansheet %"] = market["cleansheet_aggr_gap_pct"]
    market["Normalization Sensitivity"] = market["norm_sensitivity_abs_pct"]

    display_cols = existing_cols(
        market,
        [
            "Market / Xdock",
            "Market Name",
            "Carrier",
            "Signal / Classification",
            "Confidence",
            "Business Note",
            "Total Records",
            "Paid Records",
            "Zero Cost Rate",
            "Actual CPS",
            "Expected CPS",
            "Expected Low CPS P10",
            "Expected High CPS P90",
            "CPS Gap vs Expected",
            "CPS Gap vs Expected %",
            "Normalized Cost",
            "Normalization Sensitivity",
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
    business_view = business_view.sort_values(["Signal / Classification", "CPS Gap vs Expected %", "Paid Records"], ascending=[True, False, False])
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
        f"Top actual CPS is {money(top.get('Actual CPS'))} versus expected CPS of {money(top.get('Expected CPS'))}; paid record count is {int(top.get('Paid Records', 0)):,}.",
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
    paid = row.get("Paid Records", np.nan)
    conf = row.get("Confidence", "n/a")
    out = [f"Signal: {cls} with {conf} confidence."]
    out.append(f"Actual CPS is {money(actual)} versus expected CPS of {money(expected)}, a gap of {pct(gap)} based on {int(paid):,} paid records." if pd.notna(paid) else f"Actual CPS is {money(actual)} versus expected CPS of {money(expected)}, a gap of {pct(gap)}.")
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
      .app-title {{ font-size: 2.05rem; font-weight: 850; color: {MCK_BLUE}; margin: 0 0 .15rem 0; }}
      .app-rule {{ height: 3px; width: 72px; background: {MCK_GREEN}; border-radius: 2px; margin: 0 0 .8rem 0; }}
      .tool-note {{ background: #F7FAFD; border-left: 5px solid {MCK_BRIGHT}; padding: .85rem 1rem; border-radius: 6px; margin: .6rem 0; }}
      .kpi-row {{ display: flex; gap: .6rem; margin: .4rem 0 1rem 0; }}
      .kpi {{ flex: 1; background: {MCK_LIGHT}; border: 1px solid #E2E8F0; border-radius: 10px; padding: .6rem .8rem; }}
      .kpi .lbl {{ font-size: .68rem; text-transform: uppercase; letter-spacing: .45px; color: #68717A; white-space: nowrap; }}
      .kpi .val {{ font-size: 1.18rem; font-weight: 800; color: {MCK_BLUE}; line-height: 1.25; }}
      .kpi .sub {{ font-size: .72rem; color: #68717A; overflow: hidden; text-overflow: ellipsis; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Crossdock Cost Decision Tool</div>', unsafe_allow_html=True)
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
html = '<div class="kpi-row">' + "".join(
    f'<div class="kpi"><div class="lbl">{lbl}</div><div class="val">{val}</div><div class="sub">{sub}</div></div>'
    for lbl, val, sub in cards
) + "</div>"
st.markdown(html, unsafe_allow_html=True)

with st.expander("Automatic interpretation", expanded=True):
    for line in classification_summary_text(business_view, model_metrics):
        st.markdown(f"- {line}")
    st.markdown(
        "- Decision rule: use **Actual CPS vs Expected CPS** as the primary signal. Use **normalized cost and cleansheet** as supporting evidence."
    )

# --------------------------------------------------------------------------- #
# Classification selector: clickable bar + fallback selectbox
# --------------------------------------------------------------------------- #
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
fig_counts.update_layout(xaxis_title="Signal / Classification", yaxis_title="Market Groups", showlegend=False)

selected_class = None
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
except TypeError:
    st.plotly_chart(fig_counts, use_container_width=True)

class_options = classification_counts["Signal / Classification"].tolist()
def_class = "Strong overpay candidate" if "Strong overpay candidate" in class_options else class_options[0]
if selected_class is None:
    selected_class = st.selectbox("Select a signal to inspect", class_options, index=class_options.index(def_class))
else:
    st.success(f"Selected from chart: {selected_class}")

detail = business_view[business_view["Signal / Classification"].eq(selected_class)].copy()
if "CPS Gap vs Expected %" in detail.columns:
    detail = detail.sort_values("CPS Gap vs Expected %", ascending=False)

with st.expander(f"Automatic explanation for: {selected_class}", expanded=True):
    for line in selected_class_text(detail, selected_class):
        st.markdown(f"- {line}")

st.dataframe(
    detail,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Actual CPS": st.column_config.NumberColumn(format="$%.2f"),
        "Expected CPS": st.column_config.NumberColumn(format="$%.2f"),
        "Expected Low CPS P10": st.column_config.NumberColumn(format="$%.2f"),
        "Expected High CPS P90": st.column_config.NumberColumn(format="$%.2f"),
        "CPS Gap vs Expected": st.column_config.NumberColumn(format="$%.2f"),
        "CPS Gap vs Expected %": st.column_config.NumberColumn(format="%.1f%%"),
        "Normalized Cost": st.column_config.NumberColumn(format="$%.2f"),
        "Cleansheet Conservative": st.column_config.NumberColumn(format="$%.2f"),
        "Cleansheet Aggressive": st.column_config.NumberColumn(format="$%.2f"),
        "Zero Cost Rate": st.column_config.NumberColumn(format="%.1f%%"),
        "Normalization Sensitivity": st.column_config.NumberColumn(format="%.1f%%"),
        "Gap vs Conservative Cleansheet %": st.column_config.NumberColumn(format="%.1f%%"),
        "Gap vs Aggressive Cleansheet %": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

st.download_button(
    "Download selected details",
    detail.to_csv(index=False).encode("utf-8"),
    file_name=f"market_details_{selected_class.replace(' ', '_').replace('/', '_')}.csv",
    mime="text/csv",
)

# --------------------------------------------------------------------------- #
# Market investigation tool
# --------------------------------------------------------------------------- #
st.subheader("2. Market investigation")
market_options = detail["Market / Xdock"].astype(str).tolist() if not detail.empty else business_view["Market / Xdock"].astype(str).tolist()
if market_options:
    selected_market = st.selectbox("Choose market / xdock", market_options)
    row_df = detail[detail["Market / Xdock"].astype(str).eq(selected_market)]
    if row_df.empty:
        row_df = business_view[business_view["Market / Xdock"].astype(str).eq(selected_market)]
    if not row_df.empty:
        row = row_df.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Actual CPS", money(row.get("Actual CPS")), pct(row.get("CPS Gap vs Expected %")))
        cols[1].metric("Expected CPS", money(row.get("Expected CPS")), "P50")
        cols[2].metric("Expected high", money(row.get("Expected High CPS P90")), "P90")
        cols[3].metric("Normalized", money(row.get("Normalized Cost")), plain_pct(row.get("Normalization Sensitivity")))
        cols[4].metric("Paid rows", f"{int(row.get('Paid Records', 0)):,}", row.get("Confidence", ""))
        st.markdown('<div class="tool-note">' + "<br>".join(row_investigation_text(row)) + "</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Business charts
# --------------------------------------------------------------------------- #
st.subheader("3. Business views")
tab1, tab2, tab3, tab4 = st.tabs(["Actual vs expected", "Top overpay", "Cost band", "Cleansheet triangulation"])

hover_cols = existing_cols(
    business_view,
    [
        "Market / Xdock",
        "Market Name",
        "Carrier",
        "Signal / Classification",
        "Confidence",
        "Business Note",
        "Paid Records",
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

with tab1:
    plot_df = business_view[business_view["Actual CPS"].notna() & business_view["Expected CPS"].notna()].copy()
    if len(plot_df):
        fig = px.scatter(
            plot_df,
            x="Expected CPS",
            y="Actual CPS",
            color="Signal / Classification",
            size="Paid Records",
            hover_data=hover_cols,
            color_discrete_map=CLASS_COLORS,
            title="Actual CPS vs Model-Expected CPS",
        )
        max_val = np.nanmax([plot_df["Actual CPS"].max(), plot_df["Expected CPS"].max()])
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", name="Actual = Expected", line=dict(color="black", dash="dash")))
        fig.update_layout(xaxis_title="Expected CPS", yaxis_title="Actual median CPS")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dots above the diagonal cost more than expected after controlling for market profile. Larger bubbles have more paid records.")

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
    clean = business_view[business_view["Gap vs Aggressive Cleansheet %"].notna() & business_view["CPS Gap vs Expected %"].notna()].copy()
    if len(clean):
        fig = px.scatter(
            clean,
            x="Gap vs Aggressive Cleansheet %",
            y="CPS Gap vs Expected %",
            color="Signal / Classification",
            size="Paid Records",
            hover_data=hover_cols,
            color_discrete_map=CLASS_COLORS,
            title="Model gap vs aggressive cleansheet gap",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(xaxis_title="Actual CPS / aggressive cleansheet CPS - 1", yaxis_title="Actual CPS / expected model CPS - 1")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Upper-right quadrant is the strongest overpay support: above expected model CPS and above aggressive cleansheet.")
    else:
        st.info("No aggressive cleansheet values available for the current view.")

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
        - Classification is based on actual CPS versus expected CPS, with confidence checks for volume, zero-cost rate, and normalization sensitivity.

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
