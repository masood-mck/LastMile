"""
Run all LM Cost offline preprocessing steps in one command.

Steps:
1) Raw CSV -> prepared dataset artifacts
2) Prepared dataset -> default scorecard artifact

Usage:
    python precompute_lm_cost_all.py
    python precompute_lm_cost_all.py --input apps/lm_cost/data/LM_CS_slim.csv.gz --outdir apps/lm_cost/data
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

XDOCK_CANDIDATES = ["XDOCK", "XDOCK_x", "XDOCK_y", "DC_CD"]
YEAR_CANDIDATES = ["YEAR", "ROUTE_YEAR", "year"]
MONTH_CANDIDATES = ["MONTH", "ROUTE_MONTH", "month"]
TARGET = "cost per stop"


def existing_cols(df: pd.DataFrame, wanted: Iterable[str]) -> list[str]:
    return [c for c in wanted if c in df.columns]


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


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


def market_name(xdock: str) -> str:
    parts = str(xdock).split("_")
    if len(parts) >= 3 and parts[0] == "XD" and parts[1].isdigit():
        return " ".join(parts[2:]).split(".")[0].title()
    return str(xdock).replace("_", " ").split(".")[0].title()


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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

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
        df["FILL_DC_CD"] = df["FILL_DC_CD"].astype(str).str.replace(r"\\.0$", "", regex=True).str.strip()
    else:
        df["FILL_DC_CD"] = df["XDOCK"].astype(str).str.extract(r"XD_(\\d+)_", expand=False).fillna("Unknown")

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

    for c in group_cols:
        if c in work.columns:
            work[c] = work[c].astype("category")
    work["paid_flag"] = work["paid_flag"].astype("int8")
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

    if "ROUTE_ID" in paid.columns:
        route_agg_cols = existing_cols(paid, group_cols + ["ROUTE_ID"])
        route_level = paid.groupby(route_agg_cols, dropna=False, observed=True).agg(
            **{TARGET: (TARGET, "mean"), **{c: (c, "mean") for c in existing_cols(paid, metric_cols)}}
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
    categorical_features = [c for c in existing_cols(model_df, ["XDOCK", "FILL_DC_CD"]) if c in group_cols]
    feature_cols = numeric_features + categorical_features

    metrics = {
        "groups_total": int(len(market)),
        "model_groups": int(len(model_df)),
        "feature_count": int(len(feature_cols)),
        "model_status": "model",
    }

    if len(model_df) < 20 or len(feature_cols) == 0:
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

    if cons_col in market.columns and aggr_col in market.columns:
        has_both_cleansheet = market[cons_col].notna() & market[aggr_col].notna()
        market["cleansheet_mid_cost"] = np.where(
            has_both_cleansheet,
            (market[cons_col] + market[aggr_col]) / 2,
            np.nan,
        )
    else:
        market["cleansheet_mid_cost"] = np.nan

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

    market["Market / Xdock"] = market["XDOCK"] if "XDOCK" in market.columns else market[group_cols[0]]
    market["Market Name"] = market["Market / Xdock"].map(market_name)
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


def run_preprocess(input_path: str, out_dir: str) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    print(f"[Step 1/2 preprocess] Reading source: {input_path}")
    raw = pd.read_csv(input_path, low_memory=False)
    prepared = prepare(raw)

    pkl_path = os.path.join(out_dir, "LM_CS_slim_prepared.pkl")
    parquet_path = os.path.join(out_dir, "LM_CS_slim_prepared.parquet")
    manifest_path = os.path.join(out_dir, "LM_CS_slim_preprocess_manifest.json")

    prepared.to_pickle(pkl_path)
    parquet_written = False
    parquet_error = ""
    try:
        prepared.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:
        parquet_error = str(exc)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": input_path,
        "output_pickle": pkl_path,
        "output_parquet": parquet_path if parquet_written else None,
        "parquet_error": parquet_error or None,
        "rows": int(len(prepared)),
        "columns": int(prepared.shape[1]),
        "column_names": list(prepared.columns),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    duration = time.perf_counter() - start
    print(f"[Step 1/2 preprocess] Completed in {duration:,.2f}s")
    return prepared, duration


def run_default_scorecard(prepared: pd.DataFrame, out_dir: str) -> float:
    start = time.perf_counter()
    print("[Step 2/2 scorecard] Building default scorecard artifact")

    scorecard, business_view, model_metrics = build_scorecard(
        prepared,
        group_cols=["XDOCK"],
        min_group_n=30,
        overpay_strong=0.20,
        overpay_possible=0.10,
        underpay_strong=-0.20,
        underpay_possible=-0.10,
        use_p25_p75=False,
        random_state=42,
    )

    payload = {
        "config": {
            "group_cols": ["XDOCK"],
            "min_group_n": 30,
            "overpay_strong": 0.20,
            "overpay_possible": 0.10,
            "underpay_strong": -0.20,
            "underpay_possible": -0.10,
            "use_p25_p75": False,
        },
        "scorecard": scorecard,
        "business_view": business_view,
        "model_metrics": model_metrics,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_pkl = os.path.join(out_dir, "LM_CS_scorecard_default.pkl")
    out_manifest = os.path.join(out_dir, "LM_CS_scorecard_default_manifest.json")
    pd.to_pickle(payload, out_pkl)

    manifest = {
        "generated_at_utc": payload["generated_at_utc"],
        "output_path": out_pkl,
        "groups": int(len(business_view)),
        "scorecard_rows": int(len(scorecard)),
        "model_metrics": model_metrics,
        "config": payload["config"],
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    duration = time.perf_counter() - start
    print(f"[Step 2/2 scorecard] Completed in {duration:,.2f}s")
    return duration


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_outdir = os.path.join(here, "data")
    default_input = os.path.join(default_outdir, "LM_CS_slim.csv.gz")

    parser = argparse.ArgumentParser(description="Run all LM Cost preprocessing tasks.")
    parser.add_argument("--input", default=default_input, help="Raw source CSV/CSV.GZ path")
    parser.add_argument("--outdir", default=default_outdir, help="Output directory")
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.outdir)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    total_start = time.perf_counter()
    prepared, prep_seconds = run_preprocess(in_path, out_dir)
    score_seconds = run_default_scorecard(prepared, out_dir)
    total_seconds = time.perf_counter() - total_start

    print("\nSummary")
    print(f"- Preprocess step : {prep_seconds:,.2f}s")
    print(f"- Scorecard step  : {score_seconds:,.2f}s")
    print(f"- Total runtime   : {total_seconds:,.2f}s")
    print(f"- Output folder   : {out_dir}")


if __name__ == "__main__":
    main()
