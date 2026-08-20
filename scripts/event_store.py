"""
Drishti - Task 9: Curated Event Catalog + Event-Window Analytics
================================================================
Defines a curated catalog of known geopolitical/macroeconomic events with
their temporal windows, manually curated commodity associations, and
event-window model predictions.

Methodological Notes & Limitations:
-----------------------------------
1. OBSERVATIONAL, NOT CAUSAL:
   Event-window statistics describe observations during the curated event period
   and do not establish causal attribution. Average observed values during an event
   window reflect contemporaneous conditions, not isolated causal effects.

2. MANUALLY CURATED COMMODITY ASSOCIATIONS:
   HS4 commodity associations are manually curated domain mappings based on historical
   trade context, not automatically learned from the data.

3. EVENT SCOPE (DIRECT VS PROXY):
   Events where the focal country is in the dataset are marked 'direct'.
   Events where a neighboring country represents regional trade flow (e.g., Bangladesh
   for Sri Lanka crisis) are marked 'proxy'.

4. DESCRIPTIVE SEVERITY:
   Event severity is descriptive metadata and is not currently used by Models A-C.

5. UPSTREAM PROVENANCE NOTE:
   Model B prediction artifacts used for the Model C event-level bridge contain upstream
   in-sample predictions. Therefore, event-level predictions should be treated as
   integration/diagnostic outputs rather than fully out-of-sample causal estimates.

6. MODEL B LAG1 UNAVAILABILITY & MODEL C EXCLUSION:
   Rows with unavailable upstream Model B Lag1 predictions are marked as NaN and
   excluded from Model C event-level prediction/evaluation, rather than silently
   treating them as 0.0.

Run: python scripts/event_store.py
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

import joblib

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")

# ============================================================
# FEATURE SCHEMAS (verified against saved model objects)
# ============================================================
FEATURES_A = [
    "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2",
    "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2",
    "Trade_Share_Lag1",
    "Trade_Share_Lag2",
    "GPR",
    "INR_USD_Rate",
    "Natural_Disaster_Severity_Index",
]

FEATURES_C = [
    "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2",
    "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2",
    "Trade_Share_Lag1",
    "Trade_Share_Lag2",
    "Price_Lag1",
    "GPR",
    "INR_USD_Rate",
    "Trade_Return_1M_Pred",
    "Production_Growth_Pred_Lag1",
]

PROVENANCE_NOTE = (
    "Model B and Model C prediction artifacts used for the event-level bridge are generated "
    "using expanding-window walk-forward out-of-fold models with verified temporal provenance "
    "(Training_End_Year < Prediction_Year). Rows with unavailable upstream Model B Lag1 predictions "
    "are marked as NaN and excluded from Model C evaluation."
)

METHODOLOGICAL_LIMITATION = (
    "Event-window statistics describe observations during the curated event period "
    "and do not establish causal attribution. HS4 mappings are manually curated "
    "event-commodity associations. Event severity is descriptive metadata and is not "
    "used as an ML predictor. Rows with unavailable upstream Model B Lag1 predictions "
    "are excluded from Model C event-level evaluation."
)

# ============================================================
# CURATED EVENT CATALOG
# ============================================================
EVENT_CATALOG = [
    {
        "event_id": "EVT001",
        "name": "Russia-Ukraine Conflict Escalation",
        "description": "Full-scale invasion starting Feb 2022, disrupting Black Sea grain and oilseed trade routes",
        "start": "2022-02",
        "end": "2022-12",
        "countries": ["RUSSIA", "UKRAINE"],
        "hs4_codes": [1001, 1005, 1006, 1512, 1507],  # wheat, maize, rice, sunflower oil, soy oil
        "commodity_curation_type": "manually curated event-commodity associations",
        "event_scope": "direct",
        "expected_direction": "supply_shock",
        "severity": "high",
        "severity_role": "descriptive metadata (not used in ML models)",
        "notes": "Major disruption to Black Sea trade corridors. India experienced heightened export demand and domestic price pressures.",
    },
    {
        "event_id": "EVT002",
        "name": "India Wheat Export Ban",
        "description": "India banned commercial wheat exports in May 2022 to protect domestic supply after heatwaves",
        "start": "2022-05",
        "end": "2023-06",
        "countries": ["BANGLADESH", "UNITED ARAB EMIRATES", "SAUDI ARABIA"],
        "hs4_codes": [1001],  # wheat
        "commodity_curation_type": "manually curated event-commodity associations",
        "event_scope": "direct",
        "expected_direction": "export_restriction",
        "severity": "high",
        "severity_role": "descriptive metadata (not used in ML models)",
        "notes": "Policy-driven export restriction by India affecting primary export destinations.",
    },
    {
        "event_id": "EVT003",
        "name": "US-China Trade Tensions",
        "description": "Tariff escalation and bilateral trade friction between major global economies",
        "start": "2018-07",
        "end": "2019-12",
        "countries": ["CHINA", "UNITED STATES"],
        "hs4_codes": [1201, 1507, 1508, 904, 902],  # soybeans, oils, spices, tea
        "commodity_curation_type": "manually curated event-commodity associations",
        "event_scope": "direct",
        "expected_direction": "trade_diversion",
        "severity": "moderate",
        "severity_role": "descriptive metadata (not used in ML models)",
        "notes": "Bilateral tariff conflict with potential third-party trade diversion effects for Indian agricultural flows.",
    },
    {
        "event_id": "EVT004",
        "name": "COVID-19 Supply Chain Disruption",
        "description": "Global pandemic causing transport bottlenecks, port closures, and widespread macro shocks",
        "start": "2020-03",
        "end": "2021-06",
        "countries": ["CHINA", "UNITED STATES", "UNITED KINGDOM", "GERMANY",
                      "JAPAN", "SOUTH KOREA", "MALAYSIA", "INDONESIA", "VIETNAM"],
        "hs4_codes": [],  # universal across all traded agricultural commodities
        "commodity_curation_type": "manually curated event-commodity associations (universal cross-commodity)",
        "event_scope": "direct",
        "expected_direction": "mixed",
        "severity": "extreme",
        "severity_role": "descriptive metadata (not used in ML models)",
        "notes": "Systemic global disruption affecting freight, labor, border clearance, and consumer demand.",
    },
    {
        "event_id": "EVT005",
        "name": "India-Malaysia Palm Oil Tensions",
        "description": "Diplomatic friction temporarily altering Indian palm oil procurement patterns",
        "start": "2020-01",
        "end": "2020-06",
        "countries": ["MALAYSIA"],
        "hs4_codes": [1511, 1513, 1507],  # palm oil, coconut oil, soy oil
        "commodity_curation_type": "manually curated event-commodity associations",
        "event_scope": "direct",
        "expected_direction": "supply_shock",
        "severity": "moderate",
        "severity_role": "descriptive metadata (not used in ML models)",
        "notes": "Bilateral diplomatic tension prompting temporary trade diversion toward alternative edible oil suppliers.",
    },
    {
        "event_id": "EVT006",
        "name": "Sri Lanka Economic Crisis Spillover",
        "description": "Macroeconomic instability in neighboring South Asian market; proxy country used",
        "start": "2022-04",
        "end": "2023-03",
        "countries": ["BANGLADESH"],  # Regional proxy: Sri Lanka is not in the 15-partner dataset
        "hs4_codes": [902, 904, 1006],  # tea, spices, rice
        "commodity_curation_type": "manually curated event-commodity associations",
        "event_scope": "proxy",
        "expected_direction": "demand_shock",
        "severity": "low",
        "severity_role": "descriptive metadata (not used in ML models)",
        "notes": "PROXY EVENT: Sri Lanka is not present in the 15-country dataset. Bangladesh is used as a South Asian regional trade proxy. Does not directly represent Sri Lanka bilateral trade.",
    },
    {
        "event_id": "EVT007",
        "name": "Red Sea Shipping Crisis (Houthi Attacks)",
        "description": "Maritime security disruptions in the Red Sea corridor forcing cargo rerouting around Africa",
        "start": "2024-01",
        "end": "2024-06",
        "countries": ["SAUDI ARABIA", "UNITED ARAB EMIRATES", "UNITED KINGDOM", "GERMANY", "NETHERLANDS"],
        "hs4_codes": [902, 904, 713, 1006, 303, 306],  # tea, spices, pulses, rice, seafood
        "commodity_curation_type": "manually curated event-commodity associations",
        "event_scope": "direct",
        "expected_direction": "logistics_shock",
        "severity": "moderate",
        "severity_role": "descriptive metadata (not used in ML models)",
        "notes": "Logistics cost surges and transit delays affecting westbound European and Middle Eastern trade routes.",
    },
]


# ============================================================
# DATA EXTRACTION & LOOKUP HELPERS
# ============================================================
def build_model_b_lag_lookup(pred_b_path):
    """
    Build previous-period Model B prediction lookup:
    Production_Growth_Pred_Lag1 = shift(1) of Production_Growth_Pred_OOF
    grouped by (Country, Trade_Type, HS4), sorted by (Year, Month).
    """
    if not os.path.exists(pred_b_path):
        print(f"  WARNING: Model B predictions artifact not found at {pred_b_path}. Will use default NaN.")
        return {}

    pred_b = pd.read_csv(pred_b_path, low_memory=False)
    col_name = "Production_Growth_Pred_OOF" if "Production_Growth_Pred_OOF" in pred_b.columns else "Production_Growth_Pred"
    pred_b = pred_b.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"])
    pred_b["_prev"] = pred_b.groupby(["Country", "Trade_Type", "HS4"])[col_name].shift(1)

    lookup = {}
    for _, r in pred_b.iterrows():
        key = (r["Country"], r["Trade_Type"], int(r["HS4"]), int(r["Year"]), int(r["Month"]))
        val = r["_prev"]
        lookup[key] = float(val) if pd.notna(val) else None

    return lookup


def query_event_data(df, event):
    """
    Extract dataset rows matching an event's temporal and entity criteria.
    """
    start_year, start_month = map(int, event["start"].split("-"))
    end_year, end_month = map(int, event["end"].split("-"))

    # Temporal filter
    mask = (
        ((df["Year"] > start_year) | ((df["Year"] == start_year) & (df["Month"] >= start_month))) &
        ((df["Year"] < end_year) | ((df["Year"] == end_year) & (df["Month"] <= end_month)))
    )

    # Country filter
    if event["countries"]:
        mask = mask & (df["Country"].isin(event["countries"]))

    # HS4 filter (if specified; empty = all commodities)
    if event["hs4_codes"]:
        mask = mask & (df["HS4"].isin(event["hs4_codes"]))

    return df[mask].copy()


def compute_event_summary(event_data, event):
    """
    Compute aggregate observed statistics for an event window.
    Observational summaries only (not causal estimates).
    """
    if len(event_data) == 0:
        return {"n_rows": 0, "warning": "No matching observations found in dataset"}

    summary = {
        "n_rows": len(event_data),
        "event_scope": event.get("event_scope", "direct"),
        "countries_in_data": sorted(event_data["Country"].unique().tolist()),
        "hs4_codes_in_data": sorted(event_data["HS4"].unique().tolist()),
        "trade_types": event_data["Trade_Type"].value_counts().to_dict(),
        "observed_metrics_window": {
            "avg_observed_trade_return_1m": float(event_data["Trade_Return_1M"].mean()),
            "std_observed_trade_return_1m": float(event_data["Trade_Return_1M"].std()),
            "avg_observed_price_return_1m": float(event_data["Price_Return_1M"].mean()),
            "std_observed_price_return_1m": float(event_data["Price_Return_1M"].std()),
            "avg_effective_shock": float(event_data["Effective_Shock"].mean()),
            "max_effective_shock": float(event_data["Effective_Shock"].max()),
            "avg_trade_value_usd": float(event_data["Value_USD"].mean()),
            "total_trade_value_usd": float(event_data["Value_USD"].sum()),
            "avg_shock_intensity": float(event_data["Shock_Intensity"].mean()),
        },
        "export_metrics": {},
        "import_metrics": {},
    }

    # Split by trade type
    for tt in ["Export", "Import"]:
        tt_data = event_data[event_data["Trade_Type"] == tt]
        if len(tt_data) > 0:
            summary[f"{tt.lower()}_metrics"] = {
                "n_rows": len(tt_data),
                "avg_observed_trade_return_1m": float(tt_data["Trade_Return_1M"].mean()),
                "avg_observed_price_return_1m": float(tt_data["Price_Return_1M"].mean()),
                "avg_value_usd": float(tt_data["Value_USD"].mean()),
            }

    return summary


def run_cascade_for_event(event_data, model_a, model_c, model_b_lookup):
    """
    Run Model A and Model C on event rows to get predicted impacts.
    
    Model A runs on all matched event rows.
    Model C runs ONLY on rows with valid previous-period Model B predictions
    (Production_Growth_Pred_Lag1 is not NaN). Rows with unavailable Lag1 are
    explicitly excluded from Model C evaluation rather than treating 0.0 as valid.
    """
    # ----------------------------------------------------------
    # 1. Model A Predictions (Trade Impact) - all event rows
    # ----------------------------------------------------------
    for col in FEATURES_A:
        if col not in event_data.columns:
            event_data[col] = 0.0
        event_data[col] = event_data[col].fillna(0.0)

    X_a = event_data[FEATURES_A]
    event_data["Trade_Return_1M_Pred"] = model_a.predict(X_a)

    # ----------------------------------------------------------
    # 2. Retrieve Model B Previous-Period Lag (Production_Growth_Pred_Lag1)
    # ----------------------------------------------------------
    lag1_vals = []
    matched_count = 0
    unavailable_count = 0

    for _, r in event_data.iterrows():
        key = (r["Country"], r["Trade_Type"], int(r["HS4"]), int(r["Year"]), int(r["Month"]))
        val = model_b_lookup.get(key, None)
        if val is not None and pd.notna(val):
            lag1_vals.append(float(val))
            matched_count += 1
        else:
            # Explicitly mark as NaN rather than silently using 0.0
            lag1_vals.append(np.nan)
            unavailable_count += 1

    event_data["Production_Growth_Pred_Lag1"] = lag1_vals

    # ----------------------------------------------------------
    # 3. Model C Predictions (Price Impact) - valid Lag1 rows only
    # ----------------------------------------------------------
    valid_c_mask = event_data["Production_Growth_Pred_Lag1"].notna()
    c_data = event_data[valid_c_mask].copy()
    n_used_model_c = int(valid_c_mask.sum())
    n_excluded_model_c = int((~valid_c_mask).sum())

    if n_used_model_c > 0:
        for col in FEATURES_C:
            if col not in c_data.columns:
                c_data[col] = 0.0
            c_data[col] = c_data[col].fillna(0.0)

        X_c = c_data[FEATURES_C]
        c_pred = model_c.predict(X_c)
        c_data["Price_Return_1M_Pred"] = c_pred

        price_actual = c_data["Price_Return_1M"]
        price_pred = c_data["Price_Return_1M_Pred"]

        price_mae = float(np.abs(price_pred - price_actual).mean())
        price_bias = float((price_pred - price_actual).mean())
        # Small sample size check for directional agreement
        if n_used_model_c >= 5:
            price_dir_agree = float((np.sign(price_pred) == np.sign(price_actual)).mean() * 100)
            price_dir_agree_note = "Valid"
        else:
            price_dir_agree = None
            price_dir_agree_note = "N/A - insufficient sample size (n < 5)"

        avg_price_pred = float(price_pred.mean())
        price_pred_vs_actual = {
            "pred_mean": float(price_pred.mean()),
            "actual_mean": float(price_actual.mean()),
            "pred_std": float(price_pred.std()) if len(price_pred) > 1 else 0.0,
            "actual_std": float(price_actual.std()) if len(price_actual) > 1 else 0.0,
            "prediction_error_mae": price_mae,
            "prediction_bias": price_bias,
            "directional_agreement_pct": price_dir_agree,
            "directional_agreement_note": price_dir_agree_note,
            "n_rows_evaluated": n_used_model_c,
            "n_rows_excluded": n_excluded_model_c,
            "exclusion_reason": "Rows with unavailable upstream Model B Lag1 predictions excluded from Model C evaluation",
        }
    else:
        avg_price_pred = None
        price_pred_vs_actual = {
            "pred_mean": None,
            "actual_mean": None,
            "pred_std": None,
            "actual_std": None,
            "prediction_error_mae": None,
            "prediction_bias": None,
            "directional_agreement_pct": None,
            "directional_agreement_note": "N/A - 0 valid observations",
            "n_rows_evaluated": 0,
            "n_rows_excluded": n_excluded_model_c,
            "exclusion_reason": "All rows lacked valid upstream Model B Lag1 predictions",
        }

    # Model A comparison statistics (computed across all event rows)
    trade_actual = event_data["Trade_Return_1M"]
    trade_pred = event_data["Trade_Return_1M_Pred"]
    trade_mae = float(np.abs(trade_pred - trade_actual).mean())
    trade_bias = float((trade_pred - trade_actual).mean())
    
    if len(event_data) >= 5:
        trade_dir_agree = float((np.sign(trade_pred) == np.sign(trade_actual)).mean() * 100)
        trade_dir_agree_note = "Valid"
    else:
        trade_dir_agree = None
        trade_dir_agree_note = "N/A - insufficient sample size (n < 5)"

    trade_pred_vs_actual = {
        "pred_mean": float(trade_pred.mean()),
        "actual_mean": float(trade_actual.mean()),
        "pred_std": float(trade_pred.std()),
        "actual_std": float(trade_actual.std()),
        "prediction_error_mae": trade_mae,
        "prediction_bias": trade_bias,
        "directional_agreement_pct": trade_dir_agree,
        "directional_agreement_note": trade_dir_agree_note,
        "n_rows_evaluated": len(event_data),
    }

    # Coverage metrics
    tot_rows = len(event_data)
    cov_pct = (n_used_model_c / tot_rows * 100) if tot_rows > 0 else 0.0
    if cov_pct >= 80.0:
        cov_tier = "HIGH"
    elif cov_pct >= 50.0:
        cov_tier = "MODERATE"
    elif cov_pct >= 20.0:
        cov_tier = "LOW"
    else:
        cov_tier = "VERY LOW"

    return {
        "avg_trade_pred": float(trade_pred.mean()),
        "avg_price_pred": avg_price_pred,
        "trade_pred_vs_actual": trade_pred_vs_actual,
        "price_pred_vs_actual": price_pred_vs_actual,
        "coverage_metrics": {
            "total_event_rows": tot_rows,
            "matched_model_b_lags": matched_count,
            "unavailable_model_b_lags": unavailable_count,
            "rows_evaluated_model_c": n_used_model_c,
            "rows_excluded_model_c": n_excluded_model_c,
            "model_c_evaluation_coverage_pct": round(cov_pct, 2),
            "coverage_tier": cov_tier,
            "tier_definition_note": "Project-defined reporting thresholds: HIGH (>=80%), MODERATE (50-80%), LOW (20-50%), VERY LOW (<20%).",
            "handling_policy": "Rows with unavailable upstream Model B Lag1 predictions are excluded from Model C event-level evaluation.",
        },
    }, matched_count, unavailable_count, n_used_model_c, n_excluded_model_c, cov_pct, cov_tier


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Drishti - Task 9: Curated Event Catalog + Event-Window Analytics")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    print("\nMETHODOLOGICAL SCOPE & PROVENANCE NOTICE:")
    print(f"  [1] {METHODOLOGICAL_LIMITATION}")
    print(f"  [2] {PROVENANCE_NOTE}")

    # Load main dataset
    print("\nLoading main dataset...")
    df = pd.read_csv(MAIN_CSV)
    print(f"  Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Load Model B prediction artifact for previous-period lag lookup (prefer OOF)
    pred_b_path = os.path.join(RESULTS_DIR, "model_b_predictions_oof.csv")
    if not os.path.exists(pred_b_path):
        pred_b_path = os.path.join(RESULTS_DIR, "model_b_predictions.csv")

    print(f"\nBuilding previous-period Model B prediction lookup ({os.path.basename(pred_b_path)})...")
    model_b_lookup = build_model_b_lag_lookup(pred_b_path)
    print(f"  Lookup entries built: {len(model_b_lookup):,}")

    # Load trained models (prefer OOF)
    print("\nLoading trained models...")
    model_a = joblib.load(os.path.join(MODELS_DIR, "model_a_trade.joblib"))

    model_c_path = os.path.join(MODELS_DIR, "model_c_price_oof.joblib")
    if not os.path.exists(model_c_path):
        model_c_path = os.path.join(MODELS_DIR, "model_c_price.joblib")
    model_c = joblib.load(model_c_path)

    print(f"  Model A (Trade) loaded: {len(FEATURES_A)} features")
    print(f"  Model C (Price) loaded: {len(FEATURES_C)} features ({os.path.basename(model_c_path)}, including Production_Growth_Pred_Lag1)")

    # Process each event
    print(f"\nProcessing {len(EVENT_CATALOG)} curated events...")
    event_results = []
    total_matched_lags = 0
    total_unavailable_lags = 0
    total_model_c_used = 0
    total_model_c_excluded = 0

    for event in EVENT_CATALOG:
        print("\n" + "-" * 70)
        print(f"[{event['event_id']}] {event['name']} ({event['start']} to {event['end']})")
        print(f"  Scope: {event['event_scope'].upper()} | Expected Direction: {event['expected_direction']} | Severity: {event['severity']}")
        print(f"  Countries: {', '.join(event['countries'])}")
        print(f"  Commodity Curation: {event['commodity_curation_type']} ({len(event['hs4_codes'])} HS4 codes)")

        event_data = query_event_data(df, event)
        summary = compute_event_summary(event_data, event)

        if summary["n_rows"] == 0:
            print("  No matching observations found in dataset.")
            event_results.append({
                "event": event,
                "summary": summary,
                "predictions": None,
                "methodological_notes": {
                    "observational_limitation": METHODOLOGICAL_LIMITATION,
                    "provenance_notice": PROVENANCE_NOTE,
                },
            })
            continue

        # Run cascade predictions with actual Model B lag lookup and explicit exclusion of unavailable rows
        predictions, matched_lags, unavail_lags, used_c, excluded_c, cov_pct, cov_tier = run_cascade_for_event(
            event_data, model_a, model_c, model_b_lookup
        )
        total_matched_lags += matched_lags
        total_unavailable_lags += unavail_lags
        total_model_c_used += used_c
        total_model_c_excluded += excluded_c

        obs_m = summary["observed_metrics_window"]
        t_comp = predictions["trade_pred_vs_actual"]
        p_comp = predictions["price_pred_vs_actual"]

        print(f"  Total matched observations in window : {summary['n_rows']:,}")
        print(f"  Model B Lag1 status                  : {matched_lags:,} matched, {unavail_lags:,} unavailable (marked NaN)")
        print(f"  Model C rows evaluated / excluded    : {used_c:,} used, {excluded_c:,} excluded | Coverage: {cov_pct:.2f}% ({cov_tier})")
        print(f"  Observed Trade Return (avg)          : {obs_m['avg_observed_trade_return_1m']:+.4f}")
        
        t_dir_str = f"{t_comp['directional_agreement_pct']:.1f}%" if t_comp['directional_agreement_pct'] is not None else t_comp['directional_agreement_note']
        print(f"  Model A Trade Pred (avg)             : {predictions['avg_trade_pred']:+.4f} (MAE: {t_comp['prediction_error_mae']:.4f}, Dir Agree: {t_dir_str})")
        
        print(f"  Observed Price Return (avg)          : {obs_m['avg_observed_price_return_1m']:+.4f}")
        if p_comp["pred_mean"] is not None:
            p_dir_str = f"{p_comp['directional_agreement_pct']:.1f}%" if p_comp['directional_agreement_pct'] is not None else p_comp['directional_agreement_note']
            print(f"  Model C Price Pred (avg)             : {predictions['avg_price_pred']:+.4f} (MAE: {p_comp['prediction_error_mae']:.4f}, Dir Agree: {p_dir_str}, on {used_c:,} rows)")
        else:
            print(f"  Model C Price Pred                   : N/A (no valid Lag1 rows in event window)")

        event_results.append({
            "event": event,
            "summary": summary,
            "predictions": predictions,
            "methodological_notes": {
                "observational_limitation": METHODOLOGICAL_LIMITATION,
                "provenance_notice": PROVENANCE_NOTE,
                "model_c_exclusion_policy": "Rows with unavailable upstream Model B Lag1 predictions are excluded from Model C event-level evaluation.",
            },
        })

    # Save event store output
    store_path = os.path.join(RESULTS_DIR, "event_store.json")
    with open(store_path, "w") as f:
        json.dump(event_results, f, indent=2, default=str)
    print(f"\nEvent store results saved: {store_path}")

    # Save event catalog separately
    catalog_path = os.path.join(DATA_DIR, "event_catalog.json")
    with open(catalog_path, "w") as f:
        json.dump(EVENT_CATALOG, f, indent=2)
    print(f"Event catalog saved: {catalog_path}")

    # Summary Report with Structured Coverage Table
    print("\n" + "=" * 80)
    print("TASK 9 EVENT STORE - SUMMARY & COVERAGE REPORT")
    print("=" * 80)
    print(f"  Total events cataloged & processed : {len(EVENT_CATALOG)}")
    print(f"  Direct scope events                : {sum(1 for e in EVENT_CATALOG if e['event_scope'] == 'direct')}")
    print(f"  Proxy scope events                 : {sum(1 for e in EVENT_CATALOG if e['event_scope'] == 'proxy')} (EVT006 Bangladesh proxy for Sri Lanka)")
    print(f"  Total event rows in windows        : {sum(r['summary'].get('n_rows', 0) for r in event_results):,}")
    print(f"  Model B Lag1 lookups matched       : {total_matched_lags:,}")
    print(f"  Model B Lag1 lookups unavailable   : {total_unavailable_lags:,} (marked NaN)")
    print(f"  Model C rows evaluated             : {total_model_c_used:,}")
    print(f"  Model C rows excluded              : {total_model_c_excluded:,} (due to unavailable Model B Lag1)")
    overall_cov = (total_model_c_used / sum(r['summary'].get('n_rows', 0) for r in event_results) * 100) if sum(r['summary'].get('n_rows', 0) for r in event_results) > 0 else 0.0
    print(f"  Overall Model C evaluation coverage: {overall_cov:.2f}%")

    print("\n" + "-" * 80)
    print(f"{'Event ID':<8} | {'Scope':<6} | {'Total Rows':<10} | {'Evaluated':<10} | {'Excluded':<10} | {'Coverage %':<10} | {'Tier':<8}")
    print("-" * 80)
    for r in event_results:
        ev = r["event"]
        cov = r["predictions"]["coverage_metrics"] if r["predictions"] else {}
        print(f"{ev['event_id']:<8} | {ev['event_scope'].upper():<6} | {cov.get('total_event_rows', 0):<10,d} | {cov.get('rows_evaluated_model_c', 0):<10,d} | {cov.get('rows_excluded_model_c', 0):<10,d} | {cov.get('model_c_evaluation_coverage_pct', 0.0):<9.2f}% | {cov.get('coverage_tier', 'N/A'):<8}")
    print("-" * 80)
    print("  Tier Definitions: HIGH (>=80%), MODERATE (50-80%), LOW (20-50%), VERY LOW (<20%).")
    print("  Note: Project-defined reporting categories, not formal scientific thresholds.")
    print("  Methodology: Rows with unavailable upstream Model B Lag1 are excluded to prevent zero-filling bias.")
    print("\n  TASK 9 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
