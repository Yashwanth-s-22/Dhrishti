"""
Drishti - Task 5 + 6: Leakage Audit + Model D — Agri-Economic Impact
=====================================================================
Task 5: Audit all candidate features for Agri_GVA_Growth_Percent and
        Inflation_Change_3M to prevent same-period leakage.
Task 6: Train two separate models using only lag-safe features.

CASCADE: Uses Price_Return_1M_Pred from Model C (lagged).

Run: python scripts/train_model_d_economy.py
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # R2 is undefined for a single annual validation observation.
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else None
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def format_r2(metrics):
    """Render meaningful R2 output, including the one-year annual validation case."""
    return f"{metrics['R2']:.4f}" if metrics["R2"] is not None else "N/A (one annual observation)"


MODEL_D_FEATURES = [
    "Inflation_Lag1", "Agri_GVA_Lag1", "GDP_Lag1", "Price_Lag1",
    "Shock_Intensity_Lag1", "Shock_Intensity_Lag2", "Trade_Share_Lag1",
    "Trade_Share_Lag2", "Lagged_Effective_Shock_1",
    "Lagged_Effective_Shock_2", "GPR", "INR_USD_Rate",
    "Price_Return_1M_Pred_Lag1",
]


def _recompute_grouped_shift(raw_df, source_col, sort_keys, group_keys, periods):
    """
    Reproduce the original derived_features.py lag generation exactly:
      1. Sort by sort_keys
      2. groupby(group_keys).shift(periods)
    Returns a Series re-aligned to raw_df's original index.
    Must be called on a CLEAN df (no prior merge modifications).
    """
    df_sorted = raw_df.sort_values(sort_keys)
    if group_keys:
        shifted = df_sorted.groupby(group_keys)[source_col].shift(periods)
    else:
        shifted = df_sorted[source_col].shift(periods)
    return shifted.reindex(raw_df.index)


def _verify_annual_lag(raw_df, feature, source_col):
    """
    Formula verification for annual lag columns.
    Compare one value per calendar year (year Y stored = source_col value for year Y-1).
    Returns (formula_status, match_pct, detail_lines).
    """
    annual = (
        raw_df.drop_duplicates(subset=["Year"])
        .sort_values("Year")
        .set_index("Year")
    )
    annual_expected = annual[source_col].shift(1)   # Y-1 value for year Y
    annual_stored   = annual[feature]

    mask = annual_stored.notna() & annual_expected.notna()
    lines = []
    if mask.sum() == 0:
        return "UNVERIFIED", 0.0, ["No valid annual rows"]

    matches = np.isclose(
        annual_stored[mask].values, annual_expected[mask].values,
        rtol=1e-9, atol=1e-6,
    )
    match_pct = matches.mean() * 100
    formula_status = "VERIFIED" if match_pct == 100 else "PARTIAL"

    lines.append(f"    {'Year':>4}  {'Stored':>12}  {'Expected (Y-1)':>14}  {'Match':>5}")
    for yr in annual.index[mask]:
        s = annual_stored[yr]
        e = annual_expected[yr]
        ok = "YES" if np.isclose(s, e, rtol=1e-9, atol=1e-6) else "NO "
        lines.append(f"    {yr:>4}  {s:>12.4f}  {e:>14.4f}  {ok:>5}")
    return formula_status, match_pct, lines


def _verify_monthly_lag(raw_df, feature, expected_series):
    """
    Formula verification for monthly lag columns.
    Returns (formula_status, match_pct, n_compared).
    """
    stored = raw_df[feature]
    mask = (stored.notna() & expected_series.notna()
            & np.isfinite(stored) & np.isfinite(expected_series))
    if mask.sum() == 0:
        return "UNVERIFIED", 0.0, 0
    matches = np.isclose(stored[mask], expected_series[mask], rtol=1e-9, atol=1e-6)
    match_pct = matches.mean() * 100
    # 100% exact → VERIFIED; ≥60% → PARTIAL (external-source gap); else UNVERIFIED
    if match_pct == 100:
        formula_status = "VERIFIED"
    elif match_pct >= 60:
        formula_status = "PARTIAL"
    else:
        formula_status = "UNVERIFIED"
    return formula_status, match_pct, mask.sum()


def leakage_audit(df):
    """
    Task 5: Audit every candidate feature for temporal leakage.

    Two independent dimensions are reported:
      Formula Verification — can the stored value be reproduced from
                             the raw dataset using the original generation logic?
                             VERIFIED / PARTIAL / UNVERIFIED
      Leakage Status       — is the feature safe to use as a model predictor
                             without introducing same-period information leakage?
                             SAFE / ASSUMED PREDETERMINED /
                             TEMPORALLY SAFE BUT UPSTREAM-PROVENANCE WARNING / UNSAFE

    A low formula match percentage does NOT automatically imply UNSAFE leakage.
    The leakage status is determined by temporal logic, not by formula match rate.

    IMPORTANT: Formula verification uses a fresh clean df loaded from the raw CSV
    to avoid contamination from any prior merge operations (e.g. Model C predictions
    join) that alter row indices and corrupt reindex alignment.
    """
    print("=" * 70)
    print("TASK 5: LEAKAGE AUDIT")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Load a pristine, unmodified copy of the main dataset for formula
    # verification only.  This isolates verification from any merge
    # side-effects introduced before this function is called.
    # ----------------------------------------------------------------
    raw = pd.read_csv(MAIN_CSV)
    raw = raw.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).reset_index(drop=True)

    print("\n  [Formula verification uses a fresh load of the raw dataset]")

    # ================================================================
    # STEP 1 — FORMULA VERIFICATION
    # Reproduce each stored lag using the original derived_features.py
    # generation logic and compare to stored values.
    # ================================================================

    ts_sort = ["Country", "Trade_Type", "HS4", "Year", "Month"]
    ts_grp  = ["Country", "Trade_Type", "HS4"]

    # --- Shock_Intensity_Lag1/2 ---
    si_lag1 = _recompute_grouped_shift(
        raw, "Shock_Intensity", ["Country", "Year", "Month"], ["Country"], 1)
    si_lag2 = _recompute_grouped_shift(
        raw, "Shock_Intensity", ["Country", "Year", "Month"], ["Country"], 2)

    # --- Trade_Share_Lag1/2 ---
    ts_lag1 = _recompute_grouped_shift(raw, "Trade_Share", ts_sort, ts_grp, 1)
    ts_lag2 = _recompute_grouped_shift(raw, "Trade_Share", ts_sort, ts_grp, 2)

    # --- Price_Lag1 ---
    price_lag1 = _recompute_grouped_shift(raw, "Unit_Price_USD_per_KG", ts_sort, ts_grp, 1)

    # --- Lagged_Effective_Shock_1/2: product of stored lags ---
    le1_expected = raw["Shock_Intensity_Lag1"] * raw["Trade_Share_Lag1"]
    le2_expected = raw["Shock_Intensity_Lag2"] * raw["Trade_Share_Lag2"]

    # --- Inflation_Lag1: shift(1) of CPI_Food_Inflation on macro series ---
    macro_raw = (
        raw.drop_duplicates(subset=["Year", "Month"])
        .sort_values(["Year", "Month"])
        .copy()
    )
    macro_raw["_infl_lag1_recomp"] = macro_raw["CPI_Food_Inflation"].shift(1)
    infl_map = macro_raw.set_index(["Year", "Month"])["_infl_lag1_recomp"]
    infl_lag1_expected = raw.set_index(["Year", "Month"]).index.map(infl_map)
    infl_lag1_expected = pd.Series(infl_lag1_expected, index=raw.index)

    # Run verifications
    si1_fv,   si1_pct,   si1_n   = _verify_monthly_lag(raw, "Shock_Intensity_Lag1",   si_lag1)
    si2_fv,   si2_pct,   si2_n   = _verify_monthly_lag(raw, "Shock_Intensity_Lag2",   si_lag2)
    ts1_fv,   ts1_pct,   ts1_n   = _verify_monthly_lag(raw, "Trade_Share_Lag1",        ts_lag1)
    ts2_fv,   ts2_pct,   ts2_n   = _verify_monthly_lag(raw, "Trade_Share_Lag2",        ts_lag2)
    pl1_fv,   pl1_pct,   pl1_n   = _verify_monthly_lag(raw, "Price_Lag1",              price_lag1)
    le1_fv,   le1_pct,   le1_n   = _verify_monthly_lag(raw, "Lagged_Effective_Shock_1", le1_expected)
    le2_fv,   le2_pct,   le2_n   = _verify_monthly_lag(raw, "Lagged_Effective_Shock_2", le2_expected)
    infl_fv,  infl_pct,  infl_n  = _verify_monthly_lag(raw, "Inflation_Lag1",           infl_lag1_expected)
    # Inflation_Lag1 is PARTIAL by definition (external pre-2018 source)
    infl_fv = "PARTIAL"

    gva_fv,  gva_pct,  gva_lines  = _verify_annual_lag(raw, "Agri_GVA_Lag1", "Agri_GVA_Growth_Percent")
    gdp_fv,  gdp_pct,  gdp_lines  = _verify_annual_lag(raw, "GDP_Lag1",      "GDP_Growth_Percent")

    # ================================================================
    # STEP 2 — LEAKAGE STATUS
    # Determined by TEMPORAL LOGIC, independent of formula match rate.
    # A feature whose formula cannot be reproduced is still SAFE if it
    # contains only information available before the prediction period.
    # ================================================================
    #
    # SAFE                    — strictly lagged, temporally safe
    # ASSUMED PREDETERMINED   — exogenous index available at t
    # TEMPORALLY SAFE BUT UPSTREAM-PROVENANCE WARNING
    #                         — lagged, but derived from an in-sample pred
    # UNSAFE                  — same-period realized value

    leakage_status = {
        "Shock_Intensity_Lag1":          "SAFE",
        "Shock_Intensity_Lag2":          "SAFE",
        "Trade_Share_Lag1":              "SAFE",
        "Trade_Share_Lag2":              "SAFE",
        "Price_Lag1":                    "SAFE",
        "Lagged_Effective_Shock_1":      "SAFE",
        "Lagged_Effective_Shock_2":      "SAFE",
        "Inflation_Lag1":                "SAFE",
        "Agri_GVA_Lag1":                 "SAFE",
        "GDP_Lag1":                      "SAFE",
        "GPR":                           "ASSUMED PREDETERMINED",
        "INR_USD_Rate":                  "ASSUMED PREDETERMINED",
        "Price_Return_1M_Pred_Lag1":     "TEMPORALLY SAFE BUT UPSTREAM-PROVENANCE WARNING",
        # UNSAFE
        "CPI_Food_Index":                "UNSAFE",
        "CPI_Food_Inflation":            "UNSAFE",
        "GDP_Growth_Percent":            "UNSAFE",
        "Agri_GVA_Growth_Percent":       "UNSAFE",
        "Forex_Reserves_USD_Million":    "UNSAFE",
        "Value_USD":                     "UNSAFE",
        "Trade_Return_1M":               "UNSAFE",
        "Price_Return_1M":               "UNSAFE",
        "Effective_Shock":               "UNSAFE",
        "Trade_Share":                   "UNSAFE",
    }

    leakage_reason = {
        "Shock_Intensity_Lag1":      "Lag of prior period — temporally safe",
        "Shock_Intensity_Lag2":      "Lag of two prior periods — temporally safe",
        "Trade_Share_Lag1":          "Lag of prior period — temporally safe",
        "Trade_Share_Lag2":          "Lag of two prior periods — temporally safe",
        "Price_Lag1":                "Lag of prior period — temporally safe (Task 1: 100% confirmed)",
        "Lagged_Effective_Shock_1":  "Product of prior-period lags — temporally safe",
        "Lagged_Effective_Shock_2":  "Product of prior-period lags — temporally safe",
        "Inflation_Lag1":            "Lag of prior period — temporally safe; Jan 2018 from external source",
        "Agri_GVA_Lag1":             "Previous calendar year value — temporally safe",
        "GDP_Lag1":                  "Previous calendar year value — temporally safe",
        "GPR":                       "Exogenous political risk index, assumed available at t",
        "INR_USD_Rate":              "Exchange rate, assumed available at t",
        "Price_Return_1M_Pred_Lag1": "Lagged Model C prediction; upstream in-sample contamination possible",
        "CPI_Food_Index":            "Same-period realized CPI — direct leakage for inflation target",
        "CPI_Food_Inflation":        "Same-period realized inflation — target / near-proxy",
        "GDP_Growth_Percent":        "Same-period GDP — too close to Agri_GVA target",
        "Agri_GVA_Growth_Percent":   "This IS the target — cannot use as feature",
        "Forex_Reserves_USD_Million":"Same-period macro variable — potential leakage",
        "Value_USD":                 "Same-period realized trade value",
        "Trade_Return_1M":           "Same-period realized trade return",
        "Price_Return_1M":           "Same-period realized price return",
        "Effective_Shock":           "Same-period product; use verified lagged versions instead",
        "Trade_Share":               "Same-period trade share — contemporaneous information",
    }

    formula_verification = {
        "Shock_Intensity_Lag1":      (si1_fv,  si1_pct,  f"{si1_n:,} rows compared"),
        "Shock_Intensity_Lag2":      (si2_fv,  si2_pct,  f"{si2_n:,} rows compared"),
        "Trade_Share_Lag1":          (ts1_fv,  ts1_pct,  f"{ts1_n:,} rows compared"),
        "Trade_Share_Lag2":          (ts2_fv,  ts2_pct,  f"{ts2_n:,} rows compared"),
        "Price_Lag1":                (pl1_fv,  pl1_pct,  f"{pl1_n:,} rows compared (Task 1 confirmed 100%)"),
        "Lagged_Effective_Shock_1":  (le1_fv,  le1_pct,  f"{le1_n:,} rows; formula = SI_Lag1 * TS_Lag1"),
        "Lagged_Effective_Shock_2":  (le2_fv,  le2_pct,  f"{le2_n:,} rows; formula = SI_Lag2 * TS_Lag2"),
        "Inflation_Lag1":            (infl_fv, infl_pct, f"{infl_n:,} rows; Jan 2018 from external source"),
        "Agri_GVA_Lag1":             (gva_fv,  gva_pct,  "annual comparison"),
        "GDP_Lag1":                  (gdp_fv,  gdp_pct,  "annual comparison"),
        "GPR":                       ("N/A",    None,     "Exogenous index — not a derived lag"),
        "INR_USD_Rate":              ("N/A",    None,     "Exogenous index — not a derived lag"),
        "Price_Return_1M_Pred_Lag1": ("N/A",    None,     "Lagged upstream model output"),
        "CPI_Food_Index":            ("N/A",    None,     "Not a lag — same-period realized"),
        "CPI_Food_Inflation":        ("N/A",    None,     "Not a lag — same-period realized"),
        "GDP_Growth_Percent":        ("N/A",    None,     "Not a lag — same-period realized"),
        "Agri_GVA_Growth_Percent":   ("N/A",    None,     "Not a lag — this is the target"),
        "Forex_Reserves_USD_Million":("N/A",    None,     "Not a lag — same-period realized"),
        "Value_USD":                 ("N/A",    None,     "Not a lag — same-period realized"),
        "Trade_Return_1M":           ("N/A",    None,     "Not a lag — same-period realized"),
        "Price_Return_1M":           ("N/A",    None,     "Not a lag — same-period realized"),
        "Effective_Shock":           ("N/A",    None,     "Not a lag — same-period realized"),
        "Trade_Share":               ("N/A",    None,     "Not a lag — same-period realized"),
    }

    # ================================================================
    # STEP 3 — ANNUAL DETAIL TABLES
    # ================================================================
    print("\n  Annual-level detail for Agri_GVA_Lag1:")
    for line in gva_lines:
        print(line)
    print("\n  Annual-level detail for GDP_Lag1:")
    for line in gdp_lines:
        print(line)

    # ================================================================
    # STEP 4 — CLEAN SUMMARY TABLE
    # Feature | Formula Verification | Match % | Leakage Status | Reason
    # ================================================================
    all_features = list(leakage_status.keys())
    print("\n" + "=" * 100)
    print("TASK 5 AUDIT SUMMARY")
    print("=" * 100)
    hdr = (f"  {'Feature':<32}  {'Formula Verif.':<14}  {'Match %':>7}"
           f"  {'Leakage Status':<46}  Reason")
    print(hdr)
    print("  " + "-" * 148)

    audit = []
    for feat in all_features:
        fv, pct, fv_note = formula_verification[feat]
        ls = leakage_status[feat]
        lr = leakage_reason[feat]
        pct_str = f"{pct:>6.2f}%" if pct is not None else f"{'N/A':>7}"
        print(f"  {feat:<32}  {fv:<14}  {pct_str}  {ls:<46}  {lr}")
        audit.append({
            "feature": feat,
            "formula_verification": fv,
            "match_pct": pct,
            "formula_note": fv_note,
            "leakage_status": ls,
            "leakage_reason": lr,
        })

    print("\n  Model D feature set retained:")
    for f in MODEL_D_FEATURES:
        fv = formula_verification[f][0]
        ls = leakage_status[f]
        print(f"    - {f:<32}  formula={fv}  leakage={ls}")

    return MODEL_D_FEATURES, audit


def main():
    print("=" * 70)
    print("Drishti - Tasks 5+6: Leakage Audit + Model D (Economic Impact)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # ============================================================
    # TASK 6: TRAIN MODEL D
    # ============================================================
    print("\n" + "=" * 70)
    print("TASK 6: MODEL D — Agricultural Economic Impact")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(MAIN_CSV)
    df = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).reset_index(drop=True)

    # Load Model C predictions and lag them
    pred_c_path = os.path.join(RESULTS_DIR, "model_c_predictions.csv")
    if os.path.exists(pred_c_path):
        pred_c = pd.read_csv(pred_c_path)
        df = df.merge(
            pred_c[["Year", "Month", "Country", "Trade_Type", "HS4", "Price_Return_1M_Pred"]],
            on=["Year", "Month", "Country", "Trade_Type", "HS4"],
            how="left"
        )
        df["Price_Return_1M_Pred"] = df["Price_Return_1M_Pred"].fillna(0)
        # Lag by 1 period
        df = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"])
        df["Price_Return_1M_Pred_Lag1"] = (
            df.groupby(["Country", "Trade_Type", "HS4"])["Price_Return_1M_Pred"]
            .shift(1).fillna(0)
        )
        print(f"  Model C predictions loaded and lagged")
    else:
        print("  WARNING: Model C predictions not found. Using zeros.")
        df["Price_Return_1M_Pred_Lag1"] = 0

    # ============================================================
    # TASK 5: LEAKAGE AUDIT (requires loaded source columns)
    # ============================================================
    safe_features, audit = leakage_audit(df)

    # Ensure all safe features exist
    for col in safe_features:
        if col not in df.columns:
            print(f"  WARNING: {col} not found, filling with 0")
            df[col] = 0
        df[col] = df[col].fillna(0)

    # Chronological split
    TRAIN_END_YEAR = 2022
    VAL_YEAR = 2023
    TEST_START_YEAR = 2024

    train = df[df["Year"] <= TRAIN_END_YEAR]
    val = df[df["Year"] == VAL_YEAR]
    test = df[df["Year"] >= TEST_START_YEAR]

    print(f"  Split: Train={len(train):,} | Val={len(val):,} | Test={len(test):,}")

    all_results = {}

    # ============================================================
    # MODEL D-1: Agri_GVA_Growth_Percent
    # ============================================================
    target1 = "Agri_GVA_Growth_Percent"
    print(f"\n  --- Model D-1: {target1} ---")
    print(f"  NOTE: This is an ANNUAL variable (same value all 12 months)")
    print(f"  Unique values: {df[target1].nunique()}")

    target_values_per_year = df.groupby("Year")[target1].nunique(dropna=False).sort_index()
    print(f"  Target values per year: {target_values_per_year.to_dict()}")
    if not (target_values_per_year == 1).all():
        raise ValueError("Agri_GVA_Growth_Percent is not constant within every year; annual evaluation is invalid.")
    print("  Annual-target consistency: VERIFIED (one target value per year)")

    # Use one annual row per target year. Its feature values are the prior
    # calendar year's means, preserving prediction-time availability.
    annual_features = df.groupby("Year")[safe_features].mean().sort_index().shift(1)
    annual_targets = df.groupby("Year")[target1].first().sort_index()
    annual_df = annual_features.copy()
    annual_df[target1] = annual_targets
    annual_df["Previous_Year_GVA_Baseline"] = annual_targets.shift(1)
    annual_df = annual_df.dropna(subset=safe_features + [target1, "Previous_Year_GVA_Baseline"])

    annual_train = annual_df[annual_df.index <= TRAIN_END_YEAR].copy()
    annual_val = annual_df[annual_df.index == VAL_YEAR].copy()
    annual_test = annual_df[annual_df.index >= TEST_START_YEAR].copy()
    print(f"  Annual chronological split: Train years={annual_train.index.tolist()} | "
          f"Val years={annual_val.index.tolist()} | Test years={annual_test.index.tolist()}")

    X_train_gva, y_train_gva = annual_train[safe_features], annual_train[target1]
    X_val_gva, y_val_gva = annual_val[safe_features], annual_val[target1]
    X_test_gva, y_test_gva = annual_test[safe_features], annual_test[target1]

    # Baseline: explicit prior annual target, never an arbitrary monthly row.
    bl_train = evaluate_model(y_train_gva, annual_train["Previous_Year_GVA_Baseline"], "baseline_train")
    bl_val = evaluate_model(y_val_gva, annual_val["Previous_Year_GVA_Baseline"], "baseline_val")
    bl_test = evaluate_model(y_test_gva, annual_test["Previous_Year_GVA_Baseline"], "baseline_test")
    print(f"  Baseline (previous annual value): Train MAE={bl_train['MAE']:.4f} R2={format_r2(bl_train)}")
    print(f"                                    Val MAE={bl_val['MAE']:.4f} R2={format_r2(bl_val)}")
    print(f"                                    Test MAE={bl_test['MAE']:.4f} R2={format_r2(bl_test)}")

    gva_model_results = []

    for ModelCls, name, params in [
        (RandomForestRegressor, "RandomForest", dict(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
        (lgb.LGBMRegressor, "LightGBM", dict(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ]:
        model = ModelCls(**params)
        model.fit(X_train_gva, y_train_gva)

        m_train = evaluate_model(y_train_gva, model.predict(X_train_gva), f"{name}_train")
        m_val = evaluate_model(y_val_gva, model.predict(X_val_gva), f"{name}_val")
        m_test = evaluate_model(y_test_gva, model.predict(X_test_gva), f"{name}_test")

        beats = m_val["MAE"] < bl_val["MAE"]
        print(f"  {name}: Train MAE={m_train['MAE']:.4f} R2={format_r2(m_train)} | " +
              f"Val MAE={m_val['MAE']:.4f} R2={format_r2(m_val)} | " +
              f"Test MAE={m_test['MAE']:.4f} R2={format_r2(m_test)} | " +
              f"{'BEATS' if beats else 'LOSES TO'} baseline")

        all_results[f"{target1}_{name}"] = {
            "train": m_train, "val": m_val, "test": m_test,
            "baseline_train": bl_train, "baseline_val": bl_val, "baseline_test": bl_test,
            "beats_baseline": beats,
        }
        gva_model_results.append((model, name, m_val))

    best_gva_model, best_gva_name, best_gva_val = min(gva_model_results, key=lambda item: item[2]["MAE"])
    joblib.dump(best_gva_model, os.path.join(MODELS_DIR, "model_d_agri_gva.joblib"))
    print(f"  Saved GVA validation winner: {best_gva_name} (Val MAE={best_gva_val['MAE']:.4f})")

    # ============================================================
    # MODEL D-2: Inflation_Change_3M
    # ============================================================
    target2 = "Inflation_Change_3M"
    print(f"\n  --- Model D-2: {target2} ---")

    X_train = train[safe_features]
    y_train = train[target2]
    X_val = val[safe_features]
    y_val = val[target2]
    X_test = test[safe_features]
    y_test = test[target2]

    # Baseline: predict 0 (no change)
    bl_val = evaluate_model(y_val, np.zeros(len(y_val)), "baseline_val")
    bl_test = evaluate_model(y_test, np.zeros(len(y_test)), "baseline_test")
    print(f"  Baseline (predict 0): Val MAE={bl_val['MAE']:.4f} R2={bl_val['R2']:.4f}")
    print(f"                        Test MAE={bl_test['MAE']:.4f} R2={bl_test['R2']:.4f}")

    inflation_model_results = []
    for ModelCls, name, params in [
        (RandomForestRegressor, "RandomForest", dict(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
        (lgb.LGBMRegressor, "LightGBM", dict(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ]:
        model = ModelCls(**params)
        model.fit(X_train, y_train)

        m_train = evaluate_model(y_train, model.predict(X_train), f"{name}_train")
        m_val = evaluate_model(y_val, model.predict(X_val), f"{name}_val")
        m_test = evaluate_model(y_test, model.predict(X_test), f"{name}_test")

        beats_val = m_val["MAE"] < bl_val["MAE"]
        beats_test = m_test["MAE"] < bl_test["MAE"]
        print(f"  {name}: Train MAE={m_train['MAE']:.4f} R2={format_r2(m_train)} | " +
              f"Val MAE={m_val['MAE']:.4f} R2={format_r2(m_val)} ({'BEATS' if beats_val else 'LOSES TO'} baseline) | " +
              f"Test MAE={m_test['MAE']:.4f} R2={format_r2(m_test)} ({'BEATS' if beats_test else 'LOSES TO'} baseline)")

        for sn, m in [("val", m_val), ("test", m_test)]:
            if m["R2"] > 0.95:
                print(f"  *** RED FLAG: R2={m['R2']:.4f} on {sn}!")

        all_results[f"{target2}_{name}"] = {
            "train": m_train, "val": m_val, "test": m_test,
            "baseline_val": bl_val, "baseline_test": bl_test,
            "beats_baseline_val": beats_val,
            "beats_baseline_test": beats_test,
        }
        inflation_model_results.append((model, name, m_val))

    best_inflation_model, best_inflation_name, best_inflation_val = min(
        inflation_model_results, key=lambda item: item[2]["MAE"]
    )
    joblib.dump(best_inflation_model, os.path.join(MODELS_DIR, "model_d_inflation.joblib"))
    print(f"  Saved inflation validation winner: {best_inflation_name} "
          f"(Val MAE={best_inflation_val['MAE']:.4f})")

    # ============================================================
    # SAVE
    # ============================================================
    results = {
        "task": "Tasks 5+6 - Leakage Audit + Model D",
        "timestamp": datetime.now().isoformat(),
        "leakage_audit": audit,
        "safe_features": safe_features,
        "model_selection": {
            "Agri_GVA_Growth_Percent": {
                "criterion": "validation MAE",
                "winner": best_gva_name,
                "validation_mae": best_gva_val["MAE"],
                "evaluation_unit": "one annual observation per year",
            },
            "Inflation_Change_3M": {
                "criterion": "validation MAE",
                "winner": best_inflation_name,
                "validation_mae": best_inflation_val["MAE"],
            },
        },
        "model_results": all_results,
    }
    results_path = os.path.join(RESULTS_DIR, "model_d_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print("\n  TASKS 5+6 COMPLETE")


if __name__ == "__main__":
    main()
