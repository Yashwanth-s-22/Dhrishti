"""
Drishti — Task 1: Data Validation & Formula Verification
=========================================================
Validates both datasets against documented schema (Part C of project spec),
then empirically verifies undocumented derived columns (B8/B9).

Run: python scripts/task1_data_validation.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths — adjust if data is not in data/ subdirectory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Try data/ subdir first, then project root
MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")
CROP_CSV = os.path.join(DATA_DIR, "Crop_Production_Final.csv")

if not os.path.exists(MAIN_CSV):
    MAIN_CSV = os.path.join(BASE_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")
if not os.path.exists(CROP_CSV):
    CROP_CSV = os.path.join(BASE_DIR, "Crop_Production_Final.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_FILE = os.path.join(RESULTS_DIR, "task1_data_validation_report.txt")

# ============================================================
# EXPECTED SCHEMA (from Part C)
# ============================================================
EXPECTED_MAIN_ROWS = 139_626
EXPECTED_MAIN_COLS = 86
EXPECTED_MAIN_YEAR_RANGE = (2018, 2025)
EXPECTED_MAIN_COUNTRIES = 15
EXPECTED_MAIN_HS4_CODES = 168

EXPECTED_CROP_ROWS = 80_450
EXPECTED_CROP_COLS = 24
EXPECTED_CROP_YEAR_RANGE_STR = ("2018-19", "2022-23")
EXPECTED_CROP_STATES = 37
EXPECTED_CROP_DISTRICTS = 736
EXPECTED_CROP_CROPS = 54
EXPECTED_CROP_SEASONS = 5

EXPECTED_MAIN_COLUMNS = [
    "Year", "Month", "Country", "Trade_Type", "HS4", "HS2", "Commodity",
    "Value_USD", "NetWeight_KG", "Unit_Price_USD_per_KG",
    "Signed_Trade_Value_USD", "MoM_Change_Value", "Rolling_3M_Volatility",
    "Total_Event_Count", "Avg_Goldstein", "Avg_Tone", "Total_Mentions",
    "Total_Sources", "Shock_Intensity", "Conflict_Event_Count",
    "Protest_Event_Count", "Trade_Shock_Count", "Sanction_Threat_Count",
    "Incoming_Shock_Count", "Outgoing_Shock_Count", "Net_Hostility",
    "Conflict_Density", "Protest_Density", "Trade_Shock_Density",
    "Total_Country_Trade_USD", "Trade_Share", "Effective_Shock",
    "Conflict_Exposure", "Protest_Exposure", "Trade_Shock_Exposure",
    "Incoming_Shock_Exposure", "Outgoing_Shock_Exposure",
    "Net_Hostility_Exposure", "Log_Value_USD", "Shock_Intensity_Lag1",
    "Shock_Intensity_Lag2", "Trade_Share_Lag1", "Trade_Share_Lag2",
    "Lagged_Effective_Shock_1", "Lagged_Effective_Shock_2",
    "CPI_Food_Index", "CPI_Food_Inflation", "INR_USD_Rate",
    "Forex_Reserves_USD_Million", "GPR", "GDP_Growth_Percent",
    "Agri_GVA_Growth_Percent", "Trade_Return_1M", "Trade_Return_3M",
    "Trade_Return_6M", "Price_Return_1M", "Price_Return_3M", "Price_Lag1",
    "Inflation_Change_3M", "Inflation_Lag1", "Agri_GVA_Lag1", "GDP_Lag1",
    "Shock_Trend_3M", "Trade_Momentum", "Trade_Type_Encoded",
    "Country_HS4", "Weighted_Trade_Impact", "Price_Volatility_3M",
    "Country_ISO3", "Natural_Disaster_Count",
    "Natural_Disaster_Type_Count", "Natural_Disaster_Types",
    "Natural_Disaster_Flood_Count", "Natural_Disaster_Storm_Count",
    "Natural_Disaster_Drought_Count", "Natural_Disaster_Earthquake_Count",
    "Natural_Disaster_Wildfire_Count", "Natural_Disaster_Deaths",
    "Natural_Disaster_Affected_Population", "Natural_Disaster_Damage_000USD",
    "Natural_Disaster_Deaths_Observed_Events",
    "Natural_Disaster_Affected_Observed_Events",
    "Natural_Disaster_Damage_Observed_Events",
    "Natural_Disaster_Occurrence", "Natural_Disaster_Severity_Index",
    "Natural_Disaster_Trade_Exposure_USD",
]

EXPECTED_CROP_COLUMNS = [
    "State", "District", "Crop", "Season", "Season_Order", "Crop_Year",
    "Start_Year", "End_Year", "Season_Start_Month", "Season_End_Month",
    "Season_Months", "Season_Start_Year", "Season_End_Year",
    "Season_Crosses_Calendar_Year", "Area_Ha", "Production_Tonnes",
    "Yield_Ton_per_Ha", "Production_YoY_Change_Pct",
    "Yield_YoY_Change_Pct", "Area_YoY_Change_Pct", "Production_3Y_Mean",
    "Yield_3Y_Mean", "Production_Deviation_From_3Y_Mean",
    "Yield_Deviation_From_3Y_Mean",
]

# Six candidate target columns that should have zero nulls
ZERO_NULL_TARGETS = [
    "Trade_Return_1M", "Trade_Return_3M", "Price_Return_1M",
    "Price_Return_3M", "Inflation_Change_3M", "Agri_GVA_Growth_Percent",
]


def report(lines, file_handle):
    """Print and log simultaneously."""
    for line in lines if isinstance(lines, list) else [lines]:
        # Use ASCII-safe output for Windows console compatibility
        safe_line = line.encode("ascii", errors="replace").decode("ascii")
        print(safe_line)
        file_handle.write(line + "\n")


def validate_main_dataset(df, f):
    """Validate Drishti_Cascade_Final_With_EMDAT.csv against documented schema."""
    report("=" * 70, f)
    report("SECTION 1: MAIN DATASET VALIDATION", f)
    report("  Drishti_Cascade_Final_With_EMDAT.csv", f)
    report("=" * 70, f)

    issues = []

    # --- 1.1 Shape ---
    report(f"\n1.1 Shape check:", f)
    report(f"  Expected rows: {EXPECTED_MAIN_ROWS:,}  |  Actual: {len(df):,}  |  {'✓' if len(df) == EXPECTED_MAIN_ROWS else '✗ DISCREPANCY'}", f)
    report(f"  Expected cols: {EXPECTED_MAIN_COLS}  |  Actual: {df.shape[1]}  |  {'✓' if df.shape[1] == EXPECTED_MAIN_COLS else '✗ DISCREPANCY'}", f)
    if len(df) != EXPECTED_MAIN_ROWS:
        issues.append(f"Row count mismatch: expected {EXPECTED_MAIN_ROWS}, got {len(df)}")
    if df.shape[1] != EXPECTED_MAIN_COLS:
        issues.append(f"Column count mismatch: expected {EXPECTED_MAIN_COLS}, got {df.shape[1]}")

    # --- 1.2 Column names ---
    report(f"\n1.2 Column name check:", f)
    actual_cols = set(df.columns)
    expected_cols = set(EXPECTED_MAIN_COLUMNS)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    if missing:
        report(f"  ✗ Missing columns: {sorted(missing)}", f)
        issues.append(f"Missing columns: {sorted(missing)}")
    if extra:
        report(f"  ⚠ Extra columns (undocumented): {sorted(extra)}", f)
        issues.append(f"Extra columns: {sorted(extra)}")
    if not missing and not extra:
        report(f"  ✓ All {len(EXPECTED_MAIN_COLUMNS)} expected columns present, no extras", f)

    # --- 1.3 Year range ---
    report(f"\n1.3 Year range:", f)
    yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
    report(f"  Expected: {EXPECTED_MAIN_YEAR_RANGE}  |  Actual: ({yr_min}, {yr_max})  |  {'✓' if (yr_min, yr_max) == EXPECTED_MAIN_YEAR_RANGE else '✗ DISCREPANCY'}", f)
    if (yr_min, yr_max) != EXPECTED_MAIN_YEAR_RANGE:
        issues.append(f"Year range mismatch: expected {EXPECTED_MAIN_YEAR_RANGE}, got ({yr_min}, {yr_max})")

    # --- 1.4 Countries ---
    report(f"\n1.4 Partner countries:", f)
    countries = sorted(df["Country"].unique())
    n_countries = len(countries)
    report(f"  Expected count: {EXPECTED_MAIN_COUNTRIES}  |  Actual: {n_countries}  |  {'✓' if n_countries == EXPECTED_MAIN_COUNTRIES else '✗ DISCREPANCY'}", f)
    report(f"  Countries: {countries}", f)
    if n_countries != EXPECTED_MAIN_COUNTRIES:
        issues.append(f"Country count mismatch: expected {EXPECTED_MAIN_COUNTRIES}, got {n_countries}")

    # --- 1.5 HS4 codes ---
    report(f"\n1.5 HS4 commodity codes:", f)
    n_hs4 = df["HS4"].nunique()
    report(f"  Expected: {EXPECTED_MAIN_HS4_CODES}  |  Actual: {n_hs4}  |  {'✓' if n_hs4 == EXPECTED_MAIN_HS4_CODES else '✗ DISCREPANCY'}", f)
    if n_hs4 != EXPECTED_MAIN_HS4_CODES:
        issues.append(f"HS4 count mismatch: expected {EXPECTED_MAIN_HS4_CODES}, got {n_hs4}")

    # --- 1.6 HS2 chapters (should be 1-20 only) ---
    report(f"\n1.6 HS2 chapters (expected: 1-20 only, no ch.27 or ch.31):", f)
    hs2_vals = sorted(df["HS2"].unique())
    report(f"  Unique HS2 values: {hs2_vals}", f)
    has_27 = 27 in hs2_vals
    has_31 = 31 in hs2_vals
    if has_27 or has_31:
        report(f"  ✗ Unexpected: ch.27 present={has_27}, ch.31 present={has_31}", f)
        issues.append("Unexpected HS2 chapters (27/31) found")
    else:
        report(f"  ✓ No fertilizer (ch.31) or fuel (ch.27) rows", f)

    # --- 1.7 Null analysis ---
    report(f"\n1.7 Null analysis:", f)
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) == 0:
        report(f"  ✓ Zero null values across all {df.shape[1]} columns", f)
    else:
        report(f"  Columns with nulls ({len(cols_with_nulls)}):", f)
        for col, count in cols_with_nulls.items():
            pct = count / len(df) * 100
            report(f"    {col}: {count:,} ({pct:.2f}%)", f)

    # --- 1.8 Zero-null target columns ---
    report(f"\n1.8 Zero-null target columns (critical for modeling):", f)
    for col in ZERO_NULL_TARGETS:
        if col in df.columns:
            n_null = df[col].isnull().sum()
            report(f"  {col}: {n_null} nulls  |  {'✓' if n_null == 0 else '✗ HAS NULLS'}", f)
            if n_null > 0:
                issues.append(f"Target column {col} has {n_null} nulls")
        else:
            report(f"  {col}: ✗ COLUMN MISSING", f)
            issues.append(f"Target column {col} is missing")

    # --- 1.9 Trade_Type values ---
    report(f"\n1.9 Trade_Type values:", f)
    trade_types = sorted(df["Trade_Type"].unique())
    report(f"  {trade_types}", f)

    # --- 1.10 Summary statistics for key numeric columns ---
    report(f"\n1.10 Summary statistics (key columns):", f)
    key_cols = ["Value_USD", "Shock_Intensity", "Trade_Share", "Effective_Shock",
                "Trade_Return_1M", "Price_Return_1M", "CPI_Food_Inflation",
                "Agri_GVA_Growth_Percent"]
    for col in key_cols:
        if col in df.columns:
            s = df[col]
            report(f"  {col}: min={s.min():.4f}, median={s.median():.4f}, mean={s.mean():.4f}, max={s.max():.4f}, std={s.std():.4f}", f)

    return issues


def validate_crop_dataset(df, f):
    """Validate Crop_Production_Final.csv against documented schema."""
    report("\n" + "=" * 70, f)
    report("SECTION 2: CROP PRODUCTION DATASET VALIDATION", f)
    report("  Crop_Production_Final.csv", f)
    report("=" * 70, f)

    issues = []

    # --- 2.1 Shape ---
    report(f"\n2.1 Shape check:", f)
    report(f"  Expected rows: {EXPECTED_CROP_ROWS:,}  |  Actual: {len(df):,}  |  {'✓' if len(df) == EXPECTED_CROP_ROWS else '✗ DISCREPANCY'}", f)
    report(f"  Expected cols: {EXPECTED_CROP_COLS}  |  Actual: {df.shape[1]}  |  {'✓' if df.shape[1] == EXPECTED_CROP_COLS else '✗ DISCREPANCY'}", f)
    if len(df) != EXPECTED_CROP_ROWS:
        issues.append(f"Row count mismatch: expected {EXPECTED_CROP_ROWS}, got {len(df)}")
    if df.shape[1] != EXPECTED_CROP_COLS:
        issues.append(f"Column count mismatch: expected {EXPECTED_CROP_COLS}, got {df.shape[1]}")

    # --- 2.2 Column names ---
    report(f"\n2.2 Column name check:", f)
    actual_cols = set(df.columns)
    expected_cols = set(EXPECTED_CROP_COLUMNS)
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols
    if missing:
        report(f"  ✗ Missing columns: {sorted(missing)}", f)
        issues.append(f"Missing columns: {sorted(missing)}")
    if extra:
        report(f"  ⚠ Extra columns: {sorted(extra)}", f)
        issues.append(f"Extra columns: {sorted(extra)}")
    if not missing and not extra:
        report(f"  ✓ All {len(EXPECTED_CROP_COLUMNS)} expected columns present, no extras", f)

    # --- 2.3 Crop years ---
    report(f"\n2.3 Crop year range:", f)
    crop_years = sorted(df["Crop_Year"].unique())
    report(f"  Expected: {EXPECTED_CROP_YEAR_RANGE_STR}", f)
    report(f"  Actual: {crop_years[0]} to {crop_years[-1]} ({len(crop_years)} unique)", f)
    report(f"  All values: {crop_years}", f)

    # --- 2.4 Dimensions ---
    report(f"\n2.4 Dimension counts:", f)
    n_states = df["State"].nunique()
    n_districts = df["District"].nunique()
    n_crops = df["Crop"].nunique()
    seasons = sorted(df["Season"].unique())
    n_seasons = len(seasons)

    report(f"  States:    Expected {EXPECTED_CROP_STATES}  |  Actual: {n_states}  |  {'✓' if n_states == EXPECTED_CROP_STATES else '✗'}", f)
    report(f"  Districts: Expected {EXPECTED_CROP_DISTRICTS}  |  Actual: {n_districts}  |  {'✓' if n_districts == EXPECTED_CROP_DISTRICTS else '✗'}", f)
    report(f"  Crops:     Expected {EXPECTED_CROP_CROPS}  |  Actual: {n_crops}  |  {'✓' if n_crops == EXPECTED_CROP_CROPS else '✗'}", f)
    report(f"  Seasons:   Expected {EXPECTED_CROP_SEASONS}  |  Actual: {n_seasons}  |  {'✓' if n_seasons == EXPECTED_CROP_SEASONS else '✗'}", f)
    report(f"  Season values: {seasons}", f)

    for name, expected, actual in [("States", EXPECTED_CROP_STATES, n_states),
                                    ("Districts", EXPECTED_CROP_DISTRICTS, n_districts),
                                    ("Crops", EXPECTED_CROP_CROPS, n_crops),
                                    ("Seasons", EXPECTED_CROP_SEASONS, n_seasons)]:
        if actual != expected:
            issues.append(f"{name} count mismatch: expected {expected}, got {actual}")

    # --- 2.5 Null analysis ---
    report(f"\n2.5 Null analysis:", f)
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) == 0:
        report(f"  ✓ Zero null values", f)
    else:
        report(f"  Columns with nulls ({len(cols_with_nulls)}):", f)
        for col, count in cols_with_nulls.items():
            pct = count / len(df) * 100
            report(f"    {col}: {count:,} ({pct:.2f}%)", f)

    # --- 2.6 Known data quality notes ---
    report(f"\n2.6 Data quality checks (documented expectations):", f)
    prod_null_pct = df["Production_Tonnes"].isnull().sum() / len(df) * 100
    report(f"  Production_Tonnes missing: {prod_null_pct:.2f}% (expected ~0.5%)", f)
    yoy_cols = ["Production_YoY_Change_Pct", "Yield_YoY_Change_Pct", "Area_YoY_Change_Pct"]
    for col in yoy_cols:
        if col in df.columns:
            pct = df[col].isnull().sum() / len(df) * 100
            report(f"  {col} missing: {pct:.2f}% (expected ~28% cold-start)", f)

    return issues


def verify_formulas(main_df, crop_df, f):
    """
    Empirically verify undocumented formula columns (B8/B9).
    Recompute from raw components and compare to stored values.
    """
    report("\n" + "=" * 70, f)
    report("SECTION 3: FORMULA VERIFICATION (B8/B9 undocumented columns)", f)
    report("=" * 70, f)

    findings = []

    # -------------------------------------------------------
    # 3.1 Verify Trade_Return_1M
    # Intended formula:
    #   (Value_USD_t - Value_USD_{t-1}) / Value_USD_{t-1}
    #   grouped by (Country, Trade_Type, HS4),
    #   sorted chronologically by (Year, Month)
    # -------------------------------------------------------
    report(f"\n3.1 Verifying Trade_Return_1M:", f)
    report(f"  Intended formula: (Value_USD_t - Value_USD_{{t-1}}) / Value_USD_{{t-1}}", f)
    report(f"  Grouping: (Country, Trade_Type, HS4), sorted by (Year, Month)", f)

    df_sorted = main_df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).copy()

    # Explicitly compute (V_t - V_{t-1}) / V_{t-1} within each chronological group.
    df_sorted["Trade_Return_1M_intended"] = (
        df_sorted.groupby(["Country", "Trade_Type", "HS4"])["Value_USD"]
        .transform(lambda values: (values - values.shift(1)) / values.shift(1))
    )

    mask = (
        df_sorted["Trade_Return_1M"].notna() &
        df_sorted["Trade_Return_1M_intended"].notna() &
        np.isfinite(df_sorted["Trade_Return_1M_intended"])
    )
    if mask.sum() > 0:
        diff = (df_sorted.loc[mask, "Trade_Return_1M"] - df_sorted.loc[mask, "Trade_Return_1M_intended"]).abs()
        n_compared = mask.sum()
        close_pct = (diff < 1e-6).mean() * 100
        close_pct_01 = (diff < 0.01).mean() * 100
        close_pct_1 = (diff < 0.1).mean() * 100
        report(f"  Compared {n_compared:,} rows (excluding first-per-group NaN and inf)", f)
        report(f"  Match within 1e-6:  {close_pct:.2f}%", f)
        report(f"  Match within 0.01:  {close_pct_01:.2f}%", f)
        report(f"  Match within 0.1:   {close_pct_1:.2f}%", f)
        report(f"  Median absolute diff: {diff.median():.8f}", f)
        report(f"  Max absolute diff:    {diff.max():.8f}", f)

        if close_pct == 100:
            finding = "CONFIRMED: Trade_Return_1M matches intended formula (V_t - V_{{t-1}}) / V_{{t-1}}"
        else:
            finding = (f"MISMATCH: Trade_Return_1M does NOT match intended formula "
                       f"(only {close_pct:.1f}% exact match, {close_pct_01:.1f}% within 0.01)")

            # Show 5 sample rows to aid diagnosis
            mismatch_mask = mask & (diff > 0.01)
            if mismatch_mask.sum() > 0:
                sample_idx = df_sorted.loc[mismatch_mask].head(5).index
                report(f"\n  Sample mismatches (first 5):", f)
                for idx in sample_idx:
                    row = df_sorted.loc[idx]
                    report(
                        f"    {row['Country']} | {row['Trade_Type']} | HS4={int(row['HS4'])} | "
                        f"{int(row['Year'])}-{int(row['Month']):02d} | "
                        f"stored={row['Trade_Return_1M']:.8f} | "
                        f"intended={row['Trade_Return_1M_intended']:.8f} | "
                        f"diff={abs(row['Trade_Return_1M'] - row['Trade_Return_1M_intended']):.8f}",
                        f
                    )

        report(f"  -> {finding}", f)
        findings.append(finding)
    else:
        report(f"  No valid comparison rows found", f)
        findings.append("Trade_Return_1M: no valid comparison rows")

    # Also check Trade_Return_3M and Trade_Return_6M for consistency
    for periods, col_name in [(3, "Trade_Return_3M"), (6, "Trade_Return_6M")]:
        report(f"\n  Sub-check: {col_name} = (V_t - V_{{t-{periods}}}) / V_{{t-{periods}}}:", f)
        df_sorted[f"{col_name}_intended"] = (
            df_sorted.groupby(["Country", "Trade_Type", "HS4"])["Value_USD"]
            .pct_change(periods=periods)
        )
        m = (
            df_sorted[col_name].notna() &
            df_sorted[f"{col_name}_intended"].notna() &
            np.isfinite(df_sorted[f"{col_name}_intended"])
        )
        if m.sum() > 0:
            d = (df_sorted.loc[m, col_name] - df_sorted.loc[m, f"{col_name}_intended"]).abs()
            cp = (d < 1e-6).mean() * 100
            report(f"    {m.sum():,} rows | {cp:.2f}% exact match | median_diff={d.median():.8f}", f)
            findings.append(f"{col_name}: {cp:.2f}% match to intended formula")

    # -------------------------------------------------------
    # 3.2 Verify Price_Return_1M
    # Intended formula:
    #   (Unit_Price_USD_per_KG_t - Unit_Price_USD_per_KG_{t-1})
    #     / Unit_Price_USD_per_KG_{t-1}
    #   grouped by (Country, Trade_Type, HS4),
    #   sorted chronologically by (Year, Month)
    # -------------------------------------------------------
    report(f"\n3.2 Verifying Price_Return_1M:", f)
    report(f"  Intended formula: (P_t - P_{{t-1}}) / P_{{t-1}}", f)
    report(f"  where P = Unit_Price_USD_per_KG", f)
    report(f"  Grouping: (Country, Trade_Type, HS4), sorted by (Year, Month)", f)

    df_sorted["Price_Return_1M_intended"] = (
        df_sorted.groupby(["Country", "Trade_Type", "HS4"])["Unit_Price_USD_per_KG"]
        .transform(lambda prices: (prices - prices.shift(1)) / prices.shift(1))
    )

    mask = (
        df_sorted["Price_Return_1M"].notna() &
        df_sorted["Price_Return_1M_intended"].notna() &
        np.isfinite(df_sorted["Price_Return_1M_intended"])
    )
    if mask.sum() > 0:
        diff = (df_sorted.loc[mask, "Price_Return_1M"] - df_sorted.loc[mask, "Price_Return_1M_intended"]).abs()
        n_compared = mask.sum()
        close_pct = (diff < 1e-6).mean() * 100
        close_pct_01 = (diff < 0.01).mean() * 100
        close_pct_1 = (diff < 0.1).mean() * 100
        report(f"  Compared {n_compared:,} rows", f)
        report(f"  Match within 1e-6:  {close_pct:.2f}%", f)
        report(f"  Match within 0.01:  {close_pct_01:.2f}%", f)
        report(f"  Match within 0.1:   {close_pct_1:.2f}%", f)
        report(f"  Median absolute diff: {diff.median():.8f}", f)
        report(f"  Max absolute diff:    {diff.max():.8f}", f)

        if close_pct == 100:
            finding = "CONFIRMED: Price_Return_1M matches intended formula (P_t - P_{{t-1}}) / P_{{t-1}}"
        else:
            finding = (f"MISMATCH: Price_Return_1M does NOT match intended formula "
                       f"(only {close_pct:.1f}% exact match, {close_pct_01:.1f}% within 0.01)")

            mismatch_mask = mask & (diff > 0.01)
            if mismatch_mask.sum() > 0:
                sample_idx = df_sorted.loc[mismatch_mask].head(5).index
                report(f"\n  Sample mismatches (first 5):", f)
                for idx in sample_idx:
                    row = df_sorted.loc[idx]
                    report(
                        f"    {row['Country']} | {row['Trade_Type']} | HS4={int(row['HS4'])} | "
                        f"{int(row['Year'])}-{int(row['Month']):02d} | "
                        f"stored={row['Price_Return_1M']:.8f} | "
                        f"intended={row['Price_Return_1M_intended']:.8f} | "
                        f"P_t={row['Unit_Price_USD_per_KG']:.6f} | "
                        f"P_lag1={row['Price_Lag1']:.6f}",
                        f
                    )

        report(f"  -> {finding}", f)
        findings.append(finding)
    else:
        report(f"  No valid comparison rows found", f)
        findings.append("Price_Return_1M: no valid comparison rows")

    # Also check Price_Return_3M
    report(f"\n  Sub-check: Price_Return_3M = (P_t - P_{{t-3}}) / P_{{t-3}}:", f)
    df_sorted["Price_Return_3M_intended"] = (
        df_sorted.groupby(["Country", "Trade_Type", "HS4"])["Unit_Price_USD_per_KG"]
        .pct_change(periods=3)
    )
    m = (
        df_sorted["Price_Return_3M"].notna() &
        df_sorted["Price_Return_3M_intended"].notna() &
        np.isfinite(df_sorted["Price_Return_3M_intended"])
    )
    if m.sum() > 0:
        d = (df_sorted.loc[m, "Price_Return_3M"] - df_sorted.loc[m, "Price_Return_3M_intended"]).abs()
        cp = (d < 1e-6).mean() * 100
        report(f"    {m.sum():,} rows | {cp:.2f}% exact match | median_diff={d.median():.8f}", f)
        findings.append(f"Price_Return_3M: {cp:.2f}% match to intended formula")

    # -------------------------------------------------------
    # 3.3 Verify Inflation_Change_3M
    # Intended formula:
    #   CPI_Food_Inflation_t - CPI_Food_Inflation_{t-3}
    #   using chronological Year-Month ordering.
    # CPI_Food_Inflation is a national macro variable (same for all
    # Country/HS4 rows in the same Year-Month).
    # -------------------------------------------------------
    report(f"\n3.3 Verifying Inflation_Change_3M:", f)
    report(f"  Intended formula: CPI_Food_Inflation_t - CPI_Food_Inflation_{{t-3}}", f)
    report(f"  Computed using calendar Year-Month offsets on the deduplicated macro series", f)

    macro = main_df.drop_duplicates(subset=["Year", "Month"]).sort_values(["Year", "Month"]).copy()
    macro["Year_Month"] = pd.to_datetime(
        macro["Year"].astype(str) + "-" + macro["Month"].astype(str) + "-01"
    ).dt.to_period("M")
    inflation_by_month = macro.set_index("Year_Month")["CPI_Food_Inflation"]
    macro["Inflation_Change_3M_intended"] = (
        macro["CPI_Food_Inflation"]
        - (macro["Year_Month"] - 3).map(inflation_by_month)
    )

    mask = (
        macro["Inflation_Change_3M"].notna() &
        macro["Inflation_Change_3M_intended"].notna()
    )
    if mask.sum() > 0:
        diff = (macro.loc[mask, "Inflation_Change_3M"] - macro.loc[mask, "Inflation_Change_3M_intended"]).abs()
        close_pct = (diff < 1e-6).mean() * 100
        close_pct_01 = (diff < 0.01).mean() * 100
        close_pct_1 = (diff < 0.1).mean() * 100
        report(f"  Compared {mask.sum()} unique year-month rows", f)
        report(f"  Match within 1e-6:  {close_pct:.2f}%", f)
        report(f"  Match within 0.01:  {close_pct_01:.2f}%", f)
        report(f"  Match within 0.1:   {close_pct_1:.2f}%", f)
        report(f"  Median absolute diff: {diff.median():.8f}", f)
        report(f"  Max absolute diff:    {diff.max():.8f}", f)

        if close_pct == 100:
            finding = "CONFIRMED: Inflation_Change_3M matches CPI_Food_Inflation_t - CPI_Food_Inflation_{{t-3}}"
        else:
            finding = (f"MISMATCH: Inflation_Change_3M does NOT match intended formula "
                       f"(only {close_pct:.1f}% exact, {close_pct_01:.1f}% within 0.01)")

            # Print all values for manual inspection (only ~92 rows)
            report(f"\n  Full comparison table ({mask.sum()} rows):", f)
            report(f"  {'Year':>4} {'Month':>5} {'CPI_Infl':>10} {'Stored':>12} {'Intended':>12} {'Diff':>12}", f)
            for _, row in macro[mask].iterrows():
                report(
                    f"  {int(row['Year']):>4} {int(row['Month']):>5} "
                    f"{row['CPI_Food_Inflation']:>10.4f} "
                    f"{row['Inflation_Change_3M']:>12.4f} "
                    f"{row['Inflation_Change_3M_intended']:>12.4f} "
                    f"{abs(row['Inflation_Change_3M'] - row['Inflation_Change_3M_intended']):>12.6f}",
                    f
                )

        report(f"  -> {finding}", f)
        findings.append(finding)
    else:
        report(f"  No valid comparison rows", f)
        findings.append("Inflation_Change_3M: no valid comparison rows")

    # -------------------------------------------------------
    # 3.4 Verify Price_Lag1
    # Hypothesis: Unit_Price_USD_per_KG shifted by 1 period
    # -------------------------------------------------------
    report(f"\n3.4 Verifying Price_Lag1:", f)
    report(f"  Hypothesis: Unit_Price_USD_per_KG.shift(1) grouped by (Country, Trade_Type, HS4)", f)

    df_sorted["Price_Lag1_recomputed"] = (
        df_sorted.groupby(["Country", "Trade_Type", "HS4"])["Unit_Price_USD_per_KG"]
        .shift(1)
    )

    mask = (
        df_sorted["Price_Lag1"].notna() &
        df_sorted["Price_Lag1_recomputed"].notna() &
        (df_sorted["Price_Lag1"] != 0)
    )
    if mask.sum() > 0:
        diff = (df_sorted.loc[mask, "Price_Lag1"] - df_sorted.loc[mask, "Price_Lag1_recomputed"]).abs()
        close_pct = (diff < 0.001).mean() * 100
        report(f"  Compared {mask.sum():,} rows", f)
        report(f"  Rows within 0.001 tolerance: {close_pct:.2f}%", f)
        report(f"  Median absolute diff: {diff.median():.8f}", f)
        if close_pct > 95:
            finding = "CONFIRMED: Price_Lag1 ≈ Unit_Price_USD_per_KG.shift(1)"
        else:
            finding = f"DISCREPANCY: Price_Lag1 does NOT match simple lag ({close_pct:.1f}%)"
        report(f"  → {finding}", f)
        findings.append(finding)
    else:
        report(f"  ⚠ No valid comparison rows", f)
        findings.append("Price_Lag1: no valid comparison rows")

    # -------------------------------------------------------
    # 3.5 Verify documented formulas (B6) — Effective_Shock, Log_Value_USD
    # (These are documented but worth confirming implementation matches)
    # -------------------------------------------------------
    report(f"\n3.5 Cross-checking documented formulas (B6):", f)

    # Effective_Shock = Shock_Intensity × Trade_Share
    df_sorted["Effective_Shock_recomputed"] = df_sorted["Shock_Intensity"] * df_sorted["Trade_Share"]
    mask = df_sorted["Effective_Shock"].notna() & df_sorted["Effective_Shock_recomputed"].notna()
    diff = (df_sorted.loc[mask, "Effective_Shock"] - df_sorted.loc[mask, "Effective_Shock_recomputed"]).abs()
    close_pct = (diff < 0.001).mean() * 100
    report(f"  Effective_Shock = Shock_Intensity × Trade_Share: {close_pct:.2f}% match (within 0.001)", f)
    findings.append(f"Effective_Shock formula: {close_pct:.2f}% match")

    # Log_Value_USD = ln(1 + Value_USD)
    df_sorted["Log_Value_USD_recomputed"] = np.log1p(df_sorted["Value_USD"])
    mask = df_sorted["Log_Value_USD"].notna() & df_sorted["Log_Value_USD_recomputed"].notna()
    diff = (df_sorted.loc[mask, "Log_Value_USD"] - df_sorted.loc[mask, "Log_Value_USD_recomputed"]).abs()
    close_pct = (diff < 0.001).mean() * 100
    report(f"  Log_Value_USD = ln(1 + Value_USD): {close_pct:.2f}% match (within 0.001)", f)
    findings.append(f"Log_Value_USD formula: {close_pct:.2f}% match")

    # -------------------------------------------------------
    # 3.6 Verify crop dataset: Production_YoY_Change_Pct (B9)
    # -------------------------------------------------------
    report(f"\n3.6 Verifying Production_YoY_Change_Pct (Crop dataset, B9):", f)
    report(f"  Hypothesis: grouped by (State, District, Crop, Season), sorted by Crop_Year", f)

    crop_sorted = crop_df.sort_values(["State", "District", "Crop", "Season", "Crop_Year"]).copy()
    grp = crop_sorted.groupby(["State", "District", "Crop", "Season"])

    # Recompute: (Prod_t - Prod_t-1) / Prod_t-1 * 100
    prev_prod = grp["Production_Tonnes"].shift(1)
    crop_sorted["YoY_recomputed"] = (
        (crop_sorted["Production_Tonnes"] - prev_prod) / prev_prod * 100
    )

    mask = (
        crop_sorted["Production_YoY_Change_Pct"].notna() &
        crop_sorted["YoY_recomputed"].notna() &
        np.isfinite(crop_sorted["YoY_recomputed"])
    )
    if mask.sum() > 0:
        diff = (crop_sorted.loc[mask, "Production_YoY_Change_Pct"] - crop_sorted.loc[mask, "YoY_recomputed"]).abs()
        close_pct = (diff < 0.1).mean() * 100
        report(f"  Compared {mask.sum():,} rows", f)
        report(f"  Rows within 0.1 tolerance: {close_pct:.2f}%", f)
        report(f"  Median absolute diff: {diff.median():.6f}", f)
        if close_pct > 95:
            finding = "CONFIRMED: Production_YoY_Change_Pct ≈ (Prod_t - Prod_{t-1}) / Prod_{t-1} × 100, grouped by (State, District, Crop, Season)"
        else:
            # Try coarser grouping
            report(f"  Low match with (State, District, Crop, Season). Trying (Crop, Season)...", f)
            grp2 = crop_sorted.groupby(["Crop", "Season"])
            prev_prod2 = grp2["Production_Tonnes"].shift(1)
            crop_sorted["YoY_recomputed_v2"] = (
                (crop_sorted["Production_Tonnes"] - prev_prod2) / prev_prod2 * 100
            )
            mask2 = (
                crop_sorted["Production_YoY_Change_Pct"].notna() &
                crop_sorted["YoY_recomputed_v2"].notna() &
                np.isfinite(crop_sorted["YoY_recomputed_v2"])
            )
            if mask2.sum() > 0:
                diff2 = (crop_sorted.loc[mask2, "Production_YoY_Change_Pct"] - crop_sorted.loc[mask2, "YoY_recomputed_v2"]).abs()
                close_pct2 = (diff2 < 0.1).mean() * 100
                report(f"  With (Crop, Season): {close_pct2:.2f}% match", f)

            finding = f"NEEDS INVESTIGATION: Production_YoY_Change_Pct match is {close_pct:.1f}% with finest grouping"
        report(f"  → {finding}", f)
        findings.append(finding)
    else:
        report(f"  ⚠ No valid comparison rows", f)
        findings.append("Production_YoY_Change_Pct: no valid comparison rows")

    # -------------------------------------------------------
    # 3.7 Verify Shock_Trend_3M (undocumented)
    # -------------------------------------------------------
    report(f"\n3.7 Verifying Shock_Trend_3M (undocumented, B8):", f)
    report(f"  Testing hypotheses: rolling mean, diff, or slope of Shock_Intensity over 3 months", f)

    # Hypothesis A: 3-month rolling mean of Shock_Intensity (national-level)
    macro_sorted = df_sorted.drop_duplicates(subset=["Country", "Year", "Month"]).sort_values(["Country", "Year", "Month"]).copy()
    macro_sorted["Shock_Trend_3M_hyp_rollmean"] = (
        macro_sorted.groupby("Country")["Shock_Intensity"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    mask_a = macro_sorted["Shock_Trend_3M"].notna() & macro_sorted["Shock_Trend_3M_hyp_rollmean"].notna()
    if mask_a.sum() > 0:
        diff_a = (macro_sorted.loc[mask_a, "Shock_Trend_3M"] - macro_sorted.loc[mask_a, "Shock_Trend_3M_hyp_rollmean"]).abs()
        close_a = (diff_a < 0.1).mean() * 100
        report(f"  Hypothesis A (rolling 3M mean of Shock_Intensity by Country): {close_a:.2f}% match", f)

    # Hypothesis B: 3-month difference
    macro_sorted["Shock_Trend_3M_hyp_diff"] = (
        macro_sorted.groupby("Country")["Shock_Intensity"]
        .transform(lambda x: x.diff(3))
    )
    mask_b = macro_sorted["Shock_Trend_3M"].notna() & macro_sorted["Shock_Trend_3M_hyp_diff"].notna()
    if mask_b.sum() > 0:
        diff_b = (macro_sorted.loc[mask_b, "Shock_Trend_3M"] - macro_sorted.loc[mask_b, "Shock_Trend_3M_hyp_diff"]).abs()
        close_b = (diff_b < 0.1).mean() * 100
        report(f"  Hypothesis B (3-month diff of Shock_Intensity by Country): {close_b:.2f}% match", f)

    best_hyp = "unknown"
    if mask_a.sum() > 0 and mask_b.sum() > 0:
        if close_a > close_b:
            best_hyp = f"rolling 3M mean ({close_a:.1f}%)" if close_a > 90 else f"partial: rolling 3M mean ({close_a:.1f}%)"
        else:
            best_hyp = f"3M diff ({close_b:.1f}%)" if close_b > 90 else f"partial: 3M diff ({close_b:.1f}%)"
    finding = f"Shock_Trend_3M: best hypothesis = {best_hyp}"
    report(f"  → {finding}", f)
    findings.append(finding)

    # -------------------------------------------------------
    # 3.8 Verify Agri_GVA_Growth_Percent
    # Intended formula:
    #   (GVA_t - GVA_{t-1}) / GVA_{t-1}
    # GVA is annual, so "t-1" means the previous year.  Direct verification
    # requires annual GVA levels as well as the stored growth-rate column.
    # -------------------------------------------------------
    report(f"\n3.8 Verifying Agri_GVA_Growth_Percent:", f)
    report(f"  Intended formula: (GVA_t - GVA_{{t-1}}) / GVA_{{t-1}}", f)
    report(f"  Direct comparison requires annual GVA levels for years t and t-1.", f)
    report(f"  Annual constancy and lag consistency are diagnostic checks only.", f)

    # Check 1: Is it constant within each year?
    gva_per_year = main_df.groupby("Year")["Agri_GVA_Growth_Percent"].nunique()
    all_constant = (gva_per_year == 1).all()
    report(f"  Constant within each year: {'YES' if all_constant else 'NO'}", f)
    report(f"  Unique values per year: {gva_per_year.to_dict()}", f)

    # Check 2: Does Agri_GVA_Lag1 = Agri_GVA_Growth_Percent shifted by 1 year?
    annual = main_df.drop_duplicates(subset=["Year"]).sort_values("Year").copy()
    annual["Agri_GVA_Lag1_intended"] = annual["Agri_GVA_Growth_Percent"].shift(1)
    mask_gva = annual["Agri_GVA_Lag1"].notna() & annual["Agri_GVA_Lag1_intended"].notna()
    if mask_gva.sum() > 0:
        diff_gva = (annual.loc[mask_gva, "Agri_GVA_Lag1"] - annual.loc[mask_gva, "Agri_GVA_Lag1_intended"]).abs()
        close_gva = (diff_gva < 1e-6).mean() * 100
        report(f"  Agri_GVA_Lag1 == shift(1) of growth rate: {close_gva:.2f}% match", f)
        report(f"  Median diff: {diff_gva.median():.8f}", f)
    else:
        close_gva = 0
        report(f"  Agri_GVA_Lag1: no valid comparison rows", f)

    # Show the stored values
    report(f"\n  Stored values by year:", f)
    report(f"  {'Year':>6} {'Growth%':>10} {'Lag1':>10}", f)
    for _, row in annual.iterrows():
        report(f"  {int(row['Year']):>6} {row['Agri_GVA_Growth_Percent']:>10.4f} {row['Agri_GVA_Lag1']:>10.4f}", f)

    if all_constant:
        finding = (f"Agri_GVA_Growth_Percent: Annual variable, constant per year. "
                   f"Lag1 match: {close_gva:.1f}%. "
                   f"Raw GVA level absent — growth rate is the stored representation of (GVA_t-GVA_{{t-1}})/GVA_{{t-1}}.")
    else:
        finding = "Agri_GVA_Growth_Percent: NOT constant per year — unexpected."
    report(f"  -> {finding}", f)
    findings.append(finding)

    gva_level_columns = [
        col for col in ["GVA", "Agri_GVA", "Agriculture_GVA"]
        if col in main_df.columns
    ]
    if gva_level_columns:
        gva_col = gva_level_columns[0]
        annual["Agri_GVA_Growth_intended"] = annual[gva_col].pct_change()
        mask_gva_formula = (
            annual["Agri_GVA_Growth_Percent"].notna()
            & annual["Agri_GVA_Growth_intended"].notna()
            & np.isfinite(annual["Agri_GVA_Growth_intended"])
        )
        if mask_gva_formula.sum() > 0:
            diff_gva_formula = (
                annual.loc[mask_gva_formula, "Agri_GVA_Growth_Percent"]
                - annual.loc[mask_gva_formula, "Agri_GVA_Growth_intended"]
            ).abs()
            close_gva_formula = (diff_gva_formula < 1e-6).mean() * 100
            formula_finding = (
                f"{'CONFIRMED' if close_gva_formula > 95 else 'MISMATCH'}: "
                f"Agri_GVA_Growth_Percent "
                f"{'matches' if close_gva_formula > 95 else 'does NOT match'} "
                f"(GVA_t - GVA_{{t-1}}) / GVA_{{t-1}} using {gva_col} "
                f"({close_gva_formula:.2f}% exact match)."
            )
        else:
            formula_finding = "Agri_GVA_Growth_Percent: no valid GVA-level comparison rows."
    else:
        formula_finding = (
            "NOT VERIFIABLE: Agri_GVA_Growth_Percent cannot be compared to "
            "(GVA_t - GVA_{t-1}) / GVA_{t-1} because this dataset has no annual GVA-level column."
        )
    report(f"  Formula status: {formula_finding}", f)
    findings.append(formula_finding)

    return findings


def main():
    print("=" * 70)
    print("Drishti — Task 1: Data Validation & Formula Verification")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load datasets
    print(f"\nLoading main dataset: {MAIN_CSV}")
    main_df = pd.read_csv(MAIN_CSV)
    print(f"  Shape: {main_df.shape}")

    print(f"\nLoading crop dataset: {CROP_CSV}")
    crop_df = pd.read_csv(CROP_CSV)
    print(f"  Shape: {crop_df.shape}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        report(f"Drishti — Task 1: Data Validation Report", f)
        report(f"Generated: {datetime.now().isoformat()}", f)
        report(f"Random State: {RANDOM_STATE}", f)

        # Section 1: Main dataset
        main_issues = validate_main_dataset(main_df, f)

        # Section 2: Crop dataset
        crop_issues = validate_crop_dataset(crop_df, f)

        # Section 3: Formula verification
        formula_findings = verify_formulas(main_df, crop_df, f)

        # Summary
        report("\n" + "=" * 70, f)
        report("SECTION 4: OVERALL SUMMARY", f)
        report("=" * 70, f)

        all_issues = main_issues + crop_issues
        if all_issues:
            report(f"\n⚠ {len(all_issues)} schema discrepancies found:", f)
            for i, issue in enumerate(all_issues, 1):
                report(f"  {i}. {issue}", f)
        else:
            report(f"\n✓ All schema checks passed — both datasets match Part C documentation", f)

        report(f"\nFormula verification findings ({len(formula_findings)}):", f)
        for i, finding in enumerate(formula_findings, 1):
            report(f"  {i}. {finding}", f)

        report(f"\nReport saved to: {REPORT_FILE}", f)

    print(f"\n{'=' * 70}")
    print(f"Task 1 complete. Full report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
