# import pandas as pd

# # -------------------------------------------------
# # Load datasets
# # -------------------------------------------------

# trade_df = pd.read_csv("Final_Drishti_2018_2025.csv")
# gdelt_df = pd.read_csv("gdelt_monthly_features_with_country_names.csv")

# # -------------------------------------------------
# # Ensure join keys are clean
# # -------------------------------------------------

# for df in [trade_df, gdelt_df]:
#     df["Year"] = df["Year"].astype(int)
#     df["Month"] = df["Month"].astype(int)
#     df["Country"] = df["Country"].astype(str).str.strip()

# # -------------------------------------------------
# # LEFT JOIN (Trade × GDELT)
# # -------------------------------------------------

# master_df = trade_df.merge(
#     gdelt_df,
#     on=["Country", "Year", "Month"],
#     how="left"
# )

# # -------------------------------------------------
# # Handle missing GDELT months (no events)
# # -------------------------------------------------

# gdelt_feature_cols = [
#     "Total_Event_Count",
#     "Avg_Goldstein",
#     "Avg_Tone",
#     "Shock_Intensity",
#     "Conflict_Event_Count",
#     "Protest_Event_Count",
#     "Trade_Shock_Count",
#     "Sanction_Threat_Count",
#     "Incoming_Shock_Count",
#     "Outgoing_Shock_Count",
#     "Net_Hostility",
#     "Conflict_Density",
#     "Protest_Density",
#     "Trade_Shock_Density",
#     "Sanction_Threat_Density"
# ]

# master_df[gdelt_feature_cols] = master_df[gdelt_feature_cols].fillna(0)

# # -------------------------------------------------
# # Save final modeling table
# # -------------------------------------------------

# master_df.to_csv("master_trade_geopolitics_dataset.csv", index=False)

# print("✅ Master modeling table created")
# print("Rows:", len(master_df))
# print("Columns:", master_df.shape[1])



# import pandas as pd

# # =====================================================
# # CONFIGURATION
# # =====================================================
# TRADE_FILE = pd.read_csv("Final_Drishti_2018_2025.csv")
# GDELT_FILE = pd.read_csv("gdelt_monthly_features_with_country_names.csv")

# # TRADE_FILE = "trade_fact_table_2018_2025.csv"
# # GDELT_FILE = "gdelt_monthly_features_with_country_names.csv"
# OUTPUT_FILE = "combined_final_trade_gdelt_dataset.csv"

# # =====================================================
# # LOAD DATA
# # =====================================================

# print("📥 Loading datasets...")
# trade_df = pd.read_csv(TRADE_FILE)
# gdelt_df = pd.read_csv(GDELT_FILE)

# # Ensure correct dtypes
# trade_df["Year"] = trade_df["Year"].astype(int)
# trade_df["Month"] = trade_df["Month"].astype(int)

# gdelt_df["Year"] = gdelt_df["Year"].astype(int)
# gdelt_df["Month"] = gdelt_df["Month"].astype(int)

# # =====================================================
# # GDELT FEATURE COLUMNS (ACTUAL, VERIFIED)
# # =====================================================

# gdelt_feature_cols = [
#     "Total_Event_Count",
#     "Avg_Goldstein",
#     "Avg_Tone",
#     "Total_Mentions",
#     "Total_Sources",
#     "Shock_Intensity",
#     "Conflict_Event_Count",
#     "Protest_Event_Count",
#     "Trade_Shock_Count",
#     "Sanction_Threat_Count",
#     "Incoming_Shock_Count",
#     "Outgoing_Shock_Count",
#     "Net_Hostility",
#     "Conflict_Density",
#     "Protest_Density"
# ]

# # Keep only needed columns from GDELT
# gdelt_df = gdelt_df[
#     ["Country", "Year", "Month"] + gdelt_feature_cols
# ]

# # =====================================================
# # MERGE (LEFT JOIN — TRADE IS BASE)
# # =====================================================

# print("🔗 Merging trade data with GDELT features...")

# merged_df = trade_df.merge(
#     gdelt_df,
#     on=["Country", "Year", "Month"],
#     how="left"
# )

# # =====================================================
# # HANDLE MISSING GDELT VALUES
# # (No events → zero shock)
# # =====================================================

# merged_df[gdelt_feature_cols] = merged_df[gdelt_feature_cols].fillna(0)

# # =====================================================
# # FINAL CHECKS
# # =====================================================

# print("\n✅ Merge complete")
# print("Final shape:", merged_df.shape)

# print("\n📊 Sample rows:")
# print(merged_df.head())

# # =====================================================
# # SAVE OUTPUT
# # =====================================================

# merged_df.to_csv(OUTPUT_FILE, index=False)

# print(f"\n💾 Saved final dataset → {OUTPUT_FILE}")







import pandas as pd
import os

# =====================================================
# CONFIGURATION (STRING S ONLY — DO NOT TOUCH BELOW)
# =====================================================

TRADE_FILE_PATH = "Trade_Top15_Normalized.csv"
GDELT_FILE_PATH = "gdelt_monthly_features_with_country_names.csv"
OUTPUT_FILE_PATH = "yashwanths_s_final.csv"

# =====================================================
# HARD SAFETY CHECKS (CATCH BUGS EARLY)
# =====================================================

if not isinstance(TRADE_FILE_PATH, str):
    raise TypeError("TRADE_FILE_PATH must be a string path")

if not isinstance(GDELT_FILE_PATH, str):
    raise TypeError("GDELT_FILE_PATH must be a string path")

print("📁 Checking input files...")
print("Trade file:", TRADE_FILE_PATH)
print("GDELT file:", GDELT_FILE_PATH)

if not os.path.exists(TRADE_FILE_PATH):
    raise FileNotFoundError(f"Trade file not found: {TRADE_FILE_PATH}")

if not os.path.exists(GDELT_FILE_PATH):
    raise FileNotFoundError(f"GDELT file not found: {GDELT_FILE_PATH}")

# =====================================================
# LOAD DATA (NO VARIABLE REUSE)
# =====================================================

print("\n📥 Loading trade dataset...")
trade_df = pd.read_csv(TRADE_FILE_PATH)

print("📥 Loading GDELT dataset...")
gdelt_df = pd.read_csv(GDELT_FILE_PATH)

# =====================================================
# TYPE NORMALIZATION
# =====================================================

trade_df["Year"] = trade_df["Year"].astype(int)
trade_df["Month"] = trade_df["Month"].astype(int)

gdelt_df["Year"] = gdelt_df["Year"].astype(int)
gdelt_df["Month"] = gdelt_df["Month"].astype(int)

# =====================================================
# VERIFIED GDELT FEATURE COLUMNS (MATCH YOUR DATA)
# =====================================================

gdelt_feature_cols = [
    "Total_Event_Count",
    "Avg_Goldstein",
    "Avg_Tone",
    "Total_Mentions",
    "Total_Sources",
    "Shock_Intensity",
    "Conflict_Event_Count",
    "Protest_Event_Count",
    "Trade_Shock_Count",
    "Sanction_Threat_Count",
    "Incoming_Shock_Count",
    "Outgoing_Shock_Count",
    "Net_Hostility",
    "Conflict_Density",
    "Protest_Density"
]

# Keep only required columns
gdelt_df = gdelt_df[["Country", "Year", "Month"] + gdelt_feature_cols]

# =====================================================
# MERGE (TRADE × GDELT — MONTHLY COUNTRY JOIN)
# =====================================================

print("\n🔗 Merging datasets...")

final_df = trade_df.merge(
    gdelt_df,
    on=["Country", "Year", "Month"],
    how="left"
)

# =====================================================
# FILL MISSING GDELT VALUES
# (No geopolitical event → zero shock)
# =====================================================

final_df[gdelt_feature_cols] = final_df[gdelt_feature_cols].fillna(0)

# =====================================================
# SAVE OUTPUT
# =====================================================

final_df.to_csv(OUTPUT_FILE_PATH, index=False)

print("\n✅ SUCCESS")
print("Final rows:", len(final_df))
print("Saved file:", OUTPUT_FILE_PATH)
