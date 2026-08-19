import pandas as pd
import numpy as np

# ======================================================
# 1. LOAD DATA
# ======================================================
# Changed to read_excel to match your new file format
file_name = "derived_features.xlsx"

print(f"Loading {file_name}...")
# Note: Ensure you have 'openpyxl' installed (pip install openpyxl)
df = pd.read_excel(file_name, engine='openpyxl')

# --- MANDATORY COLUMN ALIGNMENT ---
# Renaming columns from your uploaded file to match the script logic
df = df.rename(columns={
    "Country_CLEAN": "Country",
    "Trade Type": "Trade_Type"
})

# Ensure correct dtypes
df["Year"] = df["Year"].astype(int)
df["Month"] = df["Month"].astype(int)
df["HS4"] = df["HS4"].astype(int)

# Sort for all time-based operations
df = df.sort_values(
    by=["Country", "Trade_Type", "HS4", "Year", "Month"]
).reset_index(drop=True)

# ======================================================
# 2. TOTAL COUNTRY TRADE (AGRICULTURAL SECTOR)
# ======================================================
print("Calculating trade shares...")
df["Total_Country_Trade_USD"] = (
    df.groupby(["Country", "Year", "Month", "Trade_Type"])["Value_USD"]
      .transform("sum")
)

# ======================================================
# 3. TRADE SHARE (ABSOLUTE VALUE ONLY)
# ======================================================
df["Trade_Share"] = df["Value_USD"] / df["Total_Country_Trade_USD"]
df["Trade_Share"] = (
    df["Trade_Share"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

# ======================================================
# 4. SHOCK × EXPOSURE (CORE FEATURES)
# ======================================================
print("Generating exposure metrics...")
df["Effective_Shock"] = df["Shock_Intensity"] * df["Trade_Share"]

df["Conflict_Exposure"] = df["Conflict_Density"] * df["Trade_Share"]
df["Protest_Exposure"]  = df["Protest_Density"]  * df["Trade_Share"]
df["Trade_Shock_Exposure"] = df["Trade_Shock_Density"] * df["Trade_Share"]

# ======================================================
# 5. DIRECTIONAL SHOCK EXPOSURE
# ======================================================
df["Incoming_Shock_Exposure"] = df["Incoming_Shock_Count"] * df["Trade_Share"]
df["Outgoing_Shock_Exposure"] = df["Outgoing_Shock_Count"] * df["Trade_Share"]
df["Net_Hostility_Exposure"]  = df["Net_Hostility"] * df["Trade_Share"]

# ======================================================
# 6. LOG TRANSFORM FOR STABILITY
# ======================================================
df["Log_Value_USD"] = np.log1p(df["Value_USD"])

# ======================================================
# 7. LAGGED MACRO SHOCKS (TIME-SAFE)
# ======================================================
print("Computing lags...")
df = df.sort_values(["Country", "Year", "Month"])

df["Shock_Intensity_Lag1"] = (
    df.groupby("Country")["Shock_Intensity"].shift(1)
)
df["Shock_Intensity_Lag2"] = (
    df.groupby("Country")["Shock_Intensity"].shift(2)
)

# ======================================================
# 8. LAGGED TRADE SHARE (HS4 + TRADE TYPE SAFE)
# ======================================================
df = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"])

df["Trade_Share_Lag1"] = (
    df.groupby(["Country", "Trade_Type", "HS4"])["Trade_Share"].shift(1)
)
df["Trade_Share_Lag2"] = (
    df.groupby(["Country", "Trade_Type", "HS4"])["Trade_Share"].shift(2)
)

# ======================================================
# 9. LAGGED EFFECTIVE SHOCK
# ======================================================
df["Lagged_Effective_Shock_1"] = (
    df["Shock_Intensity_Lag1"] * df["Trade_Share_Lag1"]
)
df["Lagged_Effective_Shock_2"] = (
    df["Shock_Intensity_Lag2"] * df["Trade_Share_Lag2"]
)

# Fill NaNs caused by lagging
lag_cols = [
    "Shock_Intensity_Lag1", "Shock_Intensity_Lag2", 
    "Trade_Share_Lag1", "Trade_Share_Lag2",
    "Lagged_Effective_Shock_1", "Lagged_Effective_Shock_2"
]
df[lag_cols] = df[lag_cols].fillna(0)

# ======================================================
# 10. SAVE FINAL DATASET
# ======================================================
output_file = "drishti_final_with_all_derived_features.csv"
df.to_csv(output_file, index=False)

print("\n✅ SUCCESS")
print(f"• Final feature set saved to {output_file}")
print(f"• Total Rows: {len(df)}")