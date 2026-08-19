"""
Drishti — Task 1 follow-up 2: Investigate remaining undocumented columns.
Focus on Trade_Return_1M pattern (shifted log return?), Shock_Trend_3M (diff(1)?),
and Inflation_Change_3M (lag-based comparison?).
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")

print("Loading main dataset...")
df = pd.read_csv(MAIN_CSV)

# =========================================================
# INVESTIGATION 1: Trade_Return_1M — closer look at the sample
# =========================================================
print("\n" + "=" * 70)
print("TRADE_RETURN_1M: Inspecting row-by-row")
print("=" * 70)

# The sample showed that row N's Trade_Return_1M = row N+1's log_return
# This suggests the column was computed as log return, but then the data
# was re-sorted (maybe differently) after computation, shifting values.
# OR it was computed at a different grouping level.

# Let's look at the actual values in context
sample = df[(df["Country"] == "CHINA") & (df["Trade_Type"] == "Import") & (df["HS4"] == 1302)].copy()
sample = sample.sort_values(["Year", "Month"]).reset_index(drop=True)
sample["log_val"] = np.log1p(sample["Value_USD"])
sample["log_diff"] = sample["log_val"].diff(1)

print("\nCHINA / Import / HS4=1302:")
print(sample[["Year", "Month", "Value_USD", "Log_Value_USD", "Trade_Return_1M", "log_diff"]].head(15).to_string(index=False))

# Check: is Trade_Return_1M of row i == log_diff of row i+1?
sample["Trade_Return_1M_shifted"] = sample["Trade_Return_1M"].shift(-1)
mask = sample["log_diff"].notna() & sample["Trade_Return_1M_shifted"].notna()
diff = (sample.loc[mask, "log_diff"] - sample.loc[mask, "Trade_Return_1M_shifted"]).abs()
close = (diff < 0.001).mean() * 100
print(f"\nTrade_Return_1M[i] == log_diff[i+1]? {close:.1f}% match")

# Maybe the column is just the log return with different sort order
# Let's check: is it the log return of the NEXT row in the original data order?
# Or check if it's lagged differently

# Actually let's read a fresh copy without sorting and check
df_orig = pd.read_csv(MAIN_CSV)
print(f"\nOriginal data first 5 rows:")
print(df_orig[["Year", "Month", "Country", "Trade_Type", "HS4", "Value_USD", "Log_Value_USD", "Trade_Return_1M"]].head(5).to_string(index=False))

# Compute log_diff on original order
df_orig["log_diff_orig"] = np.log1p(df_orig["Value_USD"]).diff(1)
mask = df_orig["Trade_Return_1M"].notna() & df_orig["log_diff_orig"].notna()
diff = (df_orig.loc[mask, "Trade_Return_1M"] - df_orig.loc[mask, "log_diff_orig"]).abs()
close = (diff < 0.001).mean() * 100
print(f"\nTrade_Return_1M == log_diff on ORIGINAL row order: {close:.1f}% match")
print(f"  Median diff: {diff.median():.6f}")

# What if it's grouped by (Country, HS4) only (no Trade_Type)?
df_s = df_orig.sort_values(["Country", "HS4", "Year", "Month"]).copy()
df_s["log_diff_ch"] = df_s.groupby(["Country", "HS4"])["Log_Value_USD"].diff(1)
mask = df_s["Trade_Return_1M"].notna() & df_s["log_diff_ch"].notna()
diff = (df_s.loc[mask, "Trade_Return_1M"] - df_s.loc[mask, "log_diff_ch"]).abs()
close = (diff < 0.001).mean() * 100
print(f"\nGrouped by (Country, HS4): {close:.1f}% match")

# What about (Country, Trade_Type, Commodity)?
df_s = df_orig.sort_values(["Country", "Trade_Type", "Commodity", "Year", "Month"]).copy()
df_s["log_diff_ctc"] = df_s.groupby(["Country", "Trade_Type", "Commodity"])["Log_Value_USD"].diff(1)
mask = df_s["Trade_Return_1M"].notna() & df_s["log_diff_ctc"].notna()
diff = (df_s.loc[mask, "Trade_Return_1M"] - df_s.loc[mask, "log_diff_ctc"]).abs()
close = (diff < 0.001).mean() * 100
print(f"\nGrouped by (Country, Trade_Type, Commodity): {close:.1f}% match")

# Maybe the grouping involves Country_HS4?
df_s = df_orig.sort_values(["Country_HS4", "Trade_Type", "Year", "Month"]).copy()
df_s["log_diff_chs4"] = df_s.groupby(["Country_HS4", "Trade_Type"])["Log_Value_USD"].diff(1)
mask = df_s["Trade_Return_1M"].notna() & df_s["log_diff_chs4"].notna()
diff = (df_s.loc[mask, "Trade_Return_1M"] - df_s.loc[mask, "log_diff_chs4"]).abs()
close = (diff < 0.001).mean() * 100
print(f"\nGrouped by (Country_HS4, Trade_Type): {close:.1f}% match")

# =========================================================
# INVESTIGATION 2: Shock_Trend_3M — diff(1)
# =========================================================
print("\n" + "=" * 70)
print("SHOCK_TREND_3M: Testing diff(1)")
print("=" * 70)

macro = df_orig.drop_duplicates(subset=["Country", "Year", "Month"]).sort_values(["Country", "Year", "Month"]).copy()
macro["shock_diff1"] = macro.groupby("Country")["Shock_Intensity"].diff(1)

mask = macro["Shock_Trend_3M"].notna() & macro["shock_diff1"].notna()
diff = (macro.loc[mask, "Shock_Trend_3M"] - macro.loc[mask, "shock_diff1"]).abs()
close = (diff < 0.001).mean() * 100
print(f"  Shock_Intensity.diff(1) by Country: {close:.1f}% match (median={diff.median():.4f})")

# Check if first row per country equals Shock_Intensity itself (the sample showed this)
first_rows = macro.groupby("Country").first()
diff_first = (first_rows["Shock_Trend_3M"] - first_rows["Shock_Intensity"]).abs()
close_first = (diff_first < 0.001).mean() * 100
print(f"  First row per country: Shock_Trend_3M == Shock_Intensity? {close_first:.1f}% match")

# Hypothesis: Shock_Trend_3M for first row = Shock_Intensity, then diff(1) afterwards
# This would be consistent with fillna(Shock_Intensity) after diff(1)
# Let's create that:
macro["shock_trend_hyp"] = macro.groupby("Country")["Shock_Intensity"].diff(1)
# Fill NaN (first row per group) with the actual Shock_Intensity
mask_na = macro["shock_trend_hyp"].isna()
macro.loc[mask_na, "shock_trend_hyp"] = macro.loc[mask_na, "Shock_Intensity"]

mask = macro["Shock_Trend_3M"].notna() & macro["shock_trend_hyp"].notna()
diff = (macro.loc[mask, "Shock_Trend_3M"] - macro.loc[mask, "shock_trend_hyp"]).abs()
close = (diff < 0.001).mean() * 100
print(f"  diff(1) + fillna(Shock_Intensity): {close:.1f}% match")

# =========================================================
# INVESTIGATION 3: Inflation_Change_3M — look at Inflation_Lag1 relationship
# =========================================================
print("\n" + "=" * 70)
print("INFLATION_CHANGE_3M: Deeper look")
print("=" * 70)

macro2 = df_orig.drop_duplicates(subset=["Year", "Month"]).sort_values(["Year", "Month"]).copy()

# The sample shows:
# 2018-01: CPI_Food_Inflation=4.70, Inflation_Change_3M=-1.90, Inflation_Lag1=5.85
# So Inflation_Lag1 is lagged CPI_Food_Inflation (5.85 was likely Dec 2017)
# Let's check: is Inflation_Change_3M = CPI_Food_Inflation - Inflation_Lag1 ?
macro2["hyp_a"] = macro2["CPI_Food_Inflation"] - macro2["Inflation_Lag1"]
mask = macro2["Inflation_Change_3M"].notna() & macro2["hyp_a"].notna()
diff = (macro2.loc[mask, "Inflation_Change_3M"] - macro2.loc[mask, "hyp_a"]).abs()
close = (diff < 0.01).mean() * 100
print(f"  CPI_Food_Inflation - Inflation_Lag1: {close:.1f}% match (median={diff.median():.4f})")

# 2018-01: 4.70 - 5.85 = -1.15 but Inflation_Change_3M = -1.90. Not matching.

# What if Inflation_Change_3M is a 3-month rolling change of CPI_Food_Index?
# (CPI_t - CPI_{t-3}) ?
macro2["cpi_diff3"] = macro2["CPI_Food_Index"] - macro2["CPI_Food_Index"].shift(3)
mask = macro2["Inflation_Change_3M"].notna() & macro2["cpi_diff3"].notna()
diff = (macro2.loc[mask, "Inflation_Change_3M"] - macro2.loc[mask, "cpi_diff3"]).abs()
close = (diff < 0.01).mean() * 100
print(f"  CPI_Food_Index - CPI_Food_Index.shift(3): {close:.1f}% match (median={diff.median():.4f})")

# Check: 2018-04: CPI=135.8, expected CPI_3m_ago = CPI Jan 2018 = 138.1
# Diff = 135.8 - 138.1 = -2.3. Actual Inflation_Change_3M = -1.50. No match.

# What if it's seasonally adjusted? Or year-over-year?
# Let's look at YoY CPI change for that month minus previous quarter
# What about CPI_Food_Inflation diff compared to 12 months ago?

# Perhaps it's (CPI_Food_Inflation_t - CPI_Food_Inflation_{t-12}) normalized?
# Or maybe we should look at the Inflation_Lag1 more carefully
macro2["infl_lag_verify"] = macro2["CPI_Food_Inflation"].shift(1)
mask = macro2["Inflation_Lag1"].notna() & macro2["infl_lag_verify"].notna()
diff_lag = (macro2.loc[mask, "Inflation_Lag1"] - macro2.loc[mask, "infl_lag_verify"]).abs()
close_lag = (diff_lag < 0.01).mean() * 100
print(f"\n  Verify Inflation_Lag1 = CPI_Food_Inflation.shift(1): {close_lag:.1f}% match")

# If Inflation_Lag1 is NOT simply shifted CPI_Food_Inflation, maybe it comes from
# a different source (e.g., quarterly data)

# Maybe Inflation_Change_3M is computed from a different inflation series entirely
# Let's see if it correlates with any simple transform
print(f"\n  Correlation matrix:")
corr_cols = ["Inflation_Change_3M", "CPI_Food_Inflation", "CPI_Food_Index", "Inflation_Lag1"]
print(macro2[corr_cols].corr().to_string())

# Maybe it's just the diff(1) of CPI_Food_Inflation regardless of the name
macro2["infl_diff1"] = macro2["CPI_Food_Inflation"].diff(1)
mask = macro2["Inflation_Change_3M"].notna() & macro2["infl_diff1"].notna()
diff = (macro2.loc[mask, "Inflation_Change_3M"] - macro2.loc[mask, "infl_diff1"]).abs()
close = (diff < 0.01).mean() * 100
print(f"\n  diff(CPI_Food_Inflation, 1): {close:.1f}% match")

# Let's just print more rows to spot the pattern manually
print("\nFull macro series (all 92 months):")
print(macro2[["Year", "Month", "CPI_Food_Index", "CPI_Food_Inflation", "Inflation_Change_3M", "Inflation_Lag1"]].to_string(index=False))

print("\n--- Done ---")
