"""
Drishti — Task 1 follow-up: Deep investigation of undocumented formula columns.
Trade_Return_1M, Price_Return_1M, and Inflation_Change_3M didn't match simple
percent-change hypotheses. This script tries additional formulas.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAIN_CSV = os.path.join(DATA_DIR, "Drishti_Cascade_Final_With_EMDAT.csv")

print("Loading main dataset...")
df = pd.read_csv(MAIN_CSV)
df = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).reset_index(drop=True)

print(f"\nShape: {df.shape}")
print(f"\nTrade_Return_1M stats:")
print(df["Trade_Return_1M"].describe())
print(f"\nPrice_Return_1M stats:")
print(df["Price_Return_1M"].describe())

# =========================================================
# INVESTIGATION 1: Trade_Return_1M
# =========================================================
print("\n" + "=" * 70)
print("INVESTIGATION 1: Trade_Return_1M")
print("=" * 70)

# Take a small sample to inspect
sample = df[df["Country"] == "CHINA"].copy()
sample = sample[sample["Trade_Type"] == "Import"]
hs4_sample = sample["HS4"].value_counts().index[0]
series = sample[sample["HS4"] == hs4_sample].sort_values(["Year", "Month"]).head(12)

print(f"\nSample series: Country=CHINA, Trade_Type=Import, HS4={hs4_sample}")
print(series[["Year", "Month", "Value_USD", "Log_Value_USD", "Trade_Return_1M", "MoM_Change_Value"]].to_string(index=False))

# Hypothesis: log return = ln(Value_t / Value_{t-1})
series_full = sample[sample["HS4"] == hs4_sample].sort_values(["Year", "Month"]).copy()
series_full["log_return"] = np.log(series_full["Value_USD"] / series_full["Value_USD"].shift(1))
series_full["log_return_safe"] = np.log1p(series_full["Value_USD"]) - np.log1p(series_full["Value_USD"].shift(1))
series_full["pct_change_raw"] = series_full["Value_USD"].pct_change(1)

print(f"\nComparing hypotheses on {len(series_full)} rows:")
for hyp_col, label in [("log_return", "ln(V_t/V_{t-1})"),
                         ("log_return_safe", "ln(1+V_t) - ln(1+V_{t-1})"),
                         ("pct_change_raw", "pct_change (no ×100)")]:
    mask = series_full["Trade_Return_1M"].notna() & series_full[hyp_col].notna() & np.isfinite(series_full[hyp_col])
    if mask.sum() > 0:
        diff = (series_full.loc[mask, "Trade_Return_1M"] - series_full.loc[mask, hyp_col]).abs()
        close = (diff < 0.001).mean() * 100
        print(f"  {label}: {close:.1f}% match (median diff={diff.median():.6f})")

# Hypothesis: diff of Log_Value_USD
series_full["log_diff"] = series_full["Log_Value_USD"].diff(1)
mask = series_full["Trade_Return_1M"].notna() & series_full["log_diff"].notna()
if mask.sum() > 0:
    diff = (series_full.loc[mask, "Trade_Return_1M"] - series_full.loc[mask, "log_diff"]).abs()
    close = (diff < 0.001).mean() * 100
    print(f"  diff(Log_Value_USD): {close:.1f}% match (median diff={diff.median():.6f})")

print(f"\nSide-by-side comparison (first 10 rows):")
cols_show = ["Year", "Month", "Value_USD", "Log_Value_USD", "Trade_Return_1M", "log_return", "log_return_safe", "log_diff", "pct_change_raw"]
print(series_full[cols_show].head(10).to_string(index=False))

# Now test the log_diff hypothesis globally
print("\n--- Global test: Trade_Return_1M = diff(Log_Value_USD) ---")
df_test = df.sort_values(["Country", "Trade_Type", "HS4", "Year", "Month"]).copy()
df_test["log_diff_global"] = df_test.groupby(["Country", "Trade_Type", "HS4"])["Log_Value_USD"].diff(1)
mask = df_test["Trade_Return_1M"].notna() & df_test["log_diff_global"].notna()
if mask.sum() > 0:
    diff = (df_test.loc[mask, "Trade_Return_1M"] - df_test.loc[mask, "log_diff_global"]).abs()
    close = (diff < 0.001).mean() * 100
    print(f"  Global match: {close:.1f}% of {mask.sum():,} rows within 0.001 tolerance")
    print(f"  Median diff: {diff.median():.8f}, Max diff: {diff.max():.8f}")

# =========================================================
# INVESTIGATION 2: Price_Return_1M
# =========================================================
print("\n" + "=" * 70)
print("INVESTIGATION 2: Price_Return_1M")
print("=" * 70)

# Hypothesis: log return of Unit_Price_USD_per_KG
df_test["log_price"] = np.log1p(df_test["Unit_Price_USD_per_KG"])
df_test["price_log_diff"] = df_test.groupby(["Country", "Trade_Type", "HS4"])["log_price"].diff(1)
mask = df_test["Price_Return_1M"].notna() & df_test["price_log_diff"].notna()
if mask.sum() > 0:
    diff = (df_test.loc[mask, "Price_Return_1M"] - df_test.loc[mask, "price_log_diff"]).abs()
    close = (diff < 0.001).mean() * 100
    print(f"  log_diff of ln(1+Price): {close:.1f}% match (median={diff.median():.8f})")

# Hypothesis: simple log return ln(P_t/P_{t-1})
df_test["price_log_return"] = np.log(df_test["Unit_Price_USD_per_KG"] / df_test.groupby(["Country", "Trade_Type", "HS4"])["Unit_Price_USD_per_KG"].shift(1))
mask = df_test["Price_Return_1M"].notna() & df_test["price_log_return"].notna() & np.isfinite(df_test["price_log_return"])
if mask.sum() > 0:
    diff = (df_test.loc[mask, "Price_Return_1M"] - df_test.loc[mask, "price_log_return"]).abs()
    close = (diff < 0.001).mean() * 100
    print(f"  ln(P_t/P_{'{t-1}'}): {close:.1f}% match (median={diff.median():.8f})")

# Just pct_change without *100
df_test["price_pct_raw"] = df_test.groupby(["Country", "Trade_Type", "HS4"])["Unit_Price_USD_per_KG"].pct_change(1)
mask = df_test["Price_Return_1M"].notna() & df_test["price_pct_raw"].notna() & np.isfinite(df_test["price_pct_raw"])
if mask.sum() > 0:
    diff = (df_test.loc[mask, "Price_Return_1M"] - df_test.loc[mask, "price_pct_raw"]).abs()
    close = (diff < 0.001).mean() * 100
    print(f"  pct_change (no *100): {close:.1f}% match (median={diff.median():.8f})")

# =========================================================
# INVESTIGATION 3: Inflation_Change_3M
# =========================================================
print("\n" + "=" * 70)
print("INVESTIGATION 3: Inflation_Change_3M")
print("=" * 70)

# Get deduplicated macro series
macro = df.drop_duplicates(subset=["Year", "Month"]).sort_values(["Year", "Month"]).copy()
print(f"\nUnique year-months: {len(macro)}")
print(f"\nInflation_Change_3M stats:")
print(macro["Inflation_Change_3M"].describe())
print(f"\nCPI_Food_Inflation stats:")
print(macro["CPI_Food_Inflation"].describe())

print(f"\nSample values:")
print(macro[["Year", "Month", "CPI_Food_Index", "CPI_Food_Inflation", "Inflation_Change_3M", "Inflation_Lag1"]].head(20).to_string(index=False))

# Hypothesis: pct_change of CPI_Food_Index over 3 months
macro["cpi_pct_3m"] = macro["CPI_Food_Index"].pct_change(3) * 100
mask = macro["Inflation_Change_3M"].notna() & macro["cpi_pct_3m"].notna()
if mask.sum() > 0:
    diff = (macro.loc[mask, "Inflation_Change_3M"] - macro.loc[mask, "cpi_pct_3m"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"\n  pct_change of CPI_Food_Index (3 periods): {close:.1f}% match")

# Hypothesis: 3-month change of CPI_Food_Inflation
macro["infl_diff_3m"] = macro["CPI_Food_Inflation"].diff(3)
mask = macro["Inflation_Change_3M"].notna() & macro["infl_diff_3m"].notna()
if mask.sum() > 0:
    diff = (macro.loc[mask, "Inflation_Change_3M"] - macro.loc[mask, "infl_diff_3m"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"  diff(CPI_Food_Inflation, 3): {close:.1f}% match")

# Hypothesis: rolling 3M mean of CPI_Food_Inflation
macro["infl_roll3m"] = macro["CPI_Food_Inflation"].rolling(3, min_periods=1).mean()
mask = macro["Inflation_Change_3M"].notna() & macro["infl_roll3m"].notna()
if mask.sum() > 0:
    diff = (macro.loc[mask, "Inflation_Change_3M"] - macro.loc[mask, "infl_roll3m"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"  rolling 3M mean of CPI_Food_Inflation: {close:.1f}% match")

# Hypothesis: month-over-month change (1-period diff)
macro["infl_diff_1m"] = macro["CPI_Food_Inflation"].diff(1)
mask = macro["Inflation_Change_3M"].notna() & macro["infl_diff_1m"].notna()
if mask.sum() > 0:
    diff = (macro.loc[mask, "Inflation_Change_3M"] - macro.loc[mask, "infl_diff_1m"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"  diff(CPI_Food_Inflation, 1): {close:.1f}% match")

# Hypothesis: (CPI_Food_Index_t - CPI_Food_Index_{t-3}) / CPI_Food_Index_{t-3} * 12/3
macro["cpi_ann_3m"] = (macro["CPI_Food_Index"] / macro["CPI_Food_Index"].shift(3) - 1) * 100 * 4
mask = macro["Inflation_Change_3M"].notna() & macro["cpi_ann_3m"].notna()
if mask.sum() > 0:
    diff = (macro.loc[mask, "Inflation_Change_3M"] - macro.loc[mask, "cpi_ann_3m"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"  Annualized 3M CPI change: {close:.1f}% match")

# Let's also check if it's related to the difference between consecutive CPI_Food_Inflation values
# but computed over all data (not just year-month unique)
# or if it's CPI_Food_Inflation lagged by 3 MINUS current
macro["infl_lag3"] = macro["CPI_Food_Inflation"].shift(3)
macro["infl_vs_lag3"] = macro["CPI_Food_Inflation"] - macro["infl_lag3"]
mask = macro["Inflation_Change_3M"].notna() & macro["infl_vs_lag3"].notna()
if mask.sum() > 0:
    diff = (macro.loc[mask, "Inflation_Change_3M"] - macro.loc[mask, "infl_vs_lag3"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"  CPI_Food_Inflation - CPI_Food_Inflation.shift(3): {close:.1f}% match")

# Maybe it's MoM change on CPI_Food_Index directly
macro["cpi_mom"] = macro["CPI_Food_Index"].pct_change(1) * 100
mask = macro["Inflation_Change_3M"].notna() & macro["cpi_mom"].notna()
if mask.sum() > 0:
    diff = (macro.loc[mask, "Inflation_Change_3M"] - macro.loc[mask, "cpi_mom"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"  MoM pct change of CPI_Food_Index: {close:.1f}% match")

# =========================================================
# INVESTIGATION 4: Shock_Trend_3M — try more hypotheses
# =========================================================
print("\n" + "=" * 70)
print("INVESTIGATION 4: Shock_Trend_3M")
print("=" * 70)

macro_country = df.drop_duplicates(subset=["Country", "Year", "Month"]).sort_values(["Country", "Year", "Month"]).copy()
print(f"Shock_Trend_3M stats:")
print(macro_country["Shock_Trend_3M"].describe())
print(f"\nSample (first country):")
country_sample = macro_country[macro_country["Country"] == "CHINA"].head(12)
print(country_sample[["Country", "Year", "Month", "Shock_Intensity", "Shock_Trend_3M"]].to_string(index=False))

# Hypothesis: Shock_Intensity - Shock_Intensity.shift(3)
macro_country["shock_diff3"] = macro_country.groupby("Country")["Shock_Intensity"].diff(3)
mask = macro_country["Shock_Trend_3M"].notna() & macro_country["shock_diff3"].notna()
if mask.sum() > 0:
    diff = (macro_country.loc[mask, "Shock_Trend_3M"] - macro_country.loc[mask, "shock_diff3"]).abs()
    close = (diff < 1).mean() * 100
    print(f"\n  Shock_Intensity.diff(3): {close:.1f}% within 1.0 tolerance")

# Hypothesis: pct_change(3)
macro_country["shock_pct3"] = macro_country.groupby("Country")["Shock_Intensity"].pct_change(3)
mask = macro_country["Shock_Trend_3M"].notna() & macro_country["shock_pct3"].notna() & np.isfinite(macro_country["shock_pct3"])
if mask.sum() > 0:
    diff = (macro_country.loc[mask, "Shock_Trend_3M"] - macro_country.loc[mask, "shock_pct3"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"  Shock_Intensity.pct_change(3): {close:.1f}% within 0.01 tolerance")

# Hypothesis: (S_t - S_rolling3) / S_rolling3  (deviation from trend)
macro_country["shock_roll3"] = macro_country.groupby("Country")["Shock_Intensity"].transform(lambda x: x.rolling(3, min_periods=1).mean())
macro_country["shock_dev"] = (macro_country["Shock_Intensity"] - macro_country["shock_roll3"]) / macro_country["shock_roll3"]
mask = macro_country["Shock_Trend_3M"].notna() & macro_country["shock_dev"].notna() & np.isfinite(macro_country["shock_dev"])
if mask.sum() > 0:
    diff = (macro_country.loc[mask, "Shock_Trend_3M"] - macro_country.loc[mask, "shock_dev"]).abs()
    close = (diff < 0.01).mean() * 100
    print(f"  (S - rolling3mean) / rolling3mean: {close:.1f}% within 0.01 tolerance")

# Let's just look at the ratio Shock_Trend_3M / Shock_Intensity to see if it's a simple transform
ratio = macro_country["Shock_Trend_3M"] / macro_country["Shock_Intensity"]
ratio_clean = ratio[np.isfinite(ratio) & ratio.notna()]
print(f"\n  Shock_Trend_3M / Shock_Intensity: mean={ratio_clean.mean():.6f}, std={ratio_clean.std():.6f}, median={ratio_clean.median():.6f}")

# EWM
macro_country["shock_ewm"] = macro_country.groupby("Country")["Shock_Intensity"].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
mask = macro_country["Shock_Trend_3M"].notna() & macro_country["shock_ewm"].notna()
if mask.sum() > 0:
    diff = (macro_country.loc[mask, "Shock_Trend_3M"] - macro_country.loc[mask, "shock_ewm"]).abs()
    close = (diff < 1).mean() * 100
    print(f"  ewm(span=3) of Shock_Intensity: {close:.1f}% within 1.0 tolerance")

print("\n--- Done ---")
