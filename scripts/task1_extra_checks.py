import pandas as pd, numpy as np

df = pd.read_csv('data/Drishti_Cascade_Final_With_EMDAT.csv')

# Weighted_Trade_Impact
df_s = df.copy()
df_s['wti_hyp'] = df_s['Value_USD'] * df_s['Shock_Intensity']
m = df_s['Weighted_Trade_Impact'].notna() & df_s['wti_hyp'].notna() & (df_s['Weighted_Trade_Impact'] != 0)
d = (df_s.loc[m,'Weighted_Trade_Impact'] - df_s.loc[m,'wti_hyp']).abs()
cl = (d < 1).mean() * 100
print(f'WTI = Value_USD * Shock_Intensity: {cl:.1f}%')

df_s['wti_hyp2'] = df_s['Effective_Shock'] * df_s['Value_USD']
d2 = (df_s.loc[m,'Weighted_Trade_Impact'] - df_s.loc[m,'wti_hyp2']).abs()
cl2 = (d2 < 1).mean() * 100
print(f'WTI = Effective_Shock * Value_USD: {cl2:.1f}%')

# Trade_Momentum
tm_min = df_s['Trade_Momentum'].min()
tm_max = df_s['Trade_Momentum'].max()
tm_mean = df_s['Trade_Momentum'].mean()
print(f'Trade_Momentum: min={tm_min:.4f}, max={tm_max:.4f}, mean={tm_mean:.4f}')

df_s['tm_hyp'] = df_s['Trade_Return_1M'] + df_s['Trade_Return_3M']
m2 = df_s['Trade_Momentum'].notna() & df_s['tm_hyp'].notna() & (df_s['Trade_Momentum'] != 0)
d = (df_s.loc[m2,'Trade_Momentum'] - df_s.loc[m2,'tm_hyp']).abs()
cl = (d < 0.001).mean() * 100
print(f'Trade_Momentum = TR1M + TR3M: {cl:.1f}%')

# Trade_Type_Encoded
enc = df_s[['Trade_Type','Trade_Type_Encoded']].drop_duplicates()
print(f'Trade_Type_Encoded: {dict(zip(enc["Trade_Type"], enc["Trade_Type_Encoded"]))}')

# Price_Volatility_3M
df_s = df.sort_values(['Country','Trade_Type','HS4','Year','Month']).copy()
df_s['pv3_hyp'] = df_s.groupby(['Country','Trade_Type','HS4'])['Unit_Price_USD_per_KG'].transform(lambda x: x.rolling(3, min_periods=1).std())
m = df_s['Price_Volatility_3M'].notna() & df_s['pv3_hyp'].notna() & (df_s['Price_Volatility_3M'] != 0)
d = (df_s.loc[m,'Price_Volatility_3M'] - df_s.loc[m,'pv3_hyp']).abs()
cl = (d < 0.001).mean() * 100
print(f'Price_Volatility_3M = rolling(3).std of Price: {cl:.1f}%')

# Rolling_3M_Volatility
df_s['r3v_hyp'] = df_s.groupby(['Country','Trade_Type','HS4'])['Value_USD'].transform(lambda x: x.rolling(3, min_periods=1).std())
m = df_s['Rolling_3M_Volatility'].notna() & df_s['r3v_hyp'].notna() & (df_s['Rolling_3M_Volatility'] != 0)
d = (df_s.loc[m,'Rolling_3M_Volatility'] - df_s.loc[m,'r3v_hyp']).abs()
cl = (d < 0.01).mean() * 100
print(f'Rolling_3M_Volatility = rolling(3).std of Value_USD: {cl:.1f}%')

# Inflation_Lag1 verification
macro = df.drop_duplicates(subset=['Year','Month']).sort_values(['Year','Month']).copy()
macro['infl_lag_verify'] = macro['CPI_Food_Inflation'].shift(1)
m = macro['Inflation_Lag1'].notna() & macro['infl_lag_verify'].notna()
d = (macro.loc[m,'Inflation_Lag1'] - macro.loc[m,'infl_lag_verify']).abs()
cl = (d < 0.01).mean() * 100
print(f'Inflation_Lag1 = CPI_Food_Inflation.shift(1): {cl:.1f}%')

# Agri_GVA_Lag1
macro['agva_lag'] = macro['Agri_GVA_Growth_Percent'].shift(1)
m = macro['Agri_GVA_Lag1'].notna() & macro['agva_lag'].notna()
d = (macro.loc[m,'Agri_GVA_Lag1'] - macro.loc[m,'agva_lag']).abs()
cl = (d < 0.01).mean() * 100
print(f'Agri_GVA_Lag1 = Agri_GVA_Growth_Percent.shift(1): {cl:.1f}%')

# GDP_Lag1
macro['gdp_lag'] = macro['GDP_Growth_Percent'].shift(1)
m = macro['GDP_Lag1'].notna() & macro['gdp_lag'].notna()
d = (macro.loc[m,'GDP_Lag1'] - macro.loc[m,'gdp_lag']).abs()
cl = (d < 0.01).mean() * 100
print(f'GDP_Lag1 = GDP_Growth_Percent.shift(1): {cl:.1f}%')

print('\nDone.')
