import pandas as pd
df = pd.read_csv('data/Drishti_Cascade_Final_With_EMDAT.csv')
combos = df.groupby(['HS4','HS2','Commodity'])['Value_USD'].sum().reset_index()
combos = combos.sort_values('Value_USD', ascending=False)
print("Total HS4 codes:", combos["HS4"].nunique())
print("\nAll HS4 commodities:")
for _, r in combos.iterrows():
    desc = str(r["Commodity"])[:80]
    print("  HS4={:4d} ch.{:2d} | {:80s} | ${:>15,.0f}".format(int(r["HS4"]), int(r["HS2"]), desc, r["Value_USD"]))
