import pandas as pd

df = pd.read_csv("./scripts/pattern_mining/patterns_readable.csv")

# supprimer tous les patterns contenant "home"
df_clean = df[~df["decoded"].str.contains("home", case=False, na=False)]

print("Patterns avant:", len(df))
print("Patterns après:", len(df_clean))

df_clean.to_csv("./scripts/pattern_mining/patterns.csv", index=False)