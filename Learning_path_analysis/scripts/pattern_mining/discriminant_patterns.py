import pandas as pd
import ast

ps = pd.read_csv("./scripts/pattern_mining/patterns_success.csv")
pf = pd.read_csv("./scripts/pattern_mining/patterns_failure.csv")

ps["pattern"] = ps["pattern"].apply(lambda x: str(ast.literal_eval(x)))
pf["pattern"] = pf["pattern"].apply(lambda x: str(ast.literal_eval(x)))

df = ps.merge(
    pf,
    on="pattern",
    how="outer",
    suffixes=("_success","_failure")
).fillna(0)

df["ratio"] = (df["support_success"]+1)/(df["support_failure"]+1)

df = df.sort_values("ratio",ascending=False)

df.to_csv("./scripts/pattern_mining/discriminant_patterns.csv",index=False)

print(df.head(20))