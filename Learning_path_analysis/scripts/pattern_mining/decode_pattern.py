import json
import pandas as pd
import ast

mapping = json.load(open("./scripts/pattern_mining/page_mapping.json"))

inv = {v:k for k,v in mapping.items()}

df = pd.read_csv("./scripts/pattern_mining/discriminant_patterns.csv")

def decode(p):

    seq = ast.literal_eval(p)

    return " → ".join(inv[i] for i in seq)

df["decoded"] = df["pattern"].apply(decode)

df.to_csv("./scripts/pattern_mining/patterns_readable.csv",index=False)

print(df.head(20))