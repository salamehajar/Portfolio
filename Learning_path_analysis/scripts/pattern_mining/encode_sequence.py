import pandas as pd
import ast
import json

df = pd.read_csv("./scripts/pattern_mining/sequences_with_grades.csv")

df["sequence"] = df["sequence"].apply(lambda x: ast.literal_eval(x))

all_pages = sorted({p for seq in df["sequence"] for p in seq})

page_to_id = {p:i for i,p in enumerate(all_pages)}

encoded = []

for seq in df["sequence"]:

    encoded.append([page_to_id[p] for p in seq])

df["encoded"] = encoded

df.to_csv("./scripts/pattern_mining/encoded_sequences.csv",index=False)

with open("./scripts/pattern_mining/page_mapping.json","w") as f:
    json.dump(page_to_id,f,indent=2)

print("Encoding complete")