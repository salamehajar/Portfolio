import pandas as pd
import ast
import math
from prefixspan import PrefixSpan


# =========================
# PARAMETERS
# =========================

MAX_PATTERN_LENGTH = 3
MIN_SUPPORT = 0.3


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("./scripts/pattern_mining/encoded_sequences.csv")

df["encoded"] = df["encoded"].apply(lambda x: ast.literal_eval(x))


# =========================
# SPLIT GROUPS
# =========================

success = df[df["group"] == "success"]["encoded"].tolist()
failure = df[df["group"] == "failure"]["encoded"].tolist()


# =========================
# COMPUTE MIN SUPPORT
# =========================

minsup_success = math.ceil(MIN_SUPPORT * len(success))
minsup_failure = math.ceil(MIN_SUPPORT * len(failure))

print("Success sequences:", len(success))
print("Failure sequences:", len(failure))

print("Min support success:", minsup_success)
print("Min support failure:", minsup_failure)


# =========================
# PREFIXSPAN
# =========================

ps_success = PrefixSpan(success)
ps_failure = PrefixSpan(failure)

ps_success.maxlen = MAX_PATTERN_LENGTH
ps_failure.maxlen = MAX_PATTERN_LENGTH

patterns_success = ps_success.frequent(minsup_success)
patterns_failure = ps_failure.frequent(minsup_failure)


# =========================
# FORMAT OUTPUT
# =========================

def convert(patterns):

    rows = []

    for sup, pat in patterns:

        rows.append({
            "support": sup,
            "pattern": pat
        })

    return pd.DataFrame(rows)


df_s = convert(patterns_success)
df_f = convert(patterns_failure)


# =========================
# SAVE
# =========================

df_s.to_csv("./scripts/pattern_mining/patterns_success.csv", index=False)
df_f.to_csv("./scripts/pattern_mining/patterns_failure.csv", index=False)

print("Patterns success:", len(df_s))
print("Patterns failure:", len(df_f))

print("Patterns extracted")