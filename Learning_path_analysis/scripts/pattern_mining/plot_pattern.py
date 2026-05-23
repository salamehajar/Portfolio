import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./scripts/pattern_mining/patterns_readable.csv")

top = df.head(20)

plt.figure(figsize=(10,8))

sns.barplot(
    data=top,
    y="decoded",
    x="ratio"
)

plt.title("Top discriminant patterns")

plt.show()