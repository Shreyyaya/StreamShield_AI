# (dummy dataset)

import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000
features = 10

X = np.random.rand(rows, features)
y = np.random.randint(0, 2, rows)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(features)])
df["label"] = y

df.to_csv("data/raw/dataset.csv", index=False)

print("Dataset created!")