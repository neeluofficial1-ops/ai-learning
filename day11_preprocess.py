import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

rng = np.random.default_rng(42)
n = 80

hours = rng.integers(1 , 11 , n).astype(float)
attendance = rng.integers(55 , 100 , n).astype(float)
dept = rng.choice(["commerce" , "arts" , "science"], n)
score = hours * 6 + attendance * 0.4 + rng.normal(0 , 8 , n)
passed = (score > 70).astype(int)

df = pd.DataFrame({
    "hours":hours,
    "attendance": attendance,
    "dept":dept,
    "passed":passed,
})

df.loc[rng.choice(n , 8 , replace=False), "hours"] = np.nan
df.loc[rng.choice(n , 6 , replace=False),"attendance"] = np.nan
df.loc[rng.choice(n , 5 , replace=False),"dept"] = np.nan
print(df.head(10))
print("\nMissing values:")
print(df.isna().sum())

x = df[["hours","attendance"]]
y = df["passed"]

num_cols = ["hours", "attendance"]

num_pipe = Pipeline([
    ("fill",SimpleImputer(strategy = "median")),
    ("scale",StandardScaler()),
])

cat_pipe = Pipeline([
    ("fill",SimpleImputer(strategy = "most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown = "ignore")),
])

prep = ColumnTransformer([
    ("num", num_pipe , num_cols),
])

model = Pipeline([
    ("prep",prep),
    ("clf",LogisticRegression()),
])

scores = cross_val_score(model,x,y,cv=5)
print("\nCV scores:",scores)
print("average:",scores.mean())

