import pandas as pd

df = pd.DataFrame({"subject":["accounts","maths","pythoin","economics","statistics"],"marks":[72,85,90,64,88]})

print(df)
print()
print(df.describe())
print()
print(df[df["marks"] > 80])