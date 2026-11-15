import pandas as pd
import matplotlib.pyplot as plt

data ={
    "name": ["arun","priya","kumar","divya","ravi","meena"],
       "dept":["bcom","bcom","bca","bcom","bca","bcom"],
       "maths":[78,85,62,None,90,70],
       "english":[65,88,70,75,80,68],
       }
for key,value in data.items():
    print(key,len(value))
df = pd.DataFrame(data)

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())
df["maths"] = df["maths"].fillna(df["maths"].mean())

print(df[df["maths"]>75])

df["total"]=df["maths"]+df["english"]

print(df.groupby("dept")["total"].mean())

df.groupby("dept")["total"].mean().plot(kind="bar")
plt.title("average total by department")
plt.xlabel("department")
plt.ylabel("average total")
plt.show()