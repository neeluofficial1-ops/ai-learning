import pandas as pd
import matplotlib.pyplot as plt
data = {
    "name":["arun","priya","kumar","divya","ravi","meena"],
    "dept":["bcom","bcom","bca","bca","bcom","bca"],
    "maths":[78,85,62,None,90,70],
    "english":[65,88,70,75,80,68],
}
df = pd.DataFrame(data)
df["maths"] = df["maths"].fillna(df["maths"].mean())
df["total"] = df["maths"]+df["english"]
plt.figure()
df["english"].plot(kind="hist",bins=5)
plt.title("english marks")
plt.xlabel("marks")
plt.show()
top = df["total"].idxmax()
print(df.loc[top,"name"],df.loc[top, "total"])