import pandas as pd
import matplotlib.pyplot as plt

data = {
    "name":["a","b","c","d","e","f","g","h","i","j"],
    "maths":[90,80,77,66,88,99,45,67,55,78],
    "science":[90,80,77,66,88,99,45,67,55,78],
}
df = pd.DataFrame(data)
df.plot(x="name",y=["maths","science"],kind="bar")
plt.title("...")
plt.show()