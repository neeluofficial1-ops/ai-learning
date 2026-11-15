import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
data = {
    "hours":[1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 1, 3, 5, 9],
    "attedance":[50, 60, 80, 55, 70, 65, 85, 60, 90, 70, 95, 75, 88, 80, 92, 96, 40, 85, 55, 90],
    "passed":[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
}

df = pd.DataFrame(data)
x = df[["hours","attedance"]]
y = df["passed"]

for rs in [0, 1, 7, 42, 99]:
    x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.3, random_state=rs
)

    model = DecisionTreeClassifier ( max_depth = 2,random_state=rs)
    model.fit(x_train,y_train)
    print(rs, model.score(x_test,y_test))

