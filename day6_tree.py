import pandas as pd
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = {
    "hours":[1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 1, 3, 5, 9],
    "attedance":[50, 60, 80, 55, 70, 65, 85, 60, 90, 70, 95, 75, 88, 80, 92, 96, 40, 85, 55, 90],
    "passed":[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
}

df = pd.DataFrame(data)
x = df[["hours","attedance"]]
y = df["passed"]

x_train,x_test,y_train,y_test = train_test_split(
    x,y, test_size = 0.3, random_state  = 42
)
model = DecisionTreeClassifier(max_depth = 3,random_state = 42)
model.fit(x_train,y_train)

pred = model.predict(x_test)
print("accuracy:",accuracy_score(y_test,pred))
print("6 hours , 80% attedance ->",model.predict([[6,80]]))
plt.figure(figsize = (10 , 6))
plot_tree(model,feature_names = ["hours","attedance"],
          class_names = ["fail","pass"], filled = True)
plt.show()
