import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
data = {
    "hours":[1,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,1.5,2.8,5.2,9.5],
    "attendance":[50,55,60,58,65,70,72,62,80,85,99,90,92,95,75,96,48,62,78,98],
    "passed":[0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,0,1,1],
}
print(len(data["hours"]),len(data["attendance"]),len(data["passed"]))
df =pd.DataFrame(data)
print(df.head())
x=df[["hours","attendance"]]
y=df["passed"]
print(x.shape)
print(y.shape)
for rs in [0,1,7,42,99]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,test_size=0.25,random_state=42
     )
    model = LogisticRegression()
    model.fit(x_train,y_train)
    print(rs,accuracy_score(y_test,model.predict(x_test)))
y_pred = model.predict(x_test)
print("actual  :",list(y_test))
print("prediction  :",list(y_pred))
print("accurancy  :",accuracy_score(y_test,y_pred))
print("confusion matrix:\n",confusion_matrix(y_test,y_pred))
new = pd.DataFrame({"hours":[3,5,8],"attendance":[55,70,90]})
print("prediction: ",model.predict(new))
print("pass probability:",model.predict_proba(new)[:,1].round(2))
