import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

data = {
    "hours":[1,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,1.5,2.8,5.2,9.5],
    "attendance":[50,55,60,58,65,70,72,62,80,85,99,90,92,95,75,96,48,62,78,98],
    "passed":[0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,0,1,1],
}
print(len(data["hours"]),len(data["attendance"]),len(data["passed"]))
df =pd.DataFrame(data)
y=df["passed"]
for name,cols in[("hours only",["hours"]),("hours+attendance",["hours","attendance"])]:
    scores = cross_val_score(LogisticRegression(),df[cols],y,cv = 5)
    x = df[cols]
    print(name,scores.mean().round(2))
    for rs in [0,1,7,42,99]:
         x_train, x_test, y_train, y_test = train_test_split(
                x,y,test_size=0.25,random_state=rs
         )
         model = LogisticRegression()
         model.fit(x_train,y_train)
         print(" ",rs,accuracy_score(y_test,model.predict(x_test)))