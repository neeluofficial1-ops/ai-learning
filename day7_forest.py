import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.DataFrame({
    "hours":[1,2,2,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,10,10],
    "attedance":[55,60,70,65,75,68,80,72,85,60,78,90,70,88,92,80,95,85,98,90],
    "passed":[0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1],
})

x = data[["hours","attedance"]]
y = data["passed"]

models= {
    "Logistic Regression":LogisticRegression(),
    "Decsion tree":DecisionTreeClassifier(max_depth=2,random_state=42),
    "random forest":RandomForestClassifier(n_estimators=100,random_state=42),
}

for name,models in models.items():
    scores = cross_val_score(models,x,y,cv=5)
    print(name,"->",scores.round(2),"average:",scores.mean().round(2))

for n in [10,50,100,300]:
    forest = RandomForestClassifier(n_estimators=n,random_state=42)
    scores = cross_val_score(forest,x,y,cv=5)
    print(n,"trees ->",scores.round(2),"average:",scores.mean().round(2))
for col,imp in zip(x.columns,forest.feature_importances_):
    print(col,"impotance:",round(imp,2))