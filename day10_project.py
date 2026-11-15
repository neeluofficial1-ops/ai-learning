import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(42)
hours = rng.uniform(1,10,60).round(1)
attendance = rng.integers(50,100,60)

score = hours * 8 + attendance * 0.5 + rng.normal(0,8,60)
passed = (score > 70).astype(int)

df = pd.DataFrame({"hours": hours , "attendance":attendance , "passed":passed})
print(df.head())
print("pass count:\n",df["passed"].value_counts())

x = df[["hours","attendance"]]
y = df["passed"]

models = {
    "logistic":make_pipeline(StandardScaler(),LogisticRegression()),
    "KNN": make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors = 3)),
    "tree":DecisionTreeClassifier(max_depth = 2 , random_state = 42),
    "forest":RandomForestClassifier(n_estimators = 100 , random_state = 42),
}

results = {}
for name,model in models.items():
    scores = cross_val_score(model, x , y , cv=5)
    results[name] = scores.mean()
    print(name,scores.round(2),"average:",round(scores.mean(),3))

best = max(results,key=results.get)
print("best model:",best)

best_model = models[best]
best_model.fit(x,y)
joblib.dump(best_model,"best_model.pkl")

new_students = pd.DataFrame({
        "hours":[2,5,9],
        "attendance":[60,75,95],
    })
print("predictions:",best_model.predict(new_students))
print("probabilities:\n",best_model.predict_proba(new_students).round(2))