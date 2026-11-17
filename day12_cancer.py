import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()
x = pd.DataFrame(data.data,columns = data.feature_names)
y = data.target

print("shape:",x.shape)
print("missing values:",x.isnull().sum().sum())
print("bening (1):",(y==1).sum(),"malignant (0):",(y == 0).sum())
print(x.iloc[:,:5].describe())

models = {
    "logistic " :Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=1000))]),
    "KNN":Pipeline([("scale",StandardScaler()),("clf",KNeighborsClassifier(n_neighbors=5))]),
    "Tree":Pipeline([("clf",DecisionTreeClassifier(max_depth=3,random_state=42))]),
}

for name, model in models.items():
    scores = cross_val_score(model, x , y , cv=5)
    print(name,scores.round(3),"average:",round(scores.mean(),3))

print("\n---Tree depth check ---")
for d in [1 , 3 , 5 , None]:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(tree, x , y , cv=5)
    print("depth" , d,scores.round(3), "average:", round(scores.mean(),3))