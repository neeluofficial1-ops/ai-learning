import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
x = data.data
y = data.target

x_train , x_test , y_train , y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

from sklearn.metrics import make_scorer , recall_score
malignant_recall = make_scorer(recall_score,pos_label=0)

models = {
    "logistic":Pipeline([("scaler",StandardScaler()),
                        ("clf",LogisticRegression(max_iter=1000))]),
    "tree":DecisionTreeClassifier(max_depth=3,random_state=42),
}
for name, model in models.items():
    scores = cross_val_score(model,x,y,cv=5,scoring=malignant_recall)
    print(name,scores,"average:",round(scores.mean(),3))

best = models["logistic"]
best.fit(x_train,y_train)
joblib.dump(best,"cancer_model.pkl")
print("model saved")

loaded = joblib.load("cancer_model.pkl")
new_patient = x_test[:1]
pred = loaded.predict(new_patient)
print("prediction:", data.target_names[pred[0]])
print("real answer:",data.target_names[y_test[0]])

