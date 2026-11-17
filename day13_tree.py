from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
x = data.data
y = data.target


x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)
model = Pipeline([
    ("scaler",StandardScaler()),
    ("clf",LogisticRegression(max_iter=1000)),
])

tree = DecisionTreeClassifier(max_depth=3,random_state=42)
tree.fit(x_train,y_train)
pred_tree = tree.predict(x_test)

print(confusion_matrix(y_test,pred_tree))
print(classification_report(y_test,pred_tree,target_names=data.target_names))
