import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix,accuracy_score,precision_score,recall_score,classification_report)
from sklearn.tree import DecisionTreeClassifier

hours = np.array([1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10])
attendance = np.array([50,55,60,65,70,75,80,85,90,95,90,60,80,50,95,55,60,70,75,85])
passed = np.array([0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1,1,1,1])

x = np.column_stack((hours,attendance))
y = passed
x_train , x_test , y_train , y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)

model = DecisionTreeClassifier(max_depth = 2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

print("confusion matrix:")
print(cm)
print("accuracy:",accuracy_score(y_test,y_pred))
print("precision:",precision_score(y_test,y_pred))
print("recall :",recall_score(y_test,y_pred))
print()
print(classification_report(y_test,y_pred))
