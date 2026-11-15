import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
hours= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]).reshape(-1,1)
marks = np.array([30,33,38,43,45,50,52,58,60,65,72,75,78,80,84,85,90,92,95])
model = LinearRegression()
model.fit(hours,marks)
print("5hours:",model.predict([[5]])[0])
print("10hours:",model.predict([[10]])[0])
prediction = model.predict([[0]])
plt.scatter(hours,marks,color="blue",label="real data")
plt.plot(hours,model.predict(hours),color="red",label="model line")
plt.legend()
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
x_train,x_text,y_train,y_test = train_test_split(
    hours,marks, test_size=0.25,random_state=1
)
model2 = LinearRegression()
model2.fit(x_train,y_train)
score = r2_score(y_test,model2.predict(x_text))
print("r2 score:",score)
