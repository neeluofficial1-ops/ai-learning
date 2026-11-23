from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , r2_score
import pandas as pd

data = fetch_california_housing()
x_train , x_test , y_train , y_test = train_test_split(
    data.data , data.target , test_size = 0.3 , random_state = 42
)

model = RandomForestRegressor(n_estimators = 50 , max_depth = 10 , random_state = 42 , n_jobs = -1)
model.fit(x_train , y_train)
pred = model.predict(x_test)

print("Train R2:",round(model.score(x_train , y_train),3))
print("Test R2:",round(r2_score(y_test,pred),3))
print("MAE:",round(mean_absolute_error(y_test , pred),3))

imp = pd.Series(model.feature_importances_,index = data.feature_names)
print(imp.sort_values(ascending = False).round(3))

for n in [10, 50 , 100]:
    m = RandomForestRegressor(n_estimators=n,max_depth = 10 , random_state = 42 , n_jobs = -1)
    m.fit(x_train , y_train)
    print(n , "Train:",round(m.score(x_train , y_train),3),"Test:",round(m.score(x_test , y_test), 3))