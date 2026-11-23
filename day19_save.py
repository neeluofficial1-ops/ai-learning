from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

data = fetch_california_housing()
x_train , x_test , y_train , y_test = train_test_split(
    data.data , data.target , test_size = 0.3 , random_state = 42
)

model = RandomForestRegressor(
    n_estimators = 100 , max_depth = 20 , min_samples_leaf = 1,
    random_state = 42 , n_jobs = -1
)
model.fit(x_train , y_train)
print("Test R2:",round(model.score(x_test , y_test), 3))

joblib.dump(model,"house_model.pkl")
print("model saved!")

loaded = joblib.load("house_model.pkl")
pred = loaded.predict([x_test[0]])
print("predicted:",round(pred[0],2),"| Real:",round(y_test[0],2))