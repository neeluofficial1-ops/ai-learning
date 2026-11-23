from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestRegressor

data = fetch_california_housing()
x_train , x_test , y_train , y_test = train_test_split(
    data.data , data.target , test_size=0.3 , random_state = 42
)

params = {
    "max_depth":[10 , 15 , 20],
    "n_estimators":[50 , 100],
    "min_samples_leaf":[1 , 3],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state = 42 , n_jobs=-1 ),
    params,
    cv = 3,
    scoring = "r2",
)
grid.fit(x_train,y_train)

print("best settings:",grid.best_params_)
print("best cv r2:",round(grid.best_score_,3))
print("test r2:",round(grid.score(x_test,y_test),3))