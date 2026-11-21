from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data = fetch_california_housing(as_frame=True)
x = data.data
y = data.target

print(x.shape)
print(x.head())

x_train,x_test,y_train,y_test = train_test_split(
    x , y , test_size=0.3,random_state=42
)

models = {
    "linear": Pipeline([("scale",StandardScaler()),("reg",LinearRegression())]),
    "tree": DecisionTreeRegressor(max_depth=5,random_state=42),
}

for d in [3,5,10,15,20]:
    t = DecisionTreeRegressor(max_depth=d,random_state=42)
    t.fit(x_train,y_train)
    print(d,"train R2:", round(t.score(x_train,y_train),3),
          "test r2:",round(t.score(x_test,y_test), 3))


for name,model in models.items():
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test,pred)
    rmse = mean_squared_error(y_test,pred)**0.5
    r2 = r2_score(y_test,pred)
    print(name,"MAE:",round(mae,3),"RMSE:",round(rmse,3),"R2:",round(r2,3))
