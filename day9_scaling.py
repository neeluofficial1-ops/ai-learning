import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(42)
n = 60

hours = rng.uniform(1,10,n)
attendance = rng.uniform(50,100,n)
fees_paid = rng.uniform(10000,50000,n)

scores = hours * 8 + attendance * 0.4 + rng.normal(0,5,n)
y = (scores > np.median(scores)).astype(int)

x2 = np.column_stack([hours,attendance])
plain = KNeighborsClassifier(n_neighbors=5)
print(cross_val_score(plain,x2,y,cv=5).mean())
