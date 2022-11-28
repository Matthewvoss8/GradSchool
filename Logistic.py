from main import LymeDisease, Grid_Search
from sklearn.linear_model import LogisticRegression

l = LymeDisease()

estimator = LogisticRegression()
param_grid = {'max_iter': [10000]}
print = Grid_Search(estimator=estimator, cv=10, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
                    x_valid=l.x_valid, y_valid=l.y_valid)
# Accuracy is pretty poor at just 71  percent validation accuracy.
