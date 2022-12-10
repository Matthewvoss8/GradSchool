from main import LymeDisease, Grid_Search
from sklearn.linear_model import LogisticRegression

l = LymeDisease()

estimator = LogisticRegression(random_state=450)
param_grid = {'max_iter': [1000]}
Grid_Search(estimator=estimator, cv=10, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
# Accuracy is ok at just 74.8% validation accuracy.
