import numpy as np
from sklearn.ensemble import RandomForestClassifier
from main import LymeDisease, Grid_Search
import pandas as pd
from sklearn.preprocessing import StandardScaler

l = LymeDisease()
random_forest = RandomForestClassifier(class_weight='balanced')

param_grid = {'n_estimators': pd.Series(range(1, 11)).apply(lambda x: x * 10).to_list(),
              'criterion': ['gini', 'entropy'],
              'max_depth': [1, 2, 5, 10, 20],
              'min_samples_leaf': list(range(1, 6))
              }
Grid_Search(estimator=random_forest, cv=10, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)

param_grid = {'n_estimators': list(range(75, 86)),
              'criterion': ['gini', 'entropy'],
              'max_depth': list(range(2, 9)),
              'min_samples_leaf': list(range(1, 3))}

Grid_Search(estimator=random_forest, cv=10, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)

param_grid = {'n_estimators': [79],
              'max_depth': [8],
              'min_samples_leaf': [2],
              'criterion': ['entropy']}

if __name__ == '__main__':
    x_training = l.x_train.copy()
    x_test = l.x_test.copy()
    y_training = l.y_train.copy()
    y_test = l.y_test.copy()
    x_test = x_test.dropna()
    y_test = y_test.dropna()
    scale = StandardScaler()
    Grid_Search(estimator=random_forest, cv=10, param_grid=param_grid, x_train=x_training, y_train=y_training,
                x_valid=x_test, y_valid=y_test)
# using the param_grid, still best validation accuracy at 77%
