from sklearn.ensemble import RandomForestClassifier
from main import LymeDisease
from sklearn.model_selection import GridSearchCV
import pandas as pd

l = LymeDisease()
param_grid = {'n_estimators': pd.Series(range(1, 11)).apply(lambda x: x*10).to_list(),
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [1, 2, 5, 10, 20],
                      'min_samples_leaf': list(range(1,6))
             }
rf = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid
)
rf.fit(l.x_train, l.y_train)
print('Best Random Forest', rf.best_params_)
param_grid = {'n_estimators': list(range(75, 86)),
                      'criterion': ['gini', 'entropy'],
                      'max_depth': list(range(2, 9)),
                      'min_samples_leaf': list(range(1, 3))}
rf = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid
)
rf.fit(l.xtrain, l.ytrain)
print('Best Random Fores, ', rf.best_params_)
best_parameters = rf.best_params_
rf = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=best_parameters
)
rf.fit(l.x_train, l.y_train)
acc = rf.score(l.x_train, l.y_train)*100
print(f'Accuracy for random forest is {acc:.2f}')
