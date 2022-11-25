from sklearn.ensemble import RandomForestClassifier
from main import LymeDisease
from sklearn.model_selection import GridSearchCV
import pandas as pd

l = LymeDisease()
random_forest = RandomForestClassifier(class_weight='balanced')
param_grid = {'n_estimators': pd.Series(range(1, 11)).apply(lambda x: x*10).to_list(),
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [1, 2, 5, 10, 20],
                      'min_samples_leaf': list(range(1,6))
             }
rf = GridSearchCV(
            estimator=random_forest,
            param_grid=param_grid
)
rf.fit(l.x_train, l.y_train)
print('Best Random Forest', rf.best_params_)
param_grid = {'n_estimators': list(range(75, 86)),
                      'criterion': ['gini', 'entropy'],
                      'max_depth': list(range(2, 9)),
                      'min_samples_leaf': list(range(1, 3))}
rf = GridSearchCV(
            estimator=random_forest,
            param_grid=param_grid,
            cv=10,
            scoring='accuracy'
)
rf.fit(l.x_train, l.y_train)
print('Best Random Forest parameters %s' % rf.best_params_)
print('Training accuracy for random forest is %.2f%%' % (rf.best_score_*100))
print('Validation accuracy for random forest is %.2f%%' % (rf.best_estimator_.score(l.x_valid, l.y_valid) * 100))
