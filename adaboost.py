from main import LymeDisease
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

def grid_search_loader(estimator, param_grid: dict, cv: int = 10):
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy'
    )
    return gs

def get_accuracy(grid_search_object, x_t, x_v, y_t, y_v):
    grid_search_object.fit(x_t, y_t)
    train_acc = (grid_search_object.best_score_)*100
    valid_acc = (grid_search_object.score(x_v, y_v))*100
    print(f'{grid_search_object.estimator} best parameters are {grid_search_object.best_params_}')
    print(f'{grid_search_object.estimator} training accuracy is {train_acc}%')
    print(f'{grid_search_object.estimator} validation accuracy is {valid_acc}%')


l = LymeDisease()
# Tune and fit models to find best training and validation accuracy
boost = AdaBoostClassifier(learning_rate=0.01)
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90]}
gs = grid_search_loader(boost, param_grid)
get_accuracy(gs, l.x_train, l.x_valid, l.y_train, l.y_valid)
param_grid = {'n_estimators': [15, 16, 17, 18, 19, 20, 21, 22, 23, 23, 24, 25]}
gs = grid_search_loader(boost, param_grid)
get_accuracy(gs, l.x_train, l.x_valid, l.y_train, l.y_valid)
