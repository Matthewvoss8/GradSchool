from main import LymeDisease, Grid_Search
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

l = LymeDisease()
# Tune and fit models to find best training and validation accuracy
boost = AdaBoostClassifier(learning_rate=0.01)
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90]}
Grid_Search(estimator=boost, cv=10, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
param_grid = {'n_estimators': [15, 16, 17, 18, 19, 20, 21, 22, 23, 23, 24, 25]}
Grid_Search(estimator=boost, cv=10, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)

# Since the accuracy is so low, I am going to try another Boosted ensemble method
# Can only use log loss since our predictor is binary.
# Warning this will take a long time to compile!! Run at your own risk
hist_boost = HistGradientBoostingClassifier(random_state=42)
param_grid = {'loss': ['log_loss'], 'max_iter': [50, 100, 150, 200], 'max_leaf_nodes': [None, 10, 20, 31, 40],
              'min_samples_leaf': [5, 10, 15, 20, 25, 30], 'max_bins': [200, 250, 300, 350, 400]}
Grid_Search(estimator=hist_boost, cv=10, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
param_grid = {'max_iter': [20, 25, 30], 'max_leaf_nodes': [10], 'min_samples_leaf': [10], 'max_bins': [200]}
Grid_Search(estimator=hist_boost, cv=2, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
param_grid = {'max_iter': [30, 35, 37], 'max_leaf_nodes': [10], 'min_samples_leaf': [10], 'max_bins': [200]}
Grid_Search(estimator=hist_boost, cv=2, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
param_grid = {'max_iter': [30], 'max_leaf_nodes': [5, 10, 15], 'min_samples_leaf': [10], 'max_bins': [200]}
Grid_Search(estimator=hist_boost, cv=2, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
param_grid = {'max_iter': [30], 'max_leaf_nodes': [4, 5, 6], 'min_samples_leaf': [10], 'max_bins': [200]}
Grid_Search(estimator=hist_boost, cv=2, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
param_grid = {'max_iter': [30], 'max_leaf_nodes': [5], 'min_samples_leaf': [5, 10, 15], 'max_bins': [200]}
Grid_Search(estimator=hist_boost, cv=2, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
param_grid = {'max_iter': [30], 'max_leaf_nodes': [5], 'min_samples_leaf': [10], 'max_bins': [190, 200, 210, 220]}
Grid_Search(estimator=hist_boost, cv=2, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
param_grid = {'max_iter': [30], 'max_leaf_nodes': [5], 'min_samples_leaf': [10], 'max_bins': [205, 208, 210, 212, 215]}
Grid_Search(estimator=hist_boost, cv=2, param_grid=param_grid, x_train=l.x_train, y_train=l.y_train,
            x_valid=l.x_valid, y_valid=l.y_valid)
# 74.78% validation accuracy
