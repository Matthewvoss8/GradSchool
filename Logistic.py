from main import LymeDisease, Grid_Search
from sklearn.linear_model import LogisticRegression

l = LymeDisease()
#
# model = LogisticRegression(max_iter=10000).fit(l.x_train, l.y_train)
# acc = model.score(l.x_valid, l.y_valid)*100
# print(f'The model accuracy is {acc:.2f}%')
estimator = LogisticRegression()
param_grid = {'max_iter': 10000}
print = Grid_Search(estimator=estimator, cv=10, param_grid=param_grid, x_train=l.x_train, y_train=l.y_vtrain,
                    x_valid=l.x_valid, y_valid=l.y_valid)
"""
Accuracy is pretty low and we desire an accuracy at least in the 80 percent range. 
"""
