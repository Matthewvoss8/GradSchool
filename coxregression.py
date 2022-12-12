from main import LymeDisease, Grid_Search
from sksurv.linear_model import CoxPHSurvivalAnalysis

l = LymeDisease()
Cox = CoxPHSurvivalAnalysis()
param_grid = {'n_iter': [200]}

Cox.fit(X=l.x_train, y=l.y_train)
