#TODO: Figure out what other models could work in this data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


class Lyme:
    def __init__(self):
        self.data = self.formatData
        self.x = None
        self.y = None
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        self.splitData()
        self.logAcc = None
        self.rfAcc = None

    @property
    def formatData(self) -> pd.DataFrame:
        """
        Return a dataframe in a tidy format such that time is one column
        and concat some useful columns to the dataframe
        :return: A tidy dataframe
        """
        lyme_disease: str = '/Users/matthewvoss/Documents/BUS767/BUS767/Lyme.xlsx'
        df1 = pd.read_excel(lyme_disease, sheet_name='Predictors of Tick Establish')
        df2 = pd.read_excel(lyme_disease, sheet_name='First Report Tick & Incidence')
        local = df1.iloc[:, [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
        local = local.set_index('GeoID')
        local.columns = [list(range(1, 11))]
        local = local.unstack()
        local = pd.DataFrame(local).reset_index()
        local.columns = ['Time', 'GeoID', 'PosCounty']
        local = pd.merge(df2, local, on=['GeoID', 'Time'])
        local = local[['TickPres', 'County', 'State', 'GeoID', 'Time', 'Rate', 'PosCounty']]
        return local

    def splitData(self):
        self.y = self.data.loc[:, 'TickPres']
        self.x = self.data.iloc[:, 3:7]
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.x, self.y, train_size=0.8,
                                                                            shuffle=True, random_state=22,
                                                                            stratify=self.y)


        #TODO: You need to use some sort of shuffler to choose multiple training and validation sets.

    def logistic(self):
        """
        Use logistic regression as the base accuracy to try and beat the original researchers' accuracy.
        :return: None
        """
        c = cross_val_score(estimator=LogisticRegression(random_state=22),
                            X=self.xtrain,
                            y=self.ytrain,
                            cv=StratifiedKFold(n_splits=10, random_state=22, shuffle=True))
        print('The base accuracy to beat is: %.2f%%' % np.mean(c*100))
        self.logAcc = np.mean(c*100)

    def randomForest(self):
        """
        The random forest is a good intro model especially if researchers are interested in Lyme Disease.
        :return:
        """
        param_grid = {'n_estimators': pd.Series(range(1, 11)).apply(lambda x: x*10).to_list(),
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [1, 2, 5, 10, 20],
                      'min_samples_leaf': list(range(1,6))
                      }
        rf = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=param_grid
        )
        c = cross_val_score(
            estimator=rf, X=self.xtrain, y=self.ytrain
        )
