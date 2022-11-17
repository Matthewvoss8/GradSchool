#TODO: Figure out what other models could work in this data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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
        #self.logistic()
        self.rfAcc = None
        #self.randomForest()


    @property
    def formatData(self) -> pd.DataFrame:
        """
        Return a dataframe in a tidy format such that time is one column
        and concat some useful columns to the dataframe
        :return: A tidy dataframe
        """
        lyme_disease: str = 'Lyme.xlsx'
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

    @property
    def getCDC(self):
        """
        The CDC Data won't be used in the model. Mostly for exploration of data later
        :return:
        """
        filePath = 'new_cdc.xlsx'
        cdc = pd.read_excel(filePath).rename(columns={'Ctyname': 'County', 'Stname': 'State'})
        cdc['County'] = cdc['County'].apply(lambda x: x.split(' ')[0])
        cdc = cdc.set_index(['County', 'State','STCODE','CTYCODE'])
        cdc.columns = list(range(2000, 2020))
        cdc = cdc.stack().reset_index()
        cdc = cdc.rename(columns={'level_4': 'Year', 0: 'Lyme Disease Rate'})
        return cdc

    def splitData(self):
        self.y = self.data.loc[:, 'TickPres']
        self.x = self.data.iloc[:, 3:7]
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.x, self.y, train_size=0.8,
                                                                            shuffle=True, random_state=22,
                                                                            stratify=self.y)
        self.xtrain, self.xvalidation, self.ytrain, self.yvalidation = train_test_split(self.xtrain, self.ytrain,
                                                                            shuffle=True, random_state=22,
                                                                            train_size=0.75, stratify=self.ytrain)

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


l = Lyme()
param_grid = {'n_estimators': pd.Series(range(1, 11)).apply(lambda x: x*10).to_list(),
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [1, 2, 5, 10, 20],
                      'min_samples_leaf': list(range(1,6))
             }
rf = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid
)
rf.fit(l.xtrain, l.ytrain)
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
rfAcc = '%.2f%%' % (rf.score(l.xvalidation, l.yvalidation)*100)
TickPresence = l.data['TickPres'].sum()
NoTickPresence = l.data.shape[0]-TickPresence
weight = NoTickPresence/TickPresence
param_grid = {'n_estimators': pd.Series(range(1, 11)).apply(lambda x: x*10).to_list()}


# if __name__=="__main__":
#     l = Lyme()
#     print(l.logAcc)
#     print(l.rfAcc)
