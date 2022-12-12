import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def create_pos_county(data):
    df = data.copy()
    df = df[df.columns[pd.Series(df.columns).str.startswith('PosCounty')]].reset_index()
    long = pd.wide_to_long(df, stubnames='PosCounty', i='index', j='Time').reset_index(['index', 'Time'])
    return long

class LymeDisease:
    def __init__(self):
        self.data = None
        self.cdc = None
        self.abb = None
        self.test = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None
        self.y_test = None
        self.x_train_tensor = None
        self.x_valid_tensor = None
        self.x_test_tensor = None
        self.y_train_tensor = None
        self.y_valid_tensor = None
        self.y_test_tensor = None
        self.scalar = None
        self.features = ['State', 'Time', 'ForestCover', 'MaxRiverOrder', 'PosCounty']
        self.target = 'InvasionStatus'
        self.read_data()
        self.prepare_data()
        self.prepare_test_data()
        self.split_data()
        self.test_prep()
        self.scale_data()
        self.tensor_data()


    def read_data(self):
        try:
            self.data = pd.read_excel('Lyme.xlsx')
            self.cdc = pd.read_excel('new_cdc.xlsx')
            self.abb = pd.read_excel('state_abb.xlsx')
        except Exception as e:
            print(e)

    def prepare_data(self):
        local = self.data.copy()
        local = local.reset_index()
        pos_counties = create_pos_county(local)
        self.data = local.merge(pos_counties, on=['index', 'Time'])

    def prepare_test_data(self):
        """
        pull cdc data and prepare it as the test set
        :return:
        """
        new = self.cdc.copy()
        new = new.rename(columns = {'Ctyname': 'County', 'Stname': 'State'})
        new = new[new['County'].str.contains('County')]
        new['County'] = new['County'].str.replace(' County', '', regex=True)
        new = new[['County', 'State', 'Cases2017', 'Cases2018', 'Cases2019']]
        new['InvasionStatus'] = new[['Cases2017', 'Cases2018', 'Cases2019']].values.max(1)
        new['InvasionStatus'] = new['InvasionStatus'].apply(lambda x: 1 if x > 0 else 0)
        new['Time'] = 11
        new = new[['County', 'State', 'Time', 'InvasionStatus']]
        new = new.merge(self.abb, on=['State'])
        new = new.drop('State', axis=1)
        new = new.rename(columns={'Abb': 'State'})
        new = new.merge(self.data.copy(), on=['State', 'County'], how='right')
        new = new.rename(columns={'Time_x': 'Time', 'InvasionStatus_x': 'InvasionStatus'})
        new = new[['State', 'Time', 'ForestCover', 'MaxRiverOrder', 'InvasionStatus', 'PosCounty']]
        self.test = new.copy()

    def scale_data(self):
        self.scalar = StandardScaler()
        self.x_train = self.scalar.fit_transform(self.x_train.to_numpy())
        self.x_valid = self.scalar.fit_transform(self.x_valid.to_numpy())
        self.x_test = self.scalar.fit_transform(self.x_test.to_numpy())

    def split_data(self):
        feature = self.data[self.features]
        x = pd.get_dummies(feature.copy())
        y = self.data[[self.target]]
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x, y, stratify=y, train_size=0.8,
                                                                                  shuffle=True, random_state=42)

    def test_prep(self):
        feature = self.test[self.features]
        x = pd.get_dummies(feature.copy())
        y = self.test[[self.target]]
        self.x_test, self.y_test = x.copy(), y.copy()


    def tensor_data(self):
        """
        The data types are wrong, y could be an int and save memory, but it's easier to work with the same dataset
        and the data size is small.
        :return:
        """
        self.x_train_tensor = torch.tensor(self.x_train.values.astype(float), dtype=torch.float)
        self.x_valid_tensor = torch.tensor(self.x_valid.values.astype(float), dtype=torch.float)
        self.x_test_tensor = torch.tensor(self.x_test.values.astype(float), dtype=torch.float)
        self.y_train_tensor = torch.tensor(self.y_train.values.astype(float), dtype=torch.float)
        self.y_valid_tensor = torch.tensor(self.y_valid.values.astype(float), dtype=torch.float)
        self.y_test_tensor = torch.tensor(self.y_test.values.astype(float), dtype=torch.float)
        self.x_train_tensor = torch.reshape(self.x_train_tensor,
                                            (self.x_train_tensor.shape[0], 1, self.x_train_tensor.shape[1]))
        self.x_valid_tensor = torch.reshape(self.x_valid_tensor,
                                            (self.x_valid_tensor.shape[0], 1, self.x_valid_tensor.shape[1]))
        self.x_test_tensor = torch.reshape(self.x_test_tensor,
                                            (self.x_test_tensor.shape[0], 1, self.x_test_tensor.shape[1]))
        self.y_train_tensor = self.y_train_tensor.reshape(-1)
        self.y_valid_tensor = self.y_valid_tensor.reshape(-1)
        self.y_test_tensor = self.y_test_tensor.reshape(-1)


class Grid_Search:
    def __init__(self, estimator, cv, param_grid, x_train, y_train, x_valid, y_valid):
        self.estimator = estimator
        self.cv = cv
        self.param_grid = param_grid
        self.gs = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.loader()
        self.get_acc()

    def loader(self):
        self.gs = GridSearchCV(
                    estimator=self.estimator,
                    param_grid=self.param_grid,
                    cv=self.cv,
                    scoring='accuracy'
        )

    def get_acc(self):
        self.gs.fit(self.x_train, self.y_train)
        train_acc = (self.gs.best_score_) * 100
        valid_acc = (self.gs.score(self.x_valid, self.y_valid)) * 100
        print(f'{self.gs.estimator} best parameters are {self.gs.best_params_}')
        print(f'{self.gs.estimator} training accuracy is {train_acc}%')
        print(f'{self.gs.estimator} validation accuracy is {valid_acc}%')
