import torch.nn
from torch import nn
import torch.optim as optim
from main import LymeDisease
import pandas as pd

l = LymeDisease()
def train(hidden_size: int, num_layers: int, lr: int = 0.01, epochs: int = 200):
    x_train = l.x_train_tensor
    y_train = l.y_train_tensor
    model = lstm_dense(x_train.shape[2], hidden_size=hidden_size, num_layers=num_layers, h_s2=125)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model.forward(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs} has MSE of: {loss.item()}')
    return model


def _transform(data, y: pd.Series)->tuple:
    pred = data.detach().numpy().tolist()
    pred = [1 if x > 0.5 else 0 for x in pred]
    dataframe = pd.DataFrame({'obs': y, 'pred': pred})
    count=0
    for obs, pred in zip(dataframe['obs'], dataframe['pred']):
        if obs==pred:
            count+=1
    accuracy = (count/dataframe.shape[0])*100
    return accuracy, dataframe


def predict(model, valid: bool=True)->pd.DataFrame:
    """
    :param model: model using feed forward architecture
    :param valid: Keep true unless you wish to use test set
    :return: pd.DataFrame
    """
    type = 'Validation'
    x_valid = l.x_valid_tensor
    y_valid = l.y_valid
    x_test = l.x_test_tensor
    y_test = l.y_test
    if valid:
        data = model(x_valid)
        accuracy, out = _transform(data, y_valid)
    else:
        type = 'Test'
        data = model(x_test)
        accuracy, out = _transform(data, y_test)
    print('%s accuracy is %.2f%%' % (type, accuracy))
    return out


class lstm_linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fully_connected = nn.Linear(hidden_size, 1)
        self.ReLU = nn.ReLU()
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))
        #out = self.ReLU(out)
        out = out[:, -1, :]
        out = self.fully_connected(out).flatten()
        return out


class lstm_dense(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, h_s2):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, h_s2)
        self.fc2 = nn.Linear(h_s2, h_s2)
        self.fc3 = nn.Linear(h_s2, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.ReLU(out)
        out = self.fc2(out)
        out = self.ReLU(out)
        out = self.fc3(out).flatten()
        return out

