import torch.nn
from torch import nn
import torch.optim as optim
from main import LymeDisease

l = LymeDisease()

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

model = lstm_linear(input_size = 13, hidden_size=125, num_layers=2)
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
x_train = l.x_train_tensor
y_train = l.y_train_tensor
for epoch in range(500):
    optimizer.zero_grad()
    out = model.forward(x_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'At epoch {epoch}, loss is: {loss.item()}')

class lstm_dense(nn.Module):
    def __init__(self, hidden_size, h_s2):
        super().__init__()
        self.lstm = nn.LSTM()
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, h_s2)
        self.fc2 = nn.Linear(h_s2, h_s2)
        self.fc3 = nn.Linear(h_s2, 1)

    def forward(self, x):
        h0 = torch.zeros()

