from torch import nn
from main import LymeDisease
import torch

l = LymeDisease()

class rnn_linear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2,batch_first=True)
        self.fully_connected = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, hn = self.rnn(x)
        out = hn[-1, :, :]
        out = self.fully_connected(out)
        return out
