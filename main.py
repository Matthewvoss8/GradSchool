from utils import Lyme
import torch

def run():
    l = Lyme()
    model = Logistic(l.format)
    return model

class Logistic():
    def __init__(self, data):
        self.data = data
        self.logisticRegression()
        self.graph()

    def logisticRegression(self):
        pass

    def graph(self):
        pass

    def train(self):
        forward(self.data)



if __name__ == '__main__':
    run()
