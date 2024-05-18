import torch.nn as nn
from torch import functional as F


def init_weights_zeros(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0)
        nn.init.constant_(m.bias, 0)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 64)   # Input layer to first hidden layer
        self.fc2 = nn.Linear(64, 64)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(64, 64)  # Second hidden layer to third hidden layer
        self.fc4 = nn.Linear(64, 2)   # Third hidden layer to output layer
        self.act1 = nn.Tanh()
        self.act2 = nn.LeakyReLU(negative_slope = 0.01)

    def forward(self, x):
        x = self.act2(self.fc1(x))  # Apply ReLU activation function to the output of the first layer
        x = self.act1(self.fc2(x))  # Apply ReLU activation function to the output of the second layer
        x = self.act2(self.fc3(x))  # Apply ReLU activation function to the output of the third layer
        x = self.act1(self.fc4(x))  # Output layer (no activation function applied)
        return x

def create_nn_4():
    model = NeuralNetwork()
    model.apply(init_weights_zeros) # Apply zero initialization to all weights
    return model