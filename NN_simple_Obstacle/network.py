import torch
import torch.nn as nn
from torch import functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(64, 64)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(64, 64)  # Second hidden layer to third hidden layer
        self.fc4 = nn.Linear(64, 2)  # Third hidden layer to output layer
        self.act = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function to the output of the first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation function to the output of the second layer
        x = torch.relu(self.fc3(x))  # Apply ReLU activation function to the output of the third layer
        x = self.act(self.fc4(x))  # Output layer (no activation function applied)
        return x

def create_nn():
    model = NeuralNetwork()
    return model