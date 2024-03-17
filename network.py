import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 20)  # Input layer to hidden layer
        self.fc2 = nn.Linear(20, 2)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function to the output of the first layer
        x = self.fc2(x)  # Output layer (no activation function applied)
        return x

def create_nn():
    model = NeuralNetwork()
    return model