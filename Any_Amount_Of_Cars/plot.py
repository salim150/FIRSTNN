import torch.nn as nn
from torch import functional as F
import graphviz

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
        x = self.act2(self.fc1(x))  # Apply LeakyReLU activation function to the output of the first layer
        x = self.act1(self.fc2(x))  # Apply Tanh activation function to the output of the second layer
        x = self.act2(self.fc3(x))  # Apply LeakyReLU activation function to the output of the third layer
        x = self.act1(self.fc4(x))  # Output layer
        return x

def create_nn():
    model = NeuralNetwork()
    model.apply(init_weights_zeros) # Apply zero initialization to all weights
    return model

def visualize_nn():
    dot = graphviz.Digraph()

    # Adding nodes with layer information
    dot.node('Input', 'Input Layer\n(size=6)')
    dot.node('H1', 'Hidden Layer 1\n(size=64)\nActivation: LeakyReLU')
    dot.node('H2', 'Hidden Layer 2\n(size=64)\nActivation: Tanh')
    dot.node('H3', 'Hidden Layer 3\n(size=64)\nActivation: LeakyReLU')
    dot.node('Output', 'Output Layer\n(size=2)\nActivation: Tanh')

    # Adding edges to represent connections between layers
    dot.edges([('Input', 'H1'), ('H1', 'H2'), ('H2', 'H3'), ('H3', 'Output')])

    # Save and render the graph
    dot.render('nn_architecture', format='png', cleanup=False)

if __name__ == "__main__":
    model = create_nn()
    visualize_nn()
