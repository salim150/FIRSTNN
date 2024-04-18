import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
 


class NeuralNetwork(nn.Module):

    def __init__(self, N):
        """
        Ni - Input size
        Nh1 - Neurons in the 1st hidden layer
        Nh2 - Neurons in the 2nd hidden layer
        No - Output size
        """
        super().__init__()
        # Definition of the hidden layers with normal initialization
        self.fc1 = nn.Linear(in_features = N[0], out_features = N[1])
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)

        self.fc2 = nn.Linear(in_features = N[1], out_features = N[2])
        #nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

        self.fc3 = nn.Linear(in_features = N[2], out_features = N[3])
        #nn.init.normal_(self.fc3.weight, mean=0.0, std=1.0)

        self.out = nn.Linear(in_features=N[3], out_features=N[4])
        #nn.init.normal_(self.out.weight, mean=0.0, std=1.0)

        # Activation function (Sigmoid, ReLU, tanh...)
        self.act = nn.ReLU()
        self.act2 = nn.Tanh()

    # Forward pass of the network, given input return the output (in our case, given current state generate the control signal u)
    def forward(self, x: torch.Tensor, additional_out=False) -> torch.Tensor:
        u = self.act(self.fc1(x))
        u = self.act(self.fc2(u))
        u = self.act(self.fc3(u))
        u = self.act2(self.out(u))
        return u