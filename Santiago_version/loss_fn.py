import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class loss_fn(nn.Module):
    def __init__(self):
        super(loss_fn, self).__init__()

    def forward(self, x, y):
        # Calculate the Euclidean distance between each point in the trajectory and the end goal
        C=nn.MSELoss()
        loss = C(x,y)

        return loss
    
