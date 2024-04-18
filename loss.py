import torch
import torch.nn as nn

class TrajectoryLoss(nn.Module):
    def __init__(self):
        super(TrajectoryLoss, self).__init__()

    def forward(self, x, y, tarjet):
        # Calculate the Euclidean distance between each point in the trajectory and the end goal
        loss = torch.mean((x-tarjet[0])**2 + (y-tarjet[1])**2)

        return loss
    

