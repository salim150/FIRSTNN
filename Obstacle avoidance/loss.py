import torch
import torch.nn as nn

class TrajectoryLoss(nn.Module):
    def __init__(self):
        super(TrajectoryLoss, self).__init__()

    def forward(self, x, y, end_goalx, end_goaly):
        # Calculate the Euclidean distance between each point in the trajectory and the end goal
        loss = (x-end_goalx)**2 + (y-end_goaly)**2
        
        return loss