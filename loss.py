import torch
import torch.nn as nn

class TrajectoryLoss(nn.Module):
    def __init__(self):
        super(TrajectoryLoss, self).__init__()

    def forward(self, x, y, end_goalx, end_goaly):
        # Calculate the Euclidean distance between each point in the trajectory and the end goal
        distances = torch.sqrt((x-end_goalx)**2 + (y-end_goaly)**2)
        
        # Sum up the distances to get the total loss
        loss = torch.sum(distances)
        
        return loss