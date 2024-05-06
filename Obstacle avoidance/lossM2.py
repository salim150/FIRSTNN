import numpy as np
import torch
import torch.nn as nn
from parameters import Params

class loss_fn(nn.Module):
    def __init__(self):
        super(loss_fn, self).__init__()
        self.alpha = Params['alpha']
        self.beta = Params['beta']
        self.gamma = Params['gamma']
        self.xmin = Params['Environment_limits'][0][0].clone().detach()
        self.ymin = Params['Environment_limits'][1][0].clone().detach()
        self.xmax = Params['Environment_limits'][0][1].clone().detach()
        self.ymax = Params['Environment_limits'][1][1].clone().detach()

        self.obssize = Params['obssize']
        self.outside_penalty_value = Params['outside_penalty_value']  # Valeur de la pénalité pour sortir du terrain
        self.obstacle_penalty_value = Params['obstacle_penalty_value']  # Valeur de la pénalité pour toucher un obstacle
        self.high_value = torch.tensor(Params['high_value'], dtype=torch.float32)
        
    def forward(self, x, y,xobs,yobs,x_goal,y_goal, time):
        # Calculate the Euclidean distance between each point in the trajectory and the end goal
        # Pénalité pour sortir de la zone
        distXmin = abs(x-self.xmin)
        distYmin = abs(y-self.ymin)
        distXmax = abs(x-self.xmax)
        distYmax = abs(y-self.ymax)
        terrain_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(
        -self.outside_penalty_value * torch.min(torch.min(distXmin, distXmax), torch.min(distYmin, distYmax)))))

        distance_to_goal = ((x - x_goal) ** 2 + (y - y_goal) ** 2) * time

        # Pénalité pour toucher un obstacle
        obstacle_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(-self.obstacle_penalty_value * 
                                    ((x - xobs) ** 2 + (y - yobs) ** 2))))
    

        loss = self.alpha * distance_to_goal + self.beta * terrain_penalty + self.gamma * obstacle_penalty
        

        return loss