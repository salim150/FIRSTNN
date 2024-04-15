import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class loss_fn(nn.Module):
    def __init__(self,alpha,beta,gamma,xmin,ymin,xmax,ymax,obssize,outside_penalty_value,obstacle_penalty_value,high_value):
        super(loss_fn, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.obssize = obssize
        self.outside_penalty_value = outside_penalty_value  # Valeur de la pénalité pour sortir du terrain
        self.obstacle_penalty_value = obstacle_penalty_value  # Valeur de la pénalité pour toucher un obstacle
        self.high_value = high_value # éviter asymptote de la fonction

    def forward(self, x, y,xobs,yobs,x_goal,y_goal):
        # Calculate the Euclidean distance between each point in the trajectory and the end goal
        # Pénalité pour sortir de la zone
        terrain_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(
        -self.outside_penalty_value * (torch.min(x - self.xmin, self.xmax - x, y - self.ymin, self.ymax - y)))))
        distance_to_goal = (x - x_goal) ** 2 + (y - y_goal) ** 2

    # Pénalité pour toucher un obstacle
        obstacle_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(-self.obstacle_penalty_value * ((x - xobs) ** 2 + (y - yobs) ** 2 -
                                   self.obssize ** 2))))
    

        loss = self.alpha *distance_to_goal + self.beta * terrain_penalty + self.gamma * obstacle_penalty
        

        return loss