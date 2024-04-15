import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class loss_fn(nn.Module):
    def __init__(self,alpha,beta,gamma,xmin,ymin,obssize,outside_penalty_value,obstacle_penalty_value):
        super(loss_fn, self).__init__()
        self.alpha = 1
        self.beta = 2
        self.gamma = 10
        self.xmin = -10
        self.ymin = -10
        self.xmax = 10
        self.ymax = 10
        self.obssize = 3
        self.outside_penalty_value = 200  # Valeur de la pénalité pour sortir du terrain
        self.obstacle_penalty_value = 500  # Valeur de la pénalité pour toucher un obstacle
        self.high_value = 100000  # éviter asymptote de la fonction

    def forward(self, x, y,xobs,yobs,x_goal,y_goal):
        # Calculate the Euclidean distance between each point in the trajectory and the end goal
        # Pénalité pour sortir de la zone
        terrain_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(
        -outside_penalty_value * (torch.min(x - self.xmin, self.xmax - x, y - self.ymin, self.ymax - y)))))
        distance_to_goal = (x - x_goal) ** 2 + (y - y_goal) ** 2

    # Pénalité pour toucher un obstacle
        obstacle_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(-obstacle_penalty_value * ((x - xobs) ** 2 + (y - yobs) ** 2 -
                                   self.obssize ** 2))))
    

        loss = self.alpha *distance_to_goal + self.beta * terrain_penalty + self.gamma * obstacle_penalty
        

        return loss