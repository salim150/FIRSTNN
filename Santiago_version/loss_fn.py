import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from parameters import Params
from is_object_outside import determine_minDist_boundary


class loss_fn(nn.Module):
    def __init__(self):
        super(loss_fn, self).__init__()

    def forward(self, x, y):
        # Calculate the Euclidean distance between each point in the trajectory and the end goal
        C=nn.MSELoss()
        loss = C(x,y)

        return loss

 

"""
import numpy as np

# Définition des constantes
alpha = 1
beta = 10
gamma = 20

outside_penalty_value = 500  # Valeur de la pénalité pour sortir du terrain
obstacle_penalty_value = 1000  # Valeur de la pénalité pour toucher un obstacle
 

# Coordonnées du point d'arrivée et de l'obstacle
x_goal, y_goal = 10, 10
x_obstacle, y_obstacle = 5, 5
obstacle_size = 3

# Limites de la zone
x_min, x_max = 0, 20
y_min, y_max = 0, 20

high_value = 10000000 # éviter asymptote de la fonction

x_robot, y_robot = 1, 1 # Coordonnées au hasard

# Loss Function
def loss_function(x_robot, y_robot):
    # Distance par rapport au point d'arrivée
    distance_to_goal = np.sqrt((x_robot - x_goal)**2 + (y_robot - y_goal)**2) # Or without sqrt
    
    # Pénalité pour sortir de la zone
    if x_robot < x_min or x_robot > x_max or y_robot < y_min or y_robot > y_max:
        terrain_penalty = outside_penalty_value
    else:
        terrain_penalty = 0
    
     # Pénalité pour toucher un obstacle
    if (x_robot - x_obstacle)**2 + (y_robot - y_obstacle)**2 <= obstacle_size**2:
        obstacle_penalty = obstacle_penalty_value
    else:
        obstacle_penalty = 0

   

    # Using differentiable function
    terrain_penalty = min(high_value,-np.log(1 - np.exp(-outside_penalty_value * (min(x_robot - x_min, x_max - x_robot, y_robot - y_min, y_max - y_robot)))))
    obstacle_penalty = min(high_value,-np.log(1 - np.exp(-obstacle_penalty_value * ((x_robot - x_obstacle)**2 + (y_robot - y_obstacle)**2 - obstacle_size**2))))



 # Calcul loss
    loss = alpha * distance_to_goal + beta * terrain_penalty + gamma*obstacle_penalty

    return loss

"""




class loss_fn2(nn.Module):
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
        self.outside_penalty_value = Params['outside_penalty_value'] 
        self.obstacle_penalty_value = Params['obstacle_penalty_value']
        self.high_value = torch.tensor(Params['high_value'], dtype=torch.float32)

        self.minDist_boundary = determine_minDist_boundary()
        
    def forward(self, x, y,xobs,yobs,x_goal,y_goal, time):
        # Déterminer si l'objet est sur le terrain
        minDist = self.minDist_boundary(x,y)
        if minDist :
            terrain_penalty = self.high_value + 100*minDist
        else :
            terrain_penalty = torch.min(self.high_value, -torch.log(1-torch.exp(-self.outside_penalty_value *
            torch.min(torch.min(x-self.xmin, y-self.ymin),torch.min(self.xmax-x, self.ymax-y)))))
        # déterminer si l'objet est dans l'obstacle
        if ((x-xobs)**2 + (y-yobs)**2 < self.obssize**2) :
            obstacle_penalty = self.high_value + 5000 - ((x-xobs)**2 + (y-yobs)**2)*5000/self.obssize**2
        else :
            obstacle_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(-self.obstacle_penalty_value * 
            ((x - xobs) ** 2 + (y - yobs) ** 2) - self.obssize**2)))
                

        distance_to_goal = ((x - x_goal) ** 2 + (y - y_goal) ** 2)
    

        loss = self.alpha * distance_to_goal + self.beta * terrain_penalty + self.gamma * obstacle_penalty
        

        return loss