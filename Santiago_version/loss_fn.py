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




