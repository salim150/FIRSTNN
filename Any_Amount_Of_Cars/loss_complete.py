import numpy as np
import torch
import torch.nn as nn
from parameters import Params
from is_object_outside import determine_minDist_boundary

class loss_fn(nn.Module):
    def __init__(self):
        super(loss_fn, self).__init__()
        self.alpha = Params['alpha']
        self.beta = Params['beta']
        self.delta = Params['delta']
        self.xmin = Params['Environment_limits'][0][0].clone().detach()
        self.ymin = Params['Environment_limits'][1][0].clone().detach()
        self.xmax = Params['Environment_limits'][0][1].clone().detach()
        self.ymax = Params['Environment_limits'][1][1].clone().detach()

        self.high_value = torch.tensor(Params['high_value'], dtype=torch.float32)

        self.safety = Params['collision_safety']
        self.car_size = Params['car_size']

        self.minDist_boundary = determine_minDist_boundary()
        
    def forward(self, x, y, x_other, y_other, x_goal, y_goal, car_idx, other_car_idx):

        terrain_penalty = 0
        distance_to_goal = 0
        collision_penalty = 0

        if (car_idx==other_car_idx) :
        # Déterminer si l'objet est sur le terrain
            minDist = self.minDist_boundary(x,y)
            if minDist :
                terrain_penalty = self.high_value + 100*minDist
            else :
                terrain_penalty = 0
                #terrain_penalty = torch.min(self.high_value, -torch.log(1-torch.exp(-self.outside_penalty_value *
                #torch.min(torch.min(x-self.xmin, y-self.ymin),torch.min(self.xmax-x, self.ymax-y)))))
            distance_to_goal = ((x - x_goal) ** 2 + (y - y_goal) ** 2)
        
        # déterminer s'il y a collision
        distance_squared = (x - x_other)**2 + (y - y_other)**2
        collision_threshold = (2 * self.car_size)**2
        safety_threshold = (2 * self.car_size + self.safety)**2

        if distance_squared < collision_threshold:
            collision_penalty = self.high_value + 1000 - (distance_squared * 1000 / collision_threshold)
        elif distance_squared < safety_threshold:
            normalized_distance = (distance_squared - collision_threshold) / (safety_threshold - collision_threshold)
            # Smoothly decrease from high_value to 0 using a quadratic decay function
            collision_penalty = self.high_value * (1 - normalized_distance)**2
    

        loss = self.alpha * distance_to_goal + self.beta * terrain_penalty + self.delta * collision_penalty
        

        return loss