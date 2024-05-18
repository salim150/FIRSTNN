import numpy as np
import torch
import torch.nn as nn
from parameters import Params
from is_object_outside import determine_minDist_boundary

class loss_fn_1(nn.Module):
    def __init__(self):
        super(loss_fn_1, self).__init__()
        self.alpha = Params['alpha']
        self.beta = Params['beta']
        self.gamma = Params['gamma']
        self.delta = Params['delta']
        self.xmin = Params['Environment_limits'][0][0].clone().detach()
        self.ymin = Params['Environment_limits'][1][0].clone().detach()
        self.xmax = Params['Environment_limits'][0][1].clone().detach()
        self.ymax = Params['Environment_limits'][1][1].clone().detach()

        self.obssize = Params['obssize']
        self.outside_penalty_value = Params['outside_penalty_value'] 
        self.obstacle_penalty_value = Params['obstacle_penalty_value']
        self.collision_penalty_value = Params['collision_penalty_value']
        self.high_value = torch.tensor(Params['high_value'], dtype=torch.float32)

        self.safety = Params['collision_safety']

        self.minDist_boundary = determine_minDist_boundary()
        
    def forward(self, x, y, x_other, y_other, x_goal, y_goal,j):
        # Déterminer si l'objet est sur le terrain
        minDist = self.minDist_boundary(x,y)
        if minDist :
            terrain_penalty = self.high_value + 10000*minDist
        else :
            terrain_penalty = torch.min(self.high_value, -torch.log(1-torch.exp(-self.outside_penalty_value *
            torch.min(torch.min(x-self.xmin, y-self.ymin),torch.min(self.xmax-x, self.ymax-y)))))
    
        '''# déterminer si l'objet est dans l'obstacle
        if ((x-xobs)**2 + (y-yobs)**2 < self.obssize**2) :
            obstacle_penalty = self.high_value + 10000 - ((x-xobs)**2 + (y-yobs)**2)*10000/self.obssize**2
        else :
            obstacle_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(-self.obstacle_penalty_value * 
            ((x - xobs) ** 2 + (y - yobs) ** 2) - self.obssize**2)))'''
        
        # déterminer s'il y a collision
        if ((x-x_other)**2 + (y_other)**2 < self.safety**2) :
            collision_penalty = self.high_value + 10000 - ((x-x_other)**2 + (y-y_other)**2)*10000/self.safety**2
        else :
            collision_penalty = torch.min(self.high_value, -torch.log(1 - torch.exp(-self.collision_penalty_value * 
            ((x - x_other) ** 2 + (y - y_other) ** 2) - self.safety**2)))
                

        distance_to_goal = ((x - x_goal) ** 2 + (y - y_goal) ** 2) 
    

        loss = self.alpha * distance_to_goal + self.beta * terrain_penalty + self.delta * collision_penalty
        

        return loss