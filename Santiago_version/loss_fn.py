import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from parameters import Params
from is_object_outside import determine_minDist_boundary


class loss_fn2(nn.Module):
    def __init__(self):
        super(loss_fn2, self).__init__()
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

        self.safety = Params['collision_safety']
        self.car_size = Params['car_size']

        self.minDist_boundary = determine_minDist_boundary()

    def forward(self, pos,obs_pos,target):
        x=pos[:,0:1]
        y=pos[:,1:2]
        xobs=obs_pos[:,0:1]
        yobs=obs_pos[:,1:2]
        x_goal=target[:,0:1]
        y_goal=target[:,1:2]

        minDist= self.minDist_boundary(x,y)

        d_1=torch.tensor(self.car_size+self.safety)

        f1=self.high_value -1000*(minDist-d_1)

        terrain_penalty =torch.max(f1,torch.tensor(0))

        
        d_square= (x-xobs)**2 + (y-yobs)**2
        d_1_square= torch.tensor((self.obssize+self.car_size)**2)
        d_2_square= torch.tensor((self.obssize+self.car_size+self.safety)**2)
        f1=self.high_value*torch.log(d_square/d_2_square)/torch.log(d_1_square/d_2_square)
        f2= 10000+ d_square*(self.high_value-10000)/d_1_square
        obstacle_penalty= torch.max(torch.min(f1,torch.max(f2,self.high_value)),torch.tensor(0))



        distance_to_goal = ((x - x_goal) ** 2 + (y - y_goal) ** 2)

        loss = self.alpha * distance_to_goal + self.beta*terrain_penalty + self.gamma * obstacle_penalty


        return loss