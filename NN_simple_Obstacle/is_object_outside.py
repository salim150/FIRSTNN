import numpy as np
import torch
import torch.nn as nn
from parameters import Params

class determine_minDist_boundary(nn.Module):
    def __init__(self):
        super(determine_minDist_boundary, self).__init__()
        self.xmin = Params['Environment_limits'][0][0].clone().detach()
        self.ymin = Params['Environment_limits'][1][0].clone().detach()
        self.xmax = Params['Environment_limits'][0][1].clone().detach()
        self.ymax = Params['Environment_limits'][1][1].clone().detach()
        
    def forward(self, x, y):
        # Déterminer si l'objet est sur le terrain ou pas
        if (x>self.xmin) and (x<self.xmax) and (y>self.ymin) and (y<self.ymax) :
            return 0
        #Déterminer la distance minimal à un des bords
        minDist = torch.tensor(0)
        if x<self.xmin :
            minDist = torch.max(minDist, self.xmin-x)
        if y<self.ymin :
            minDist = torch.max(minDist, self.ymin-y)
        if x>self.xmax :
            minDist = torch.max(minDist, x-self.xmax)
        if y>self.ymax :
            minDist = torch.max(minDist, y-self.ymax)
        
        return minDist