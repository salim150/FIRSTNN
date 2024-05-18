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
        d1=torch.min(self.xmax-x,torch.tensor(0))
        d2=torch.min(x-self.xmin,torch.tensor(0))
        d3=torch.min(self.ymax-y,torch.tensor(0))
        d4=torch.min(y-self.ymin,torch.tensor(0))



        d5=torch.min(d1,d2)
        d6=torch.min(d3,d4)

        minDist_out=torch.min(d5,d6)


        d1=torch.max(self.xmax-x,torch.tensor(0))
        d2=torch.max(x-self.xmin,torch.tensor(0))
        d3=torch.max(self.ymax-y,torch.tensor(0))
        d4=torch.max(y-self.ymin,torch.tensor(0))

        d5=torch.min(d1,d2)
        d6=torch.min(d3,d4)

        minDist_in= torch.min(d5,d6)


        
        return minDist_out,minDist_in