import numpy as np
from parameters import Params
import torch

#Class to generate obstacles 
class Obstacle_generator():
    def __init__(self):
        self.obssize = Params['obssize']
        self.Xmin_coord = Params['Environment_limits'][0][0] + self.obssize
        self.Xmax_coord = Params['Environment_limits'][0][1] - self.obssize
        self.Ymin_coord = Params['Environment_limits'][1][0] + self.obssize
        self.Ymax_coord = Params['Environment_limits'][1][1] - self.obssize

    def generate_obstacle(self):
        object_middle = [np.random.uniform(low=self.Xmin_coord, high=self.Xmax_coord), np.random.uniform(low=self.Ymin_coord, high=self.Ymax_coord)]
        return torch.tensor(object_middle)