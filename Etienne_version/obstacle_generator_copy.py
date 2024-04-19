import numpy as np
from parameters import Params
import torch

#Class to generate obstacles 
class Obstacle_generator():
    def __init__(self):
        self.min_coord = Params['Environment_limits'][0][0]
        self.max_coord = Params['Environment_limits'][0][1]
        self.obssize = Params['obssize']

    def generate_obstacle(self):
        object_middle = [np.random.uniform(low=self.min_coord+self.obssize, high=self.max_coord-self.obssize), np.random.uniform(low=self.min_coord, high=self.max_coord)]
        return torch.tensor(object_middle)