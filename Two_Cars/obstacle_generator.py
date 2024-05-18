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
        self.radius = Params['start_radius']

    def generate_obstacle(self, x_start, y_start, x_end, y_end):
        while (True) :
            # generating the object randomly
            object_middle = [np.random.uniform(low=self.Xmin_coord, high=self.Xmax_coord), np.random.uniform(low=self.Ymin_coord, high=self.Ymax_coord)]
            # checking if the object isn't on top of the starting or ending points
            if ((x_start-object_middle[0])**2 + (y_start-object_middle[1])**2 > (self.obssize + self.radius) **2):
                if ((x_end-object_middle[0])**2 + (y_end-object_middle[1])**2 > self.obssize **2):
                    break
        return torch.tensor(object_middle)