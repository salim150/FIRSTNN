#Code for the function to avoid collision between cars

import numpy as np
import torch
import torch.nn as nn
from parameters import Params

class CBF:
    def __init__(self, x_car1, y_car1, x_car2, y_car2, speed_car1, angle_car1, speed_car2, angle_car2, environment_limits):
        self.x_car1 = x_car1
        self.y_car1 = y_car1
        self.x_car2 = x_car2
        self.y_car2 = y_car2
        self.speed_car1 = speed_car1
        self.angle_car1 =angle_car1
        self.speed_car2 = speed_car2
        self.angle_car2 = angle_car2

        self.environment_limtis = 'Environment_limits'
        self.trajectory_crossing_value = Params['trajectory_crossing_value']

    def barrier_function(x_car1, y_car1, x_car2, y_car2, safety_distance):
        distance = torch.sqrt((x_car1 - x_car2)**2 + (y_car1 - y_car2)**2)
        #Define barrier function based on the distance
        barrier_value = torch.maximum(torch.zeros_like(distance), distance - safety_distance)

        return barrier_value
    
    def forward(barrier_value):
        if barrier_value > 3: #Can change the value of the threshold
            pass              #If the barrier value is below the threshold then nothing happens
        else:                 #If it is smaller than the threshold then we implement a penalty (can be quadratic or exponential etc etc..)
            pass
    

     


