import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from parameters import Params




class ObjectMovement:
    def __init__(self, state,kinematics):
        self.x = state[0]
        self.y = state[1]
        self.speed = kinematics[0]
        self.angle = kinematics[1]
        self.max_delta_speed = Params['max_acc']   # Maximum change in speed
        self.max_delta_angle = Params['max_ang_acc']  # Maximum change in angle (in radians)
        self.dt = Params['dt']
        self.nn_coeff_speed = Params['nn_coeff_speed']
        self.nn_coeff_angle = Params['nn_coeff_angle']

    def dynamics(self, u, PD):
        delta_speed_nn=u[0]
        delta_angle_nn=u[1]
        delta_speed_P=PD[0]
        delta_angle_P=PD[1]

        delta_speed = self.nn_coeff_speed * delta_speed_nn + delta_speed_P
        delta_angle = self.nn_coeff_angle * delta_angle_nn + delta_angle_P


        # Apply constraints on change in speed and angle
        delta_speed = torch.max(-self.max_delta_speed,torch.min(self.max_delta_speed, delta_speed))
        delta_angle = torch.max(-self.max_delta_angle,torch.min(self.max_delta_angle, delta_angle))


        # Update speed and angle in the [-π, π] range
        self.speed = self.speed + delta_speed
        self.angle = (self.angle + delta_angle + torch.pi) % (2*torch.pi) - torch.pi
        #self.angle = self.angle + delta_angle

        # Update x and y coordinates based on speed and angle
        new_x = self.x + self.speed * torch.cos(self.angle) * self.dt
        new_y = self.y + self.speed * torch.sin(self.angle) * self.dt

        new_position=torch.stack((new_x,new_y),0)
        new_kinematics=torch.stack((self.speed,self.angle),0)

        return new_position, new_kinematics
    
