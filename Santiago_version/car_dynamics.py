import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from parameters import Params


class Car_dyn:

    def __init__(self, A:torch.Tensor, B: torch.Tensor):

        self.A = A
        self.B = B

    # evaluation of the next state given current state and input
    def dynamics(self, xk:torch.Tensor, u:torch.Tensor) -> torch.Tensor:
        x_next = self.A @ xk + self.B @ u
        return x_next



class ObjectMovement:
    def __init__(self, state,kinematics):
        self.x = state[0]
        self.y = state[1]
        self.speed = kinematics[0]
        self.angle = kinematics[1]

    def dynamics(self, u):
        delta_speed=u[0]
        delta_angle=u[1]
        # Apply constraints on maximum change in speed and angle
        max_delta_speed = Params['max_acc']  # Maximum change in speed
        max_delta_angle = Params['max_ang_acc']  # Maximum change in angle (in radians)

        # Apply constraints on change in speed and angle
        delta_speed = max_delta_speed * delta_speed
        delta_angle = max_delta_angle * delta_angle

        dt = 1

        # Update speed and angle
        self.speed = self.speed + delta_speed
        self.angle = self.angle + delta_angle

        # Update x and y coordinates based on speed and angle
        new_x = self.x + self.speed * torch.cos(self.angle) * dt
        new_y = self.y + self.speed * torch.sin(self.angle) * dt

        new_position=torch.stack((new_x,new_y),0)
        new_kinematics=torch.stack((self.speed,self.angle),0)

        return new_position, new_kinematics
    


class ObjectMovement2:
    def __init__(self, state,kinematics):
        self.x = state[:,0]
        self.y = state[:,1]
        self.speed = kinematics[:,0]
        self.angle = kinematics[:,1]


    def dynamics(self, u):
        delta_speed=u[:,0]
        delta_angle=u[:,1]
        # Apply constraints on maximum change in speed and angle
        max_delta_speed = Params['max_acc']  # Maximum change in speed
        max_delta_angle = Params['max_ang_acc']  # Maximum change in angle (in radians)

        # Apply constraints on change in speed and angle
        delta_speed = max_delta_speed * delta_speed
        delta_angle = max_delta_angle * delta_angle

        dt = 1

        # Update speed and angle
        self.speed = self.speed + delta_speed
        self.angle = self.angle + delta_angle

        # Update x and y coordinates based on speed and angle
        new_x = self.x + self.speed * torch.cos(self.angle) * dt
        new_y = self.y + self.speed * torch.sin(self.angle) * dt

        new_position=torch.stack((new_x,new_y),0)
        new_kinematics=torch.stack((self.speed,self.angle),0)

        return torch.transpose(new_position,0,1), torch.transpose(new_kinematics,0,1)
