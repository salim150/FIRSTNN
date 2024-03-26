import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

'''
class ObjectMovement:
    def __init__(self, state):
        self.x = state[0]
        self.y = state[1]
        self.speed = state[2]
        self.angle = state[3]

    def move_object(self, controller_output, parameters=torch.tensor([0.2,1,20])):
        # Apply constraints on maximum change in speed and angle
        max_delta_speed = parameters[0] # Maximum change in speed
        max_speed = parameters[1]
        max_delta_angle = parameters[2].deg2rad() # Maximum change in angle (in radians)

        # Apply constraints on change in speed and angle
        delta_speed = torch.min(torch.max(controller_output[0], -max_delta_speed), max_delta_speed)
        delta_angle = torch.min(torch.max(controller_output[1], -max_delta_angle), max_delta_angle)
        
        # Update speed and angle
        newspeed=torch.min(torch.max(self.speed + delta_speed, -max_speed), max_speed)
        newangle=self.angle + delta_angle
        self.speed = newspeed
        self.angle = newangle

        # Update x and y coordinates based on speed and angle

        newx=self.x + self.speed * torch.cos((self.angle))
        newy=self.y + self.speed * torch.sin((self.angle))
        self.x = newx
        self.y = newy

        new_state= torch.cat((self.x.unsqueeze(0), self.y.unsqueeze(0), self.speed.unsqueeze(0), self.angle.unsqueeze(0)),0)

        return new_state
'''

class Car_dyn:

    def __init__(self, A:torch.Tensor, B: torch.Tensor):

        self.A = A
        self.B = B

    # evaluation of the next state given current state and input
    def dynamics(self, xk:torch.Tensor, u:torch.Tensor) -> torch.Tensor:
        x_next = self.A @ xk + self.B @ u
        return x_next
