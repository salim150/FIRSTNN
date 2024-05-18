import torch
from parameters import Params

class ObjectMovement_2:
    def __init__(self, x, y, speed, angle):
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle
        self.max_delta_speed = torch.tensor(2)  # Maximum change in speed
        self.max_delta_angle = torch.tensor(torch.pi / 4)  # Maximum change in angle (in radians)
        self.dt = 0.2
        self.nn_coeff_speed = Params['nn_coeff_speed']
        self.nn_coeff_angle = Params['nn_coeff_angle']

    def move_object(self, delta_speed_nn, delta_angle_nn, delta_speed_P, delta_angle_P):
        
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

        return new_x, new_y, self.speed, self.angle