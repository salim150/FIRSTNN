import torch
from parameters import Params

class ObjectMovement:
    def __init__(self, x, y, speed, angle):
        self.x = x # Initialize the x-coordinate of the object
        self.y = y # Initialize the y-coordinate of the object
        self.speed = speed # Initialize the speed of the object
        self.angle = angle # Initialize the angle of the object
        self.max_delta_speed = torch.tensor(2)  # Maximum change in speed
        self.max_delta_angle = torch.tensor(torch.pi / 4)  # Maximum change in angle (in radians)
        self.dt = Params['dt']  # Time step for updating the position, retrieved from Params
        self.nn_coeff_speed = Params['nn_coeff_speed'] # Coefficient for neural network-based speed adjustment
        self.nn_coeff_angle = Params['nn_coeff_angle'] # Coefficient for neural network-based angle adjustment

    # Define the method to move the object
    def move_object(self, delta_speed_nn, delta_angle_nn, delta_speed_P, delta_angle_P):

        # Calculate the change in speed and angle using both neural network and proportional control inputs
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

         # Return the new position (x, y), speed, and angle of the object
        return new_x, new_y, self.speed, self.angle
    



