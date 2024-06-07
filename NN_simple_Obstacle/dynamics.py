import numpy as np
import torch

class ObjectMovement:
    def __init__(self, x, y, speed, angle):
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle

    def move_object(self, delta_speed, delta_angle):
        # Apply constraints on maximum change in speed and angle
        max_delta_speed = 0.6  # Maximum change in speed
        max_delta_angle = np.radians(20)  # Maximum change in angle (in radians)

        # Apply constraints on change in speed and angle
        delta_speed = max_delta_speed * delta_speed
        delta_angle = max_delta_angle * delta_angle

        dt = 1

        # Update speed and angle
        self.speed = self.speed + delta_speed
        self.angle = (self.angle + delta_angle + torch.pi) % (2*torch.pi) - torch.pi

        # Update x and y coordinates based on speed and angle
        new_x = self.x + self.speed * torch.cos(self.angle) * dt
        new_y = self.y + self.speed * torch.sin(self.angle) * dt

        return new_x, new_y, self.speed, self.angle