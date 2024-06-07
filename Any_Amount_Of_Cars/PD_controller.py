from parameters import Params
import torch

class controller():
    def __init__(self):
        super(controller, self).__init__()
        self.K_pd = Params['Prop_coeff_distance']
        self.K_pa = Params['Prop_coeff_angle']
        self.K_pv = Params['Prop_coeff_speed']

    def forward(self, x, y, x_end, y_end, current_speed, current_angle):
        distance = torch.sqrt((x-x_end)**2 + (y-y_end)**2)

        angle = torch.arctan2((y_end-y),(x_end-x))
        delta_angle = angle - current_angle

        # Adjust the angle difference to be within [-π, π] radians
        #delta_angle = (delta_angle + torch.pi) % (2 * torch.pi) - torch.pi
        delta_angle = angle - current_angle
        # Adjust delta_angle to be within [-π, π] radians
        if delta_angle > torch.pi:
            delta_angle -= 2 * torch.pi
        elif delta_angle < -torch.pi:
            delta_angle += 2 * torch.pi


        u_speed = self.K_pd * distance - self.K_pv*current_speed
        u_angle = self.K_pa * delta_angle

        return u_speed, u_angle