from parameters import Params
import torch

class Prop_controller():
    def __init__(self):
        super(Prop_controller, self).__init__()
        self.K_pd = Params['Prop_coeff_distance']
        self.K_pa = Params['Prop_coeff_angle']
        self.K_pv = Params['Prop_coeff_speed']

    def forward(self, controller_input):
        pos=controller_input[0:2]
        kin=controller_input[4:6]
        xf=controller_input[2:4]
        


        distance = torch.sqrt((pos[0]-xf[0])**2 + (pos[1]-xf[1])**2)

        angle = torch.arctan2((xf[1]-pos[1]),(xf[0]-pos[0]))
        delta_angle = angle - kin[1]

        # Adjust the angle difference to be within [-π, π] radians
        #delta_angle = (delta_angle + torch.pi) % (2 * torch.pi) - torch.pi
        delta_angle = angle - kin[1]
        # Adjust delta_angle to be within [-π, π] radians
        if delta_angle > torch.pi:
            delta_angle -= 2 * torch.pi
        elif delta_angle < -torch.pi:
            delta_angle += 2 * torch.pi


        u_speed = self.K_pd * distance - self.K_pv*kin[0]
        u_angle = self.K_pa * delta_angle

        u_PD=torch.stack((u_speed,u_angle),0)

        return u_PD