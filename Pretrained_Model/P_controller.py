from parameters import Params
import torch

class Prop_controller():
    def __init__(self):
        super(Prop_controller, self).__init__()
        self.K_pd = Params['Prop_coeff_distance']
        self.K_pa = Params['Prop_coeff_angle']
        self.K_pv = Params['Prop_coeff_speed']

    def forward(self, controller_input):
        pos=torch.transpose(torch.transpose(controller_input,0,1)[0:2],0,1)
        kin=torch.transpose(torch.transpose(controller_input,0,1)[4:6],0,1)
        xf=torch.transpose(torch.transpose(controller_input,0,1)[2:4],0,1)

        distance = torch.sqrt((pos[:,0:1]-xf[:,0:1])**2 + (pos[:,1:2]-xf[:,1:2])**2)

        angle = torch.arctan2((xf[:,1:2]-pos[:,1:2]),(xf[:,0:1]-pos[:,0:1]))
        delta_angle = angle - kin[:,1:2]

        # Adjust the angle difference to be within [-π, π] radians
        #delta_angle = (delta_angle + torch.pi) % (2 * torch.pi) - torch.pi
        delta_angle = angle - kin[:,1:2]
        # Adjust delta_angle to be within [-π, π] radians

        delta_angle= (delta_angle + torch.pi) % (2 * torch.pi) - torch.pi

        u_speed = self.K_pd * distance - self.K_pv*kin[:,0:1]
        u_angle = self.K_pa * delta_angle

        u_PD=torch.cat((u_speed,u_angle),1)

        return u_PD